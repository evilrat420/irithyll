//! Version-specific readout feature construction for Mamba.

use super::StreamingMamba;

/// Dispatch version-specific readout feature building.
pub(super) fn build_readout_features(
    model: &StreamingMamba,
    gated_output: &[f64],
    state: &[f64],
    raw_input: &[f64],
) -> Vec<f64> {
    use crate::ssm::mamba_config::MambaVersion;
    match model.config.version {
        MambaVersion::V1 => build_readout_features_v1(model, gated_output, state),
        // V3 (Tustin MIMO-lite) and V3Mimo: gated output + per-group Frobenius energy.
        MambaVersion::V3 | MambaVersion::V3Mimo { .. } => {
            build_readout_features_v3(model, gated_output, state)
        }
        // V3Exp: base features + tanh random feature lift over [base ; raw_input].
        // The random tanh projection lifts the linear RLS readout to a Random
        // Feature Network (Rahimi & Recht 2008): linear regression on `n_lift`
        // random nonlinear features approximates kernel ridge regression with
        // the Gaussian RBF kernel. This is mathematically necessary because
        // the V3Exp BASE features (gated output, per-group energy, per-component
        // Re/Im/|h|/|h|^2) reach at most degree-4 in input bits, which a
        // closed-form sufficiency probe shows is insufficient for k-bit XOR
        // with `k > 4`. The Gaussian RBF kernel is a universal approximator;
        // the lift makes V3Exp's complex modes USEFUL by composing them with
        // a high-degree-polynomial-capable readout. V1 stays linear because
        // V1's claim is "real scalar SSM cannot express parity" — the V3Exp
        // arm carries the lift, V1 does not.
        MambaVersion::V3Exp { .. } => {
            build_readout_features_v3exp(model, gated_output, state, raw_input)
        }
        MambaVersion::BlockDiagonal { block_size } => {
            build_readout_features_bd(model, gated_output, state, block_size)
        }
    }
}

/// V1 readout features: gated SSM output (`d_in`) + per-channel state energy (`d_in`).
///
/// The gated output (SSM output ⊗ SiLU gate + residual) provides the
/// primary signal after content-dependent filtering. The per-channel
/// state energy (L2 norm of each channel's `n_state`-dimensional hidden
/// state vector) captures nonlinear temporal activation magnitude that
/// the linear C projection may not fully represent.
fn build_readout_features_v1(
    model: &StreamingMamba,
    gated_output: &[f64],
    state: &[f64],
) -> Vec<f64> {
    let d_in = model.config.d_in;
    let n_state = model.config.n_state;
    let mut rf = Vec::with_capacity(d_in * 2);

    // Primary: gated SSM output (C-projected + gate + residual)
    rf.extend_from_slice(gated_output);

    // Secondary: per-channel state energy (L2 norm of each channel's state vector).
    for d in 0..d_in {
        let offset = d * n_state;
        let energy: f64 = state[offset..offset + n_state]
            .iter()
            .map(|&s| s * s)
            .sum::<f64>();
        rf.push(energy.sqrt());
    }

    rf
}

/// V3 readout features: gated SSM output (`d_in`) + per-group state energy (`n_groups`).
///
/// For Mamba-3, the state is organized into `n_groups` MIMO groups with complex
/// values. The per-group energy is the L2 norm over all state elements in that group.
fn build_readout_features_v3(
    model: &StreamingMamba,
    gated_output: &[f64],
    state: &[f64],
) -> Vec<f64> {
    let d_in = model.config.d_in;
    let n_groups = model.config.n_groups;
    let mut rf = Vec::with_capacity(d_in + n_groups);

    // Primary: gated SSM output
    rf.extend_from_slice(gated_output);

    // Secondary: per-group state energy.
    let per_group = state.len().checked_div(n_groups).unwrap_or(0);

    for g in 0..n_groups {
        let group_start = g * per_group;
        let group_end = (group_start + per_group).min(state.len());
        let group_slice = if group_start < state.len() {
            &state[group_start..group_end]
        } else {
            &[]
        };
        let energy: f64 = group_slice.iter().map(|&s| s * s).sum::<f64>();
        rf.push(energy.sqrt());
    }

    rf
}

/// V3Exp readout features.
///
/// Layout (`d_in + 3 * n_groups + 4 * n_groups * n_state`):
/// - `d_in` gated output (C-projected linear-in-state, phase-sensitive).
/// - `n_groups` per-group L2 Frobenius energy `sqrt(Σ_n |h_n|^2)`.
/// - `n_groups` per-group L4 modulus squared `Σ_n |h_n|^4` — the
///   load-bearing higher-order amplitude feature for parity tasks.
/// - `n_groups` per-group max modulus squared `max_n |h_n|^2` — surface
///   the dominant complex-state mode without averaging.
/// - `4 * n_groups * n_state` per-component features for every (g, n):
///   `Re(h)`, `Im(h)`, `|h|`, `|h|^2`.
///
/// V3Exp's `SelectiveSSMv3Exp` cell maintains a complex-diagonal hidden
/// state of length `2 * n_groups * n_state` interleaved as
/// `[re, im, re, im, ...]` (Lahoti et al., ICLR 2026, §3). Per state
/// component `n` in group `g`:
///
/// ```text
/// h_idx = (g * n_state + n) * 2  // re part
/// h_idx + 1                       // im part
/// |h_{g,n}| = sqrt(re^2 + im^2)   // modulus (phase-invariant amplitude)
/// ```
///
/// The C projection used by the gated SSM output is a learned linear mixer
/// over the real-axis component only, so the gated output is phase-sensitive
/// and cannot express phase-invariant amplitude patterns on its own. The
/// existing per-group Frobenius energy collapses every component in a group
/// into one scalar, losing per-component discriminative information AND
/// caps at 2nd-order in the cell input — too low for parity-class tasks
/// over many bits.
///
/// **Why mod^2 and L4-norm matter for parity.** The V3Exp cell pre-projects
/// each step's input with a *bilinear* `bx = (W_B · x) * mean(x_group)`
/// (Lahoti et al. §3.1), so `bx` is already 2nd-order in the input bits.
/// `|h_n|^2 = re^2 + im^2` then squares this, giving up-to-4th-order
/// products of input bits per component. `Σ_n |h_n|^4` squares once more,
/// reaching up-to-8th-order products — precisely what 8-bit XOR parity
/// needs to be linearly separable in the readout's feature space. Pure
/// linear-in-state surfaces (`Re(h)`, `Im(h)`, gated output) cap at
/// 2nd-order by construction and cannot express 8th-order parity.
///
/// **What each feature contributes.**
/// - `Re(h)`, `Im(h)` — phase-sensitive linear-in-state. Full access to
///   the complex state.
/// - `|h|` — saturating monotone amplitude (sqrt of |h|^2). Useful for
///   the RLS posterior because the sqrt has a different gradient profile
///   than the squared form, and the linear head picks the better-scaled
///   feature.
/// - `|h|^2` — cleanest 4th-order-in-input feature.
/// - Per-group `Σ_n |h_n|^4` — 8th-order-in-input aggregate, the
///   parity-class feature that lifts the model above linear-readout
///   chance for full-bit XOR.
/// - Per-group `max_n |h_n|^2` — picks out the dominant complex mode
///   without the smoothing that average-energy applies; preserves
///   discrimination on streams where one mode dominates.
///
/// **Bounded-feature invariant.** `SelectiveSSMv3Exp` enforces
/// `|A_bar|^2 < 1` on every eigenvalue (debug-asserted), so each `|h_n|`
/// is bounded by a finite sum of bounded inputs. `|h|^2` and `Σ |h|^4`
/// inherit boundedness from `|h|`. No tanh/clamp added — stability is
/// enforced inside the cell, not at the feature surface (per the
/// bounded-readout invariant in AGENTS.md: stability where it belongs,
/// not as a band-aid at the readout).
fn build_readout_features_v3exp(
    model: &StreamingMamba,
    gated_output: &[f64],
    state: &[f64],
    raw_input: &[f64],
) -> Vec<f64> {
    let d_in = model.config.d_in;
    let n_groups = model.config.n_groups;
    let n_state = model.config.n_state;
    let n_complex = n_groups * n_state;
    let n_lift = model.n_lift;
    let mut rf = Vec::with_capacity(d_in + n_groups + 4 * n_complex + n_lift);

    // Primary: gated SSM output (C-projected + gate + residual, phase-sensitive).
    rf.extend_from_slice(gated_output);

    // Secondary: per-group L2 energy (Frobenius over the whole group state slice).
    let per_group = state.len().checked_div(n_groups).unwrap_or(0);
    for g in 0..n_groups {
        let group_start = g * per_group;
        let group_end = (group_start + per_group).min(state.len());
        let group_slice = if group_start < state.len() {
            &state[group_start..group_end]
        } else {
            &[][..]
        };
        let energy: f64 = group_slice.iter().map(|&s| s * s).sum::<f64>();
        rf.push(energy.sqrt());
    }

    // Tertiary: per-component complex-state surfacing. For every (g, n), emit
    // four features: Re(h_{g,n}), Im(h_{g,n}), |h_{g,n}|, |h_{g,n}|^2.
    //
    // - Re/Im (linear-in-state): phase-sensitive features, give the readout
    //   full access to the complex state.
    // - |h| (sqrt of |h|^2): saturating modulus, monotone in amplitude.
    // - |h|^2 (modulus squared): cleanest quadratic-in-state feature.
    //   Algebraically `|h|^2 = re^2 + im^2`, which expands into pairwise
    //   products of state components. Through the bilinear cell pre-projection
    //   `bx = (W_B·x) * mean(x_group)` (Lahoti et al. §3.1), `|h|^2` carries
    //   up-to-4th-order products of the input bits. Pure linear-in-state
    //   projections cannot express this.
    //
    // State layout: interleaved [re_{g0,n0}, im_{g0,n0}, re_{g0,n1}, im_{g0,n1}, ...].
    let base_end_before_complex = rf.len();
    for g in 0..n_groups {
        for n in 0..n_state {
            let re_idx = (g * n_state + n) * 2;
            let im_idx = re_idx + 1;
            let re = state.get(re_idx).copied().unwrap_or(0.0);
            let im = state.get(im_idx).copied().unwrap_or(0.0);
            let mod_sq = re * re + im * im;
            rf.push(re);
            rf.push(im);
            rf.push(mod_sq.sqrt());
            rf.push(mod_sq);
        }
    }
    debug_assert_eq!(
        rf.len(),
        base_end_before_complex + 4 * n_complex,
        "V3Exp tertiary block size mismatch"
    );

    // Quaternary: tanh random-feature lift over the raw input bits (rescaled
    // to `{-1, +1}`).
    //
    // Rationale (random feature theory, Rahimi & Recht 2008): linear regression
    // on `n_lift` random nonlinear features approximates kernel ridge regression
    // with the Gaussian RBF kernel as `n_lift → ∞`. The Gaussian kernel is a
    // universal approximator, so lifted-linear regression solves any smooth
    // target — including k-bit XOR for `k = d_in`. Without the lift, V3Exp's
    // BASE features (max degree 4 in bits) cannot express XOR for `k > 4`
    // (closed-form sufficiency probe: 0.49 acc on 8-bit parity).
    //
    // Why tanh rather than cosine: tanh is bounded in `[-1, 1]` exactly, which
    // satisfies the bounded-readout invariant naturally. RFF-cos has the same
    // bound but `cos` doesn't go through the origin — the symmetric tanh keeps
    // the lift centered when the base features are mean-zero.
    //
    // Why raw input only (not base features ++ input): the closed-form probe
    // showed that mixing temporally-smoothed base features into the projection
    // dilutes the bit signal (V3Exp+base+input lift: 0.50 batch-LS; pure-input
    // lift: 1.00). For i.i.d. inputs the temporal smoothing inside the SSM is
    // noise, not signal. Base features still feed RLS directly (linear path),
    // so V3Exp retains its complex-mode information; the lift is a pure-input
    // nonlinear capability layered on top.
    //
    // Input transformation: raw_input bits in `{0, 1}` are remapped to
    // `{-1, +1}` before projection. This makes `||x_i - x_j||^2` symmetric
    // under bit-flips, which is what XOR needs.
    if let (Some(weights), Some(biases)) =
        (model.lift_weights.as_deref(), model.lift_bias.as_deref())
    {
        let lift_input_dim = model.lift_input_dim;
        debug_assert_eq!(
            lift_input_dim, d_in,
            "V3Exp lift input dim must match d_in for raw-input projection"
        );
        // Bipolar recode of raw input: bit ∈ {0, 1} → {-1, +1}. For non-binary
        // inputs this is `2x - 1`, which still preserves the centering. Pad
        // with zeros if raw_input is shorter than d_in (defensive guard).
        let mut bipolar: Vec<f64> = Vec::with_capacity(lift_input_dim);
        for &x in raw_input.iter().take(d_in) {
            bipolar.push(2.0 * x - 1.0);
        }
        while bipolar.len() < lift_input_dim {
            bipolar.push(0.0);
        }
        for (m, &bias) in biases.iter().take(n_lift).enumerate() {
            let row_start = m * lift_input_dim;
            let row = &weights[row_start..row_start + lift_input_dim];
            let dot: f64 = row.iter().zip(bipolar.iter()).map(|(w, z)| w * z).sum();
            rf.push((dot + bias).tanh());
        }
    }

    rf
}

/// BD-LRU readout features: gated SSM output (`d_in`) + per-block state energy (`n_blocks`).
///
/// State layout: `n_blocks * n_state * block_size`. Each block's state energy
/// is the L2 norm over all state elements in that block.
fn build_readout_features_bd(
    model: &StreamingMamba,
    gated_output: &[f64],
    state: &[f64],
    block_size: usize,
) -> Vec<f64> {
    let d_in = model.config.d_in;
    let n_state = model.config.n_state;
    let n_blocks = d_in / block_size;
    let block_state_size = n_state * block_size;
    let mut rf = Vec::with_capacity(d_in + n_blocks);

    // Primary: gated SSM output
    rf.extend_from_slice(gated_output);

    // Secondary: per-block state energy
    for b in 0..n_blocks {
        let start = b * block_state_size;
        let end = (start + block_state_size).min(state.len());
        let energy: f64 = state[start..end].iter().map(|&s| s * s).sum::<f64>();
        rf.push(energy.sqrt());
    }

    rf
}
