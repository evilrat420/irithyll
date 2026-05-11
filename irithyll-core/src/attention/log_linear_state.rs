//! Hierarchical Fenwick-tree state for Log-Linear Attention.
//!
//! Implements the per-head state container for Log-Linear Attention
//! (Han Guo et al., ICLR 2026, arXiv:2506.04761). Each head owns a stack
//! of up to `max_levels` matrix states, organized as a binary-counter
//! (Fenwick) decomposition of the prefix `[0, t)`. After `t` tokens the
//! ACTIVE levels correspond exactly to the 1-bits of `t` (paper §2);
//! storage is padded to `max_levels` so the public state vector is
//! constant-shaped, satisfying the diagnostic-consumer invariant
//! "state().len() is constant" (paper §3.4 in R1, "Option B —
//! Recommended").
//!
//! # Carry-propagation algorithm
//!
//! On `push_leaf(s_leaf)`:
//! 1. Place `s_leaf` at level 0.
//! 2. While level ℓ has TWO buckets of equal size 2^ℓ, sum them into a
//!    bucket of size 2^(ℓ+1) at level ℓ+1, freeing both children.
//! 3. Continue until the carry stops or `max_levels` is exceeded.
//!
//! This is identical to incrementing a binary counter. After `t` pushes
//! the active levels are precisely the 1-bits of `t`, so the maximum
//! occupancy is `popcount(t) ≤ ⌊log₂(t)⌋ + 1` — the O(log T) state
//! bound advertised by the paper.
//!
//! # Padding to `max_levels` (NOT popcount)
//!
//! The paper-mandated stability choice (R1 §3.4): pad to a constant
//! `max_levels` so `state()` length is stable across stream length. A
//! popcount-sized vector would change shape every token, breaking the
//! `AttentionLayer::state()` contract that downstream diagnostics
//! depend on. Inactive levels are zero matrices.
//!
//! # `max_levels` capacity
//!
//! `max_levels = ⌊log₂(T_max)⌋ + 1`. For T_max = 2^32 (4 billion
//! tokens), `max_levels = 33`. The recommended default is **32**,
//! matching R1 §3.5: covers streams up to ~4 G tokens with constant
//! overhead `max_levels * d_k * d_v` per head.

use alloc::vec;
use alloc::vec::Vec;

use super::state::AttentionState;

/// Hierarchical stack of matrix states, one per active Fenwick level.
///
/// Storage is fixed at `max_levels` slots; each slot is a `d_k x d_v`
/// matrix (zeros when inactive). The `active` mask records which slots
/// currently hold a real bucket. The `size` field counts tokens pushed
/// so far — equivalently, after `size = t` pushes, the bits of `t`
/// indicate which levels are active.
///
/// # Paper reference
///
/// Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon
/// Kim. *Log-Linear Attention*. ICLR 2026. arXiv:2506.04761, §2-§3.
#[derive(Clone, Debug)]
pub struct LogLinearState {
    /// Per-level matrix states, length `max_levels`. Each entry is a
    /// `d_k x d_v` matrix; inactive entries hold all-zero data.
    levels: Vec<AttentionState>,
    /// Active mask: `active[ℓ] == true` iff level ℓ holds a real bucket.
    /// Length `max_levels`. Equivalent to bit ℓ of `size`, but kept as a
    /// separate vector for branch-free read access in hot paths.
    active: Vec<bool>,
    /// Token count pushed so far. The bit pattern of `size` matches
    /// `active` exactly after each successful `push_leaf`.
    size: u64,
    /// Hard cap on hierarchy depth. State storage is fixed at
    /// `max_levels` regardless of `size`.
    max_levels: usize,
    /// Per-head key dimension.
    d_k: usize,
    /// Per-head value dimension.
    d_v: usize,
    /// Flat state cache: concatenated levels in row-major
    /// `[L0 | L1 | … | L_{max_levels-1}]` form, length
    /// `max_levels * d_k * d_v`. Zeroed slots remain zero.
    state_cache: Vec<f64>,
}

impl LogLinearState {
    /// Create a new state with all `max_levels` matrices zero-initialized.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `max_levels == 0`, `d_k == 0`, or
    /// `d_v == 0`.
    pub fn new(max_levels: usize, d_k: usize, d_v: usize) -> Self {
        debug_assert!(max_levels > 0, "max_levels must be positive");
        debug_assert!(d_k > 0, "d_k must be positive");
        debug_assert!(d_v > 0, "d_v must be positive");

        let levels: Vec<AttentionState> = (0..max_levels)
            .map(|_| AttentionState::new_matrix(d_k, d_v))
            .collect();
        let active = vec![false; max_levels];
        let state_cache = vec![0.0; max_levels * d_k * d_v];

        Self {
            levels,
            active,
            size: 0,
            max_levels,
            d_k,
            d_v,
            state_cache,
        }
    }

    /// Hierarchy depth cap (`max_levels`). Storage is always padded to
    /// this size.
    #[inline]
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }

    /// Per-head key dimension.
    #[inline]
    pub fn d_k(&self) -> usize {
        self.d_k
    }

    /// Per-head value dimension.
    #[inline]
    pub fn d_v(&self) -> usize {
        self.d_v
    }

    /// Number of tokens pushed so far. Equivalent to `t` in the paper.
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Number of currently active levels = `popcount(size)`.
    ///
    /// Always `≤ max_levels`. After exhausting capacity (size ≥ 2^max_levels),
    /// the highest level absorbs further carries (see `push_leaf`).
    pub fn active_level_count(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    /// Whether level `ℓ` currently holds a real bucket.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `level >= max_levels`.
    #[inline]
    pub fn is_active(&self, level: usize) -> bool {
        debug_assert!(
            level < self.max_levels,
            "level {} out of range (max_levels={})",
            level,
            self.max_levels
        );
        self.active[level]
    }

    /// Borrow level `ℓ`'s matrix state (zero matrix if inactive).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `level >= max_levels`.
    #[inline]
    pub fn level(&self, level: usize) -> &AttentionState {
        debug_assert!(
            level < self.max_levels,
            "level {} out of range (max_levels={})",
            level,
            self.max_levels
        );
        &self.levels[level]
    }

    /// Push a new leaf bucket holding the outer product `k * v^T`,
    /// then run carry-propagation upward.
    ///
    /// Algorithm (paper §2.1):
    /// 1. Set level 0 to `k * v^T`. If level 0 was already active, the
    ///    new leaf would collide — but classical Fenwick increment
    ///    means that case happens iff the previous push produced a
    ///    carry that did NOT consume level 0. By construction the
    ///    invariant holds: after every prior push, level 0 is active
    ///    iff bit 0 of `size` is set (== `size` is odd). So before
    ///    push: `level0_active iff size_was_odd`. We treat this with
    ///    standard binary-increment: place the new bucket at level 0
    ///    pre-emptively, then run the standard carry loop.
    ///
    /// In the paper this is the carry-propagation form of the Fenwick
    /// scan; in irithyll terms it's an in-place rewrite of the level
    /// stack, no allocation past `max_levels`.
    ///
    /// # Capacity overflow
    ///
    /// If a carry would propagate above level `max_levels - 1`, the
    /// excess bucket is folded into the topmost level via matrix
    /// addition. This preserves the invariant "total information
    /// captured by the Fenwick tree" at the cost of resolution at
    /// the very deepest scale — equivalent to the paper's note that
    /// `max_levels = ⌊log₂(T_max)⌋ + 1` should be chosen so
    /// `T_max` exceeds the expected stream length.
    ///
    /// # Arguments
    ///
    /// - `k` — key vector, length `d_k`.
    /// - `v` — value vector, length `d_v`.
    pub fn push_leaf(&mut self, k: &[f64], v: &[f64]) {
        debug_assert_eq!(k.len(), self.d_k, "k length must match d_k");
        debug_assert_eq!(v.len(), self.d_v, "v length must match d_v");

        // Sanity: classical binary-counter increment makes level 0
        // collisions impossible when invariants hold; assert in debug.
        // Specifically, before this push, level 0 active <=> size is
        // odd. After push, level 0 active <=> (size+1) is odd.
        debug_assert_eq!(
            self.active[0],
            self.size & 1 == 1,
            "Fenwick invariant: level 0 active iff size is odd"
        );

        // The new leaf must enter at level 0. If level 0 is active
        // (i.e., size was odd), classical binary increment carries up
        // — but in the matrix interpretation, the "carry" means the
        // existing level-0 bucket sums with the new leaf and is then
        // written to level 1, then potentially summing with level 1's
        // existing bucket, and so on, until we hit an inactive level.

        // Build the new bucket as outer product (k * v^T).
        let mut carry = AttentionState::new_matrix(self.d_k, self.d_v);
        carry.add_outer_product(k, v);

        let mut ell = 0usize;
        loop {
            if ell >= self.max_levels {
                // Capacity exhausted: fold the carry into the topmost
                // level (max_levels - 1). This caps memory at the
                // configured bound while still accumulating information.
                let top = self.max_levels - 1;
                add_matrix_in_place(&mut self.levels[top], &carry);
                self.active[top] = true;
                break;
            }

            if !self.active[ell] {
                // Slot is free — write the carry here, halt.
                replace_matrix(&mut self.levels[ell], carry);
                self.active[ell] = true;
                break;
            }

            // Slot ℓ is active: sum the existing bucket into carry
            // and clear ℓ. Continue propagation upward.
            let existing = take_matrix(&mut self.levels[ell], self.d_k, self.d_v);
            self.active[ell] = false;
            add_matrix_in_place(&mut carry, &existing);
            ell += 1;
        }

        self.size = self.size.saturating_add(1);
        self.refresh_cache();
    }

    /// Reset all levels to zero and clear `size`. After reset,
    /// `state()` returns all zeros and `active_level_count() == 0`.
    pub fn reset(&mut self) {
        for state in self.levels.iter_mut() {
            state.reset();
        }
        for a in self.active.iter_mut() {
            *a = false;
        }
        self.size = 0;
        for x in self.state_cache.iter_mut() {
            *x = 0.0;
        }
    }

    /// Flat view of the padded state — concatenation of all
    /// `max_levels` levels in row-major order.
    ///
    /// Length is always `max_levels * d_k * d_v`, regardless of
    /// `active_level_count()`. Inactive levels contribute all-zero
    /// blocks. This is the constant-shape contract required by
    /// `AttentionLayer::state()` consumers.
    #[inline]
    pub fn flat_state(&self) -> &[f64] {
        &self.state_cache
    }

    /// Compute the λ-weighted readout `Σ_ℓ λ_ℓ · q^T · S^(ℓ)` over all
    /// `max_levels` slots and write into `out` (length `d_v`).
    ///
    /// Inactive levels contribute zero (their `S^(ℓ)` is the zero
    /// matrix). The caller supplies `lambdas` of length `max_levels`
    /// (typically a softplus-softmax mix bounding `Σ λ ≤ 1`).
    ///
    /// # Arguments
    ///
    /// - `q` — query vector, length `d_k`.
    /// - `lambdas` — per-level non-negative mix weights, length
    ///   `max_levels`.
    /// - `out` — output buffer, length `d_v`. Overwritten.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `q.len() != d_k`,
    /// `lambdas.len() != max_levels`, or `out.len() != d_v`.
    pub fn query_mixed(&self, q: &[f64], lambdas: &[f64], out: &mut [f64]) {
        debug_assert_eq!(q.len(), self.d_k, "q length must match d_k");
        debug_assert_eq!(
            lambdas.len(),
            self.max_levels,
            "lambdas length must match max_levels"
        );
        debug_assert_eq!(out.len(), self.d_v, "out length must match d_v");

        for o in out.iter_mut() {
            *o = 0.0;
        }
        for (ell, &lam) in lambdas.iter().enumerate() {
            if !self.active[ell] || lam == 0.0 {
                continue;
            }
            // Per-level readout: o_ℓ = q^T · S^(ℓ) (length d_v).
            let o_l = self.levels[ell].query(q);
            for (oi, ol) in out.iter_mut().zip(o_l.iter()) {
                *oi += lam * ol;
            }
        }
    }

    /// Refresh the flat cache from the level matrices. Cheap: total
    /// work is `max_levels * d_k * d_v` per token, equal to the
    /// log-linear state size already advertised.
    fn refresh_cache(&mut self) {
        let mut offset = 0;
        for state in self.levels.iter() {
            let slice = state.as_slice();
            let len = slice.len();
            self.state_cache[offset..offset + len].copy_from_slice(slice);
            offset += len;
        }
    }
}

/// In-place matrix add: `dst += src` (both `d_k x d_v` row-major).
fn add_matrix_in_place(dst: &mut AttentionState, src: &AttentionState) {
    match (dst, src) {
        (
            AttentionState::Matrix { data: dst_data, .. },
            AttentionState::Matrix { data: src_data, .. },
        ) => {
            debug_assert_eq!(
                dst_data.len(),
                src_data.len(),
                "matrix addition shape mismatch"
            );
            for (d, s) in dst_data.iter_mut().zip(src_data.iter()) {
                *d += *s;
            }
        }
        _ => panic!("add_matrix_in_place: both states must be Matrix"),
    }
}

/// Move `src` into `*dst`, leaving `dst` holding the new bucket.
/// Equivalent to assignment but uses the existing buffer of `dst`
/// when possible to avoid alloc churn — copies element-wise.
fn replace_matrix(dst: &mut AttentionState, src: AttentionState) {
    match (dst, src) {
        (
            AttentionState::Matrix { data: dst_data, .. },
            AttentionState::Matrix { data: src_data, .. },
        ) => {
            debug_assert_eq!(
                dst_data.len(),
                src_data.len(),
                "matrix replace shape mismatch"
            );
            dst_data.copy_from_slice(&src_data);
        }
        _ => panic!("replace_matrix: both states must be Matrix"),
    }
}

/// Read out the existing matrix at `dst` into a new owned
/// `AttentionState`, leaving `dst` zeroed in place. Avoids a swap by
/// copying then zeroing — the old data is preserved in the returned
/// state.
fn take_matrix(dst: &mut AttentionState, d_k: usize, d_v: usize) -> AttentionState {
    let mut taken = AttentionState::new_matrix(d_k, d_v);
    if let (
        AttentionState::Matrix { data: dst_data, .. },
        AttentionState::Matrix {
            data: taken_data, ..
        },
    ) = (dst, &mut taken)
    {
        taken_data.copy_from_slice(dst_data);
        for d in dst_data.iter_mut() {
            *d = 0.0;
        }
    } else {
        panic!("take_matrix: state must be Matrix");
    }
    taken
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_has_zero_size_and_no_active_levels() {
        let s = LogLinearState::new(8, 4, 4);
        assert_eq!(s.size(), 0, "fresh state has size 0");
        assert_eq!(
            s.active_level_count(),
            0,
            "fresh state has no active levels"
        );
        assert!(
            s.flat_state().iter().all(|&x| x == 0.0),
            "fresh state cache is all zeros"
        );
    }

    #[test]
    fn log_linear_state_padded_to_max_levels() {
        // The flat state slice MUST equal max_levels * d_k * d_v
        // regardless of how many tokens have been pushed. This is the
        // paper-mandated stability choice (R1 §3.4 Option B).
        let max_levels = 8;
        let d_k = 4;
        let d_v = 4;
        let mut s = LogLinearState::new(max_levels, d_k, d_v);
        let expected_len = max_levels * d_k * d_v;
        assert_eq!(
            s.flat_state().len(),
            expected_len,
            "flat state must be max_levels * d_k * d_v at t=0"
        );

        // Push one token: should add a leaf at level 0.
        s.push_leaf(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5, 0.25, -0.25]);
        assert_eq!(
            s.flat_state().len(),
            expected_len,
            "flat state must remain max_levels * d_k * d_v after t=1"
        );
        assert_eq!(s.size(), 1);
        assert_eq!(s.active_level_count(), 1, "popcount(1) = 1");
        assert!(s.is_active(0), "after 1 push, level 0 is active");

        // Push three more tokens (size = 4 = 0b100), expect only
        // level 2 active (popcount = 1).
        for i in 0..3 {
            let f = (i + 1) as f64;
            s.push_leaf(&[f, f, f, f], &[f, f, f, f]);
        }
        assert_eq!(s.size(), 4);
        assert_eq!(s.active_level_count(), 1, "popcount(4) = 1");
        assert!(s.is_active(2), "size=4 -> level 2 active");
        assert!(!s.is_active(0));
        assert!(!s.is_active(1));
        assert_eq!(
            s.flat_state().len(),
            expected_len,
            "flat state still padded to max_levels"
        );
    }

    #[test]
    fn log_linear_state_reset_clears_all_levels() {
        let max_levels = 8;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        for i in 0..50u64 {
            let f = i as f64 + 1.0;
            s.push_leaf(&[f, f, f, f], &[f, f, f, f]);
        }
        assert!(s.size() > 0);
        assert!(s.active_level_count() > 0);
        assert!(
            s.flat_state().iter().any(|&x| x != 0.0),
            "after pushes, cache should have non-zero entries"
        );

        s.reset();

        assert_eq!(s.size(), 0, "reset clears size");
        assert_eq!(s.active_level_count(), 0, "reset deactivates all levels");
        assert!(
            s.flat_state().iter().all(|&x| x == 0.0),
            "reset clears flat state"
        );
        for ell in 0..max_levels {
            assert!(
                !s.is_active(ell),
                "level {} must be inactive after reset",
                ell
            );
            assert!(
                s.level(ell).as_slice().iter().all(|&x| x == 0.0),
                "level {} matrix must be zero after reset",
                ell
            );
        }
    }

    #[test]
    fn fenwick_active_levels_match_popcount_of_size() {
        // After t pushes, the active levels MUST equal the 1-bits of
        // t (Han Guo et al., ICLR 2026 §2). Verify across t = 1..32.
        let max_levels = 8;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        let k = [0.5; 4];
        let v = [0.5; 4];

        for t in 1..=31u64 {
            s.push_leaf(&k, &v);
            for ell in 0..max_levels {
                let bit_set = (t >> ell) & 1 == 1;
                assert_eq!(
                    s.is_active(ell),
                    bit_set,
                    "at size={}, level {} active should match bit {} of size",
                    t,
                    ell,
                    ell
                );
            }
            assert_eq!(
                s.active_level_count() as u32,
                t.count_ones(),
                "active count must equal popcount of size"
            );
        }
    }

    #[test]
    fn level_matrix_size_doubles_with_level() {
        // After 2^k tokens with all-equal leaves, the merged bucket at
        // level k is the SUM of 2^k identical outer products, i.e., the
        // outer-product magnitude at level k is 2^k times the single
        // leaf magnitude. This verifies the merge semantics
        // (matrix addition of equal-size siblings, paper §2.1).
        let max_levels = 8;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        let k_vec = [1.0, 0.0, 0.0, 0.0];
        let v_vec = [1.0, 0.0, 0.0, 0.0];

        // Push exactly 4 = 2^2 tokens. Only level 2 should be active,
        // and its (0,0) element should be 4 (outer product (k * v^T) at
        // (0,0) = 1, summed 4 times).
        for _ in 0..4 {
            s.push_leaf(&k_vec, &v_vec);
        }
        assert_eq!(s.size(), 4);
        assert_eq!(s.active_level_count(), 1);
        assert!(s.is_active(2));
        let entry = s.level(2).get_matrix(0, 0);
        assert!(
            (entry - 4.0).abs() < 1e-12,
            "level 2 (0,0) should accumulate 4 leaves, got {}",
            entry
        );
    }

    #[test]
    fn query_mixed_zero_lambdas_gives_zero_output() {
        let max_levels = 8;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        s.push_leaf(&[1.0, 2.0, 3.0, 4.0], &[0.5, 0.5, 0.5, 0.5]);

        let q = [1.0; 4];
        let lambdas = [0.0; 8];
        let mut out = [42.0; 4];
        s.query_mixed(&q, &lambdas, &mut out);
        for &o in &out {
            assert_eq!(o, 0.0, "zero λ produces zero output");
        }
    }

    #[test]
    fn query_mixed_uniform_lambdas_sums_active_levels() {
        // With λ = 1.0 on all levels, output equals the unweighted
        // sum of per-level queries (only active levels contribute).
        let max_levels = 8;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        let k = [1.0, 0.0, 0.0, 0.0];
        let v = [1.0, 1.0, 1.0, 1.0];
        s.push_leaf(&k, &v); // level 0: k * v^T

        let q = [1.0, 0.0, 0.0, 0.0];
        let lambdas = [1.0; 8];
        let mut out = [0.0; 4];
        s.query_mixed(&q, &lambdas, &mut out);
        // S^(0) at (0,*) = v = [1,1,1,1]; S^T q at index j = sum_i S[i][j] * q[i] = S[0][j]*1 = v[j].
        for &o in &out {
            assert!(
                (o - 1.0).abs() < 1e-12,
                "uniform λ readout should equal v, got {}",
                o
            );
        }
    }

    #[test]
    fn query_mixed_inactive_levels_skipped() {
        // After 2 pushes (size=2 = 0b10), only level 1 is active.
        // λ on inactive levels must contribute exactly zero.
        let max_levels = 4;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        s.push_leaf(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
        s.push_leaf(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
        assert!(s.is_active(1));
        assert!(!s.is_active(0));
        assert!(!s.is_active(2));

        let q = [1.0, 0.0, 0.0, 0.0];
        // Compare:
        //   - All λ=1: only level 1 contributes
        //   - λ=1 only on level 0 (inactive): output should be zero.
        let mut out_all = [0.0; 4];
        s.query_mixed(&q, &[1.0; 4], &mut out_all);

        let mut out_inactive = [0.0; 4];
        s.query_mixed(&q, &[1.0, 0.0, 0.0, 0.0], &mut out_inactive);
        for &o in &out_inactive {
            assert_eq!(
                o, 0.0,
                "λ on inactive level 0 must contribute zero (level 0 is empty), got {}",
                o
            );
        }

        // The "all λ=1" output should be non-zero (level 1 has 2-leaf
        // accumulated bucket).
        assert!(
            out_all.iter().any(|&o| o != 0.0),
            "active level 1 with λ=1 must contribute non-zero output"
        );
    }

    #[test]
    fn capacity_overflow_folds_into_top_level() {
        // With max_levels=2, after 4 pushes the carry would propagate
        // to level 2 (out of range). Spec: fold into top level
        // (max_levels - 1 = 1).
        let max_levels = 2;
        let mut s = LogLinearState::new(max_levels, 4, 4);
        let k = [1.0, 0.0, 0.0, 0.0];
        let v = [1.0, 0.0, 0.0, 0.0];
        for _ in 0..4 {
            s.push_leaf(&k, &v);
        }
        assert_eq!(s.size(), 4);
        // Top level should hold the accumulated information.
        assert!(s.is_active(1), "top level must be active after overflow");
        let entry = s.level(1).get_matrix(0, 0);
        assert!(
            entry > 0.0,
            "top level should accumulate folded carries, got {}",
            entry
        );
    }

    #[test]
    fn flat_state_matches_concatenated_levels() {
        let max_levels = 4;
        let d_k = 3;
        let d_v = 3;
        let mut s = LogLinearState::new(max_levels, d_k, d_v);
        for i in 0..7u64 {
            let f = (i + 1) as f64 * 0.1;
            s.push_leaf(&[f, f, f], &[f, f, f]);
        }
        // Size = 7 = 0b111: levels 0, 1, 2 active.
        let flat = s.flat_state();
        assert_eq!(flat.len(), max_levels * d_k * d_v);
        let block = d_k * d_v;
        for ell in 0..max_levels {
            let level_slice = s.level(ell).as_slice();
            let cache_slice = &flat[ell * block..(ell + 1) * block];
            assert_eq!(
                level_slice, cache_slice,
                "flat cache for level {} must match level matrix",
                ell
            );
        }
    }

    #[test]
    fn deterministic_construction() {
        let mut a = LogLinearState::new(8, 4, 4);
        let mut b = LogLinearState::new(8, 4, 4);
        for t in 1..=20u64 {
            let f = t as f64 * 0.1;
            a.push_leaf(&[f, f, f, f], &[f, -f, f, -f]);
            b.push_leaf(&[f, f, f, f], &[f, -f, f, -f]);
        }
        for (x, y) in a.flat_state().iter().zip(b.flat_state().iter()) {
            assert!(
                (x - y).abs() < 1e-15,
                "identical pushes produce identical state"
            );
        }
    }
}
