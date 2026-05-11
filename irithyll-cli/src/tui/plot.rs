//! Sixel-backed academic-quality chart rendering for the Metrics tab.
//!
//! Renders a [`MetricKind`] series to an RGB image via `plotters`. The image
//! is handed to `ratatui-image`, which embeds it inline using whichever
//! protocol the terminal speaks (sixel on Windows Terminal 1.22+, Kitty
//! graphics on Kitty/Konsole, iTerm2 inline on iTerm2, half-blocks fallback
//! elsewhere). The renderer is purely about pixels — no ratatui types leak in
//! — so this module also serves as a "headless" chart utility usable for
//! exporting plots to disk in the future.
//!
//! ## Hybrid pipeline
//!
//! As of the hybrid refactor, this module renders ONLY the chart geometry
//! (axis lines, gridlines, the data curve, drift overlay). All TEXT —
//! title, Y-axis labels, X-axis labels — is rendered natively by ratatui in
//! `mod.rs`. Mixing plotters' bitmap-rasterized text with the terminal's
//! native font produced a visible quality gap; native text is sharp at every
//! DPI and respects the terminal's font preferences.
//!
//! The Y/X bounds and the per-metric label formatter are computed in `mod.rs`
//! using the [`compute_y_bounds`], [`compute_x_max`], and [`y_label_formatter`]
//! helpers exposed here, then handed back into [`PlotConfig`] so the geometry
//! and the labels stay coordinated.
//!
//! ## Anti-aliased line
//!
//! Plotters' `LineSeries` with a 2px `ShapeStyle` produces an anti-aliased
//! line. No marker dots — the line is the visual element. Drift events
//! (loss only) overlay as 1px solid orange vertical lines.

use image::{DynamicImage, RgbImage};
use plotters::prelude::*;

use crate::tui::app::{AppState, DriftEvent, MetricKind};

/// Configuration passed in from the renderer at every regen.
///
/// Pixel dimensions only — bounds, log-scale, and the active series live on
/// [`MetricsSnapshot`], which is constructed under a brief `AppState::read()`
/// lock and then handed off lock-free to [`render_metric_from_snapshot`].
pub struct PlotConfig {
    /// Width of the rendered image in pixels.
    pub width: u32,
    /// Height of the rendered image in pixels.
    pub height: u32,
}

/// Detached copy of the [`AppState`] fields the plotters renderer needs to
/// produce the metrics chart. Constructed under a brief `AppState::read()`
/// lock (see `cache_regen_pass` in `mod.rs`), then handed lock-free to
/// [`render_metric_from_snapshot`] so the heavy CPU work (rasterization +
/// 2x supersample + Lanczos3 downsample, ~5–20 ms) never holds the lock.
///
/// Cloning cost is bounded — each history vector caps at one entry per
/// update interval (~200 entries over a 20k-sample run); drift events are
/// similarly bounded. Total clone work per regen is well under 1 ms.
pub struct MetricsSnapshot {
    /// Active metric kind (drives label, axis, and which history is plotted).
    pub kind: MetricKind,
    /// Whether the chart should render in log10 space.
    pub log_scale: bool,
    /// Sample count at the moment the snapshot was taken — used to position
    /// drift event markers on the X axis.
    pub n_samples: u64,
    /// Upper X bound (sample count) for the cartesian plane. Lower is 0.
    pub x_max: f64,
    /// Lower Y bound for the cartesian plane.
    pub y_lo: f64,
    /// Upper Y bound for the cartesian plane.
    pub y_hi: f64,
    /// Owned copy of the active metric's history series.
    pub history: Vec<f64>,
    /// Owned copy of the drift events list.
    pub drift_events: Vec<DriftEvent>,
    /// Whether to draw orange vertical drift markers (loss view only).
    pub show_drift_overlay: bool,
}

impl MetricsSnapshot {
    /// Build a snapshot from `&AppState` plus the resolved log/drift flags.
    ///
    /// Called by `cache_regen_pass` while holding `AppState::read()`. Clones
    /// the active history series and drift events so the snapshot can outlive
    /// the lock.
    pub fn from_state(state: &AppState, log_scale: bool, show_drift_overlay: bool) -> Self {
        let kind = state.active_metric;
        let pinball_slot = state
            .pinball_history
            .get(state.active_pinball_q_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let history: Vec<f64> = match kind {
            MetricKind::LossEma => state.loss_history.clone(),
            MetricKind::R2 => state.r2_history.clone(),
            MetricKind::Accuracy => state.accuracy_history.clone(),
            MetricKind::PinballLoss => pinball_slot.to_vec(),
            MetricKind::Mae => state.mae_history.clone(),
        };
        let (y_lo, y_hi) = compute_y_bounds(state, log_scale);
        let x_max = compute_x_max(state);
        Self {
            kind,
            log_scale,
            n_samples: state.n_samples,
            x_max,
            y_lo,
            y_hi,
            history,
            drift_events: state.drift_events.clone(),
            show_drift_overlay,
        }
    }
}

// --- Theme colors mirrored from `theme.rs` ---
// We keep these as `RGBColor` constants instead of importing `theme.rs` to
// avoid a circular feature gate (`plot.rs` is plotters-only, `theme.rs` is
// ratatui-only). The values must stay in sync with `theme.rs` — both are
// scientific palette choices that change rarely.
const C_BASE: RGBColor = RGBColor(0, 0, 0);
const C_GRID: RGBColor = RGBColor(40, 40, 48);
const C_AXIS: RGBColor = RGBColor(150, 155, 180);
const C_LINE: RGBColor = RGBColor(57, 255, 20); // GREEN
const C_DRIFT: RGBColor = RGBColor(255, 130, 30); // PEACH

/// Render the active metric series to a `DynamicImage`.
///
/// Returns `Err` if the plotters backend fails (e.g. zero-sized buffer). The
/// caller is expected to fall back to the native ratatui Chart in that case.
///
/// In the hybrid pipeline this draws ONLY the chart geometry — axes,
/// gridlines, the data curve, drift overlay. Text (title, axis labels) is
/// rendered by ratatui in `mod.rs`.
///
/// Operates entirely on the detached [`MetricsSnapshot`] — no `AppState`
/// access, so this function runs lock-free. The caller (cache_regen_pass)
/// builds the snapshot under a brief read lock and hands it in here.
pub fn render_metric_from_snapshot(
    snapshot: &MetricsSnapshot,
    cfg: &PlotConfig,
) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    if cfg.width == 0 || cfg.height == 0 {
        return Err("zero-sized canvas".into());
    }

    let kind = snapshot.kind;
    let is_loss = matches!(kind, MetricKind::LossEma);
    let log_scale = snapshot.log_scale;
    let history: &[f64] = &snapshot.history;

    // Apply log10 transform for the loss view if requested. Floor at 1e-12 to
    // keep `log10` finite on degenerate zero/negative entries.
    let transform = |v: f64| -> f64 {
        if log_scale {
            v.max(1e-12).log10()
        } else {
            v
        }
    };

    // Build (x, y) pairs in *sample-index* coordinates. The history vector
    // is a downsampled snapshot of the per-sample signal; we map history
    // index `i` to sample index `i * n_samples / n_history` so the X axis
    // is always in real samples-seen units. We then bucket-mean down to
    // ~800 points to keep the anti-aliased curve crisp regardless of
    // history length.
    let max_points = 800; // ~screen-width pixel budget at 1080p
    let series: Vec<(f64, f64)> = downsample(history, snapshot.x_max, max_points)
        .into_iter()
        .map(|(x, y)| (x, transform(y)))
        .collect();

    // --- 2x supersampled backbuffer ---
    //
    // Plotters draws into a 2x-resolution buffer, then we downsample with
    // Lanczos3 to the target dimensions. Sixel can only encode 256 colors per
    // image, so the gradient anti-aliasing along the line edge gets quantized
    // — at native res those few intermediate shades create visible banding,
    // looking "chunky." At 2x the line edge has more pixels carrying the
    // intermediate intensity, and Lanczos averaging dilutes the quantization
    // band into a smooth gradient before sixel quantizes the final image.
    //
    // Cost: ~3x render time per regen. Cached at 50ms throttle so per-frame
    // amortizes to negligible.
    const SUPERSAMPLE: u32 = 2;
    let render_w = cfg.width.saturating_mul(SUPERSAMPLE).max(1);
    let render_h = cfg.height.saturating_mul(SUPERSAMPLE).max(1);
    let mut buf = vec![0u8; (render_w as usize) * (render_h as usize) * 3];
    {
        let backend = BitMapBackend::with_buffer(&mut buf, (render_w, render_h));
        let root = backend.into_drawing_area();
        root.fill(&C_BASE)?;

        // No caption, no label areas — ratatui owns text. Margin scales with
        // supersample factor so the visible margin stays 1px after downsample.
        let mut chart = ChartBuilder::on(&root)
            .margin(SUPERSAMPLE as i32)
            .set_label_area_size(LabelAreaPosition::Left, 0)
            .set_label_area_size(LabelAreaPosition::Bottom, 0)
            .build_cartesian_2d(0.0_f64..snapshot.x_max, snapshot.y_lo..snapshot.y_hi)?;

        chart
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .disable_x_axis()
            .disable_y_axis()
            .axis_style(ShapeStyle::from(&C_AXIS).stroke_width(SUPERSAMPLE))
            .light_line_style(C_GRID.mix(0.3))
            .bold_line_style(C_GRID.mix(0.6))
            .x_label_formatter(&|_| String::new())
            .y_label_formatter(&|_| String::new())
            .draw()?;

        // Drift overlay (loss view only). Drawn BEFORE the line so the line
        // sits on top.
        if snapshot.show_drift_overlay && is_loss {
            for ev in &snapshot.drift_events {
                if let Some(x) = drift_x_position(ev, snapshot.n_samples, snapshot.x_max) {
                    chart.draw_series(LineSeries::new(
                        [(x, snapshot.y_lo), (x, snapshot.y_hi)],
                        ShapeStyle::from(&C_DRIFT).stroke_width(SUPERSAMPLE),
                    ))?;
                }
            }
        }

        // Data series — 2px nominal stroke ⇒ 4px at 2x supersample.
        chart.draw_series(LineSeries::new(
            series,
            ShapeStyle::from(&C_LINE).stroke_width(2 * SUPERSAMPLE),
        ))?;

        root.present()?;
    }

    // Wrap the supersampled RGB buffer, then Lanczos3-resize to target.
    let big = RgbImage::from_raw(render_w, render_h, buf)
        .expect("buffer length matches render_w * render_h * 3 by construction");
    let big_dyn = DynamicImage::ImageRgb8(big);
    let resized =
        big_dyn.resize_exact(cfg.width, cfg.height, image::imageops::FilterType::Lanczos3);
    Ok(resized)
}

/// Mean-bucket downsample to at most `max_points` points, mapping bucket
/// centers from history-index space onto the `[0, x_max]` sample-index axis.
///
/// The training thread pushes one history entry per update interval, so the
/// history is a strided sample of the underlying per-sample signal. We
/// linearly remap so the chart's X axis reads in real samples-seen.
fn downsample(history: &[f64], x_max: f64, max_points: usize) -> Vec<(f64, f64)> {
    if history.is_empty() {
        return Vec::new();
    }
    let n_hist = history.len();
    let scale = x_max / n_hist as f64;

    if n_hist <= max_points {
        return history
            .iter()
            .enumerate()
            .map(|(i, v)| ((i as f64 + 0.5) * scale, *v))
            .collect();
    }
    let bucket = n_hist.div_ceil(max_points);
    history
        .chunks(bucket)
        .enumerate()
        .map(|(bi, chunk)| {
            let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            // Bucket center in history-index space, then scaled to sample space.
            let center = bi * bucket + chunk.len() / 2;
            (center as f64 * scale, mean)
        })
        .collect()
}

/// Compute the upper X bound (sample count) for the chart. Lower is always 0.
///
/// Falls back to history length when `n_samples == 0` so the axes still draw
/// before the training loop has reported its first batch.
pub fn compute_x_max(state: &AppState) -> f64 {
    let history_len = match state.active_metric {
        MetricKind::LossEma => state.loss_history.len(),
        MetricKind::R2 => state.r2_history.len(),
        MetricKind::Accuracy => state.accuracy_history.len(),
        MetricKind::PinballLoss => state.pinball_history.len(),
        MetricKind::Mae => state.mae_history.len(),
    };
    if state.n_samples > 0 {
        state.n_samples as f64
    } else {
        history_len.max(1) as f64
    }
}

/// Compute Y bounds for a metric. Unit-range metrics are pinned; others get
/// data range + 10% padding (range floored at 0.001 to avoid degenerate plots).
///
/// Exposed publicly so `mod.rs` can drive both the geometry render AND the
/// ratatui-rendered Y-axis tick labels from a single source of truth.
pub fn compute_y_bounds(state: &AppState, log_scale: bool) -> (f64, f64) {
    let kind = state.active_metric;
    match kind {
        // Unit-bounded metrics: linear bounds [0, 1]; log10 bounds [-3, 0]
        // (covering 0.001 to 1.0 — enough decades to surface convergence
        // detail near the ceiling without the line plotting off-canvas).
        MetricKind::R2 if log_scale => (-3.0, 0.0),
        MetricKind::Accuracy if log_scale => (-3.0, 0.0),
        MetricKind::R2 => (0.0, 1.0),
        MetricKind::Accuracy => (0.0, 1.0),
        _ => {
            // LossEma / PinballLoss / MAE — auto-range from data with 10% pad.
            // Empty series gets a small symmetric range so axes still draw.
            let pinball_slot = state
                .pinball_history
                .get(state.active_pinball_q_idx)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let history: &[f64] = match kind {
                MetricKind::LossEma => &state.loss_history,
                MetricKind::PinballLoss => pinball_slot,
                MetricKind::Mae => &state.mae_history,
                _ => unreachable!(),
            };
            if history.is_empty() {
                return if log_scale { (-3.0, 0.0) } else { (0.0, 1.0) };
            }
            let transform = |v: f64| -> f64 {
                if log_scale {
                    v.max(1e-12).log10()
                } else {
                    v
                }
            };
            let mut y_min = f64::INFINITY;
            let mut y_max = f64::NEG_INFINITY;
            for v in history.iter() {
                let t = transform(*v);
                if t < y_min {
                    y_min = t;
                }
                if t > y_max {
                    y_max = t;
                }
            }
            // Defensive — all-NaN history would leave both at infinity.
            if !y_min.is_finite() || !y_max.is_finite() {
                return if log_scale { (-3.0, 0.0) } else { (0.0, 1.0) };
            }
            let range = (y_max - y_min).max(0.001);
            let pad = range * 0.1;
            (y_min - pad, y_max + pad)
        }
    }
}

/// Pin axis label format per metric — fixes the "1.0500" precision bug.
///
/// Exposed so ratatui can render Y-axis tick labels with the same per-metric
/// formatter the plotters renderer used to use.
pub fn y_label_formatter(
    kind: MetricKind,
    log_scale: bool,
    y_lo: f64,
    y_hi: f64,
) -> Box<dyn Fn(f64) -> String> {
    if log_scale {
        // Log10 axis — labels are exponents. `1e{n}` reads cleaner than `10^n`.
        return Box::new(|v: f64| format!("1e{:.0}", v));
    }
    match kind {
        MetricKind::R2 | MetricKind::Accuracy => Box::new(|v: f64| format!("{:.2}", v)),
        // Loss / Pinball / MAE: auto-pick precision by magnitude. The
        // threshold is "max abs > 100" — beyond that, decimals are noise.
        _ => {
            let max_abs = y_lo.abs().max(y_hi.abs());
            if max_abs >= 100.0 {
                Box::new(|v: f64| format!("{:.0}", v))
            } else if max_abs >= 10.0 {
                Box::new(|v: f64| format!("{:.2}", v))
            } else {
                Box::new(|v: f64| format!("{:.3}", v))
            }
        }
    }
}

/// Format X-axis tick (sample count). Compact for large numbers.
pub fn format_x_label(v: f64) -> String {
    let n = v as i64;
    if n >= 1_000_000 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}k", v / 1_000.0)
    } else {
        format!("{}", n)
    }
}

/// Map a drift event's sample index to chart X coordinate, returning `None`
/// if the event lies outside the visible range or the chart is empty.
fn drift_x_position(ev: &DriftEvent, n_samples: u64, x_max: f64) -> Option<f64> {
    if n_samples == 0 || x_max <= 0.0 {
        return None;
    }
    // Drift events also need to be filtered by signal type. Stable signals
    // are recorded but never drawn — they're noise on the chart.
    if matches!(ev.signal, irithyll::DriftSignal::Stable) {
        return None;
    }
    let x = ev.sample_index as f64;
    if !x.is_finite() || x < 0.0 || x > x_max {
        return None;
    }
    Some(x)
}
