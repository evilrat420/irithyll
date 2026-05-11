//! Application state shared between the TUI renderer and the training thread.
//!
//! State is wrapped in a [`parking_lot::RwLock`] so the renderer (read-heavy,
//! ~10 Hz) does not contend with the training thread (write-heavy, hot path).

use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

use irithyll::DriftSignal;

/// Quantiles tracked simultaneously for the cyclable pinball-loss display.
///
/// Asymmetric quantiles (0.1, 0.9) surface under-prediction vs over-prediction
/// risk separately. Median (0.5) is mathematically equivalent to ½·MAE, kept
/// for completeness. Cycle via `[` / `]` in the Metrics tab.
pub const PINBALL_QUANTILES: &[f64] = &[0.1, 0.25, 0.5, 0.75, 0.9];

/// Default pinball quantile index (q=0.1 — under-prediction loss). Chosen so
/// the default differs visually from MAE; the user can cycle to others with `[` / `]`.
pub const PINBALL_DEFAULT_IDX: usize = 0;

/// Model family selectable from the TUI demo. Each family maps to a different
/// `irithyll` factory function (SGBT trees, Mamba SSM, TTT fast weights, KAN
/// B-splines, ESN reservoir, NG-RC polynomial features, SpikeNet e-prop, or
/// linear SGD). Cycled with `f` / `F`; the change is announced in the footer
/// and applies on the next demo run — switching mid-training would mean
/// rebuilding the model and dropping the metric history, which is a v10.1
/// concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Streaming gradient-boosted trees (Hoeffding base learners).
    Sgbt,
    /// Selective state-space model (Mamba V1 default).
    Mamba,
    /// Test-time training with Titans-style fast weights.
    Ttt,
    /// Kolmogorov-Arnold network with B-spline edges.
    Kan,
    /// Echo State Network — fixed-weight reservoir + RLS readout.
    Esn,
    /// Next-generation reservoir computing — delay-line polynomial features.
    Ngrc,
    /// Spiking neural network with e-prop learning (SNN, fixed-point core).
    SpikeNet,
    /// Streaming linear regression with SGD.
    Linear,
}

impl ModelFamily {
    /// All families in cycle order — drives the `f` / `F` rotation.
    pub const ALL: &'static [ModelFamily] = &[
        ModelFamily::Sgbt,
        ModelFamily::Mamba,
        ModelFamily::Ttt,
        ModelFamily::Kan,
        ModelFamily::Esn,
        ModelFamily::Ngrc,
        ModelFamily::SpikeNet,
        ModelFamily::Linear,
    ];

    /// Short label used in the header and Diagnostics tab section titles.
    pub fn label(self) -> &'static str {
        match self {
            ModelFamily::Sgbt => "SGBT",
            ModelFamily::Mamba => "Mamba",
            ModelFamily::Ttt => "TTT",
            ModelFamily::Kan => "KAN",
            ModelFamily::Esn => "ESN",
            ModelFamily::Ngrc => "NG-RC",
            ModelFamily::SpikeNet => "SpikeNet",
            ModelFamily::Linear => "Linear",
        }
    }

    /// Cycle to the next family.
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&f| f == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Cycle to the previous family.
    pub fn prev(self) -> Self {
        let i = Self::ALL.iter().position(|&f| f == self).unwrap_or(0);
        Self::ALL[(i + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    /// True if this family has a natural per-input attribution we can compute
    /// from publicly-exposed state. SGBT uses split-gain importances, KAN
    /// aggregates first-layer spline coefficient magnitudes, Linear uses
    /// `|w_i|`. The reservoir / SSM / spiking families project all inputs
    /// through internal state and don't expose per-feature attribution.
    pub fn supports_feature_importance(self) -> bool {
        matches!(
            self,
            ModelFamily::Sgbt | ModelFamily::Kan | ModelFamily::Linear
        )
    }
}

/// Active tab in the right-hand panel. Cycled with `Tab` / `Shift+Tab`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    /// Cyclable metrics chart — loss EMA, R², directional accuracy, pinball, MAE.
    Metrics,
    /// Per-model diagnostic signals.
    Diagnostics,
    /// Top-K feature importances.
    Importances,
}

impl Tab {
    /// All tabs in cycle order.
    pub const ALL: [Tab; 3] = [Tab::Metrics, Tab::Diagnostics, Tab::Importances];

    /// Short label rendered in the tab strip.
    pub fn label(self) -> &'static str {
        match self {
            Tab::Metrics => "Metrics",
            Tab::Diagnostics => "Diagnostics",
            Tab::Importances => "Importances",
        }
    }

    /// Cycle to the next tab.
    pub fn next(self) -> Self {
        let idx = Self::ALL.iter().position(|t| *t == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    /// Cycle to the previous tab.
    pub fn prev(self) -> Self {
        let idx = Self::ALL.iter().position(|t| *t == self).unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// Which metric series the Metrics tab is currently displaying. Cycled with
/// the `m` / `M` key shortcut. Loss-specific behaviors (drift markers, log
/// y-axis) only apply when [`MetricKind::LossEma`] is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Exponential moving average of squared loss.
    LossEma,
    /// Coefficient of determination (regression goodness-of-fit).
    R2,
    /// Accuracy — classification accuracy in eval mode, sign-of-residual
    /// directional accuracy vs running mean in the regression demo.
    Accuracy,
    /// Pinball loss at quantile q = 0.5 (equivalent to 0.5·MAE).
    PinballLoss,
    /// Running mean absolute error.
    Mae,
}

impl MetricKind {
    /// All metrics in cycle order.
    pub const ALL: &'static [MetricKind] = &[
        MetricKind::LossEma,
        MetricKind::R2,
        MetricKind::Accuracy,
        MetricKind::PinballLoss,
        MetricKind::Mae,
    ];

    /// Long label used as chart title.
    pub fn label(self) -> &'static str {
        match self {
            MetricKind::LossEma => "Loss (EMA)",
            MetricKind::R2 => "R² (Coefficient of Determination)",
            MetricKind::Accuracy => "Accuracy",
            MetricKind::PinballLoss => "Pinball Loss (q=0.5)",
            MetricKind::Mae => "MAE",
        }
    }

    /// Cycle to the next metric.
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&k| k == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// Cycle to the previous metric.
    pub fn prev(self) -> Self {
        let i = Self::ALL.iter().position(|&k| k == self).unwrap_or(0);
        Self::ALL[(i + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// A drift event observed during training, used to draw a vertical marker on
/// the loss chart.
#[derive(Debug, Clone, Copy)]
pub struct DriftEvent {
    /// Sample index at which the event was observed.
    pub sample_index: u64,
    /// Severity of the drift signal.
    pub signal: DriftSignal,
}

/// Image cache for the Metrics tab.
///
/// Plotters renders the chart at ~5 Hz (200 ms throttle) instead of every
/// terminal frame. We cache the decoded `DynamicImage` so the plotters
/// rasterization is amortized — `ratatui-image` re-encodes the cached bitmap
/// to the active terminal protocol (sixel / kitty / iterm2) on each render,
/// which is comparatively cheap.
///
/// Invalidation triggers (any one):
/// * `now - last_render > 200 ms`
/// * `active_metric != last_metric` (user pressed `m`/`M`)
/// * `n_samples - last_n_samples > N_TOTAL / 50` (significant data growth)
/// * `show_drift_overlay` differs (user pressed `d`)
/// * `log_scale` differs (user pressed `l` and metric is loss)
/// * `drift_events.len() != last_drift_count` (new drift event recorded)
///
/// Owned by [`AppState`] so the renderer can regen + render in one critical
/// section.
pub struct ImageCache {
    /// Wall-clock instant of the last regen.
    pub last_render: std::time::Instant,
    /// Cached image — `None` before first regen, or after invalidation.
    pub image: Option<image::DynamicImage>,
    /// Sample count at last regen — drives the "data growth" trigger.
    pub last_n_samples: u64,
    /// Active metric at last regen — drives the "user switched view" trigger.
    pub last_metric: Option<MetricKind>,
    /// Whether to draw orange vertical drift markers (toggled by `d`).
    pub show_drift_overlay: bool,
    /// Log scale at last regen — drives invalidation on `l` toggle.
    pub last_log_scale: bool,
    /// Drift count at last regen — invalidates when new drift events arrive.
    pub last_drift_count: usize,
    /// Y-axis lower bound at last regen — invalidates when auto-ranging
    /// shifts the Y window so the cached geometry no longer matches the
    /// ratatui-rendered tick labels.
    pub last_y_lo: f64,
    /// Y-axis upper bound at last regen — see `last_y_lo`.
    pub last_y_hi: f64,
    /// X-axis upper bound at last regen — invalidates when sample count
    /// growth shifts the X axis enough that the cached image is stale.
    pub last_x_max: f64,
}

impl ImageCache {
    /// Default state — no cached image, regen on first frame.
    pub fn new() -> Self {
        Self {
            last_render: std::time::Instant::now(),
            image: None,
            last_n_samples: 0,
            last_metric: None,
            show_drift_overlay: false,
            last_log_scale: false,
            last_drift_count: 0,
            last_y_lo: 0.0,
            last_y_hi: 0.0,
            last_x_max: 0.0,
        }
    }
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks training progress and metrics for TUI display.
pub struct AppState {
    /// Number of samples processed so far.
    pub n_samples: u64,
    /// Total number of samples to process.
    pub n_total: u64,
    /// Current metric snapshots: `(name, value)`.
    pub metrics: Vec<(String, f64)>,
    /// Rolling history of loss EMA values (one push per update interval).
    pub loss_history: Vec<f64>,
    /// Rolling history of R² values.
    pub r2_history: Vec<f64>,
    /// Rolling history of directional-accuracy values (sign agreement vs running mean).
    pub accuracy_history: Vec<f64>,
    /// Rolling history of pinball loss values, indexed by quantile slot.
    /// Slots correspond to [`PINBALL_QUANTILES`]. The Metrics tab plots
    /// `pinball_history[active_pinball_q_idx]`; the user cycles which
    /// quantile is displayed via the `[` / `]` keys.
    pub pinball_history: Vec<Vec<f64>>,
    /// Index into [`PINBALL_QUANTILES`] for the currently-displayed pinball curve.
    pub active_pinball_q_idx: usize,
    /// Rolling history of mean absolute error values.
    pub mae_history: Vec<f64>,
    /// Feature importances: `(feature_name, importance)`.
    pub feature_importances: Vec<(String, f64)>,
    /// Current throughput in samples per second.
    pub throughput: f64,
    /// Wall-clock seconds elapsed since training started.
    pub elapsed_secs: f64,
    /// Whether training is currently in progress.
    pub is_training: bool,
    /// Whether training has completed.
    pub is_done: bool,
    /// Whether training is paused (renderer keeps drawing; training thread idles).
    pub is_paused: bool,
    /// Whether the loss chart should use log10 scale on the y-axis.
    pub log_scale: bool,
    /// Currently active right-panel tab.
    pub active_tab: Tab,
    /// Currently displayed metric on the Metrics tab. Cycled with `m`/`M`.
    pub active_metric: MetricKind,
    /// Whether the help overlay is shown.
    pub show_help: bool,
    /// Status message displayed in the footer.
    pub status_message: String,
    /// Dataset identity rendered in the header — bench name (e.g. "friedman")
    /// for the no-subcommand demo, CSV filename for `train --tui` /
    /// `eval --tui`. Empty until the training loop sets it.
    pub dataset_label: String,
    /// Drift events observed during training, marked on the loss chart.
    pub drift_events: Vec<DriftEvent>,

    // -- Diagnostics --
    /// Total tree replacements across all boosting steps.
    pub total_replacements: u64,
    /// Compact diagnostic signals: [residual_alignment, reg_sensitivity,
    /// depth_sufficiency, effective_dof, uncertainty].
    pub diagnostics_array: [f64; 5],
    /// Honest sigma (DistributionalSGBT only).
    pub honest_sigma: f64,
    /// Currently active model family — drives factory selection and
    /// per-family vital-signs / diagnostics rendering. Set at startup; cycle
    /// hint is announced via [`AppState::status_message`] on `f` / `F`.
    pub active_family: ModelFamily,
    /// Model type string for conditional diagnostics display.
    ///
    /// Derived from [`AppState::active_family`] at update-interval ticks; kept
    /// as an owned `String` so the per-frame renderer never has to format the
    /// label under a write lock. Updated by the training loop alongside the
    /// family-specific diagnostic rows.
    pub model_type: String,
    /// Rich structured diagnostics rows for the Diagnostics tab.
    ///
    /// Each entry is `(label, value, color_class)` where `color_class` is one of
    /// `"good"`, `"neutral"`, `"warn"`, or `"error"`. Section-header rows have an
    /// empty `value` string and `color_class` `"neutral"`. Populated by the training
    /// loop; empty until the first diagnostics refresh.
    pub diagnostic_rows: Vec<(String, String, String)>,

    /// Image cache for the Metrics tab — regenerated at ~5 Hz, not every frame.
    ///
    /// Holds the most recent plotters-rendered chart bitmap so the renderer
    /// can hand it to `ratatui-image` for sixel/kitty/iterm2 embedding
    /// without paying the rasterization cost on every terminal redraw.
    ///
    /// Wrapped in a separate [`parking_lot::Mutex`] so the renderer can hold
    /// the cache lock independently of the [`AppState`] `RwLock`. This is the
    /// crux of the lock decoupling: plotters cache regen runs lock-free on a
    /// snapshot, then publishes via this `Mutex` without ever taking
    /// `AppState::write()` — meaning the training thread's per-sample writes
    /// only contend with brief `read()` windows.
    pub image_cache: Mutex<ImageCache>,
}

impl AppState {
    /// Create a new state for a training run of `n_total` samples.
    pub fn new(n_total: u64) -> Self {
        Self {
            n_samples: 0,
            n_total,
            metrics: Vec::new(),
            loss_history: Vec::new(),
            r2_history: Vec::new(),
            accuracy_history: Vec::new(),
            pinball_history: PINBALL_QUANTILES.iter().map(|_| Vec::new()).collect(),
            active_pinball_q_idx: PINBALL_DEFAULT_IDX,
            mae_history: Vec::new(),
            feature_importances: Vec::new(),
            throughput: 0.0,
            elapsed_secs: 0.0,
            is_training: true,
            is_done: false,
            is_paused: false,
            log_scale: false,
            active_tab: Tab::Metrics,
            active_metric: MetricKind::LossEma,
            show_help: false,
            status_message: String::from("Initializing..."),
            dataset_label: String::new(),
            drift_events: Vec::new(),
            total_replacements: 0,
            diagnostics_array: [0.0; 5],
            honest_sigma: 0.0,
            active_family: ModelFamily::Sgbt,
            model_type: String::new(),
            diagnostic_rows: Vec::new(),
            image_cache: Mutex::new(ImageCache::new()),
        }
    }

    /// Progress as a percentage (0.0 to 100.0).
    pub fn progress_pct(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.n_samples as f64 / self.n_total as f64 * 100.0
        }
    }

    /// Progress as a ratio (0.0 to 1.0), clamped for `LineGauge`.
    pub fn progress_ratio(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            (self.n_samples as f64 / self.n_total as f64).min(1.0)
        }
    }

    /// Estimated time remaining in seconds, based on current throughput.
    /// Returns `None` if throughput is zero or training is done.
    pub fn eta_secs(&self) -> Option<f64> {
        if self.is_done || self.throughput <= 0.0 || self.n_samples >= self.n_total {
            return None;
        }
        let remaining = self.n_total.saturating_sub(self.n_samples) as f64;
        Some(remaining / self.throughput)
    }

    /// Format ETA as a human-readable string.
    pub fn eta_display(&self) -> String {
        match self.eta_secs() {
            Some(secs) if secs < 60.0 => format!("{:.0}s", secs),
            Some(secs) if secs < 3600.0 => {
                format!("{}m {:.0}s", (secs / 60.0) as u64, secs % 60.0)
            }
            Some(secs) => {
                let h = (secs / 3600.0) as u64;
                let m = ((secs % 3600.0) / 60.0) as u64;
                format!("{}h {}m", h, m)
            }
            None => "--".to_string(),
        }
    }

    /// Bucket-mean downsample of the FULL loss history into `n` slots, scaled
    /// to `0..100` against the all-time max.
    ///
    /// Two design choices that matter:
    /// 1. We downsample the whole history (not just the last N samples) so the
    ///    sparkline shows the entire learning arc — the descending trend is the
    ///    point. A trailing window normalized to its own min/max amplifies tiny
    ///    bumps and obscures the drop the model achieved early on.
    /// 2. We normalize to the all-time max (not min/max), so when loss
    ///    converges to a small value, bars genuinely shrink toward the floor.
    pub fn sparkline_data(&self, n: usize) -> Vec<u64> {
        if self.loss_history.is_empty() || n == 0 {
            return Vec::new();
        }

        let history = &self.loss_history;
        let buckets: Vec<f64> = if history.len() <= n {
            history.clone()
        } else {
            let bucket_size = history.len().div_ceil(n);
            history
                .chunks(bucket_size)
                .map(|c| c.iter().sum::<f64>() / c.len() as f64)
                .collect()
        };

        let max = buckets
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(f64::EPSILON);

        buckets.iter().map(|v| ((v / max) * 100.0) as u64).collect()
    }

    /// Record a drift event observed during training.
    pub fn record_drift(&mut self, signal: DriftSignal) {
        self.drift_events.push(DriftEvent {
            sample_index: self.n_samples,
            signal,
        });
    }
}

/// Thread-safe handle to the shared application state.
///
/// Reads (renderer, ~10 Hz) and writes (training thread, every sample) share an
/// `RwLock` to avoid serializing the hot path through a single mutex.
pub type SharedState = Arc<RwLock<AppState>>;
