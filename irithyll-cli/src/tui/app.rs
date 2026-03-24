//! Application state shared between the TUI renderer and the training thread.

use std::sync::{Arc, Mutex};

/// Tracks training progress and metrics for TUI display.
pub struct AppState {
    /// Number of samples processed so far.
    pub n_samples: u64,
    /// Total number of samples to process.
    pub n_total: u64,
    /// Current metric snapshots: `(name, value)`.
    pub metrics: Vec<(String, f64)>,
    /// Rolling history of loss values for the chart.
    pub loss_history: Vec<f64>,
    /// Rolling history of accuracy values for the chart.
    pub accuracy_history: Vec<f64>,
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
    /// Status message displayed in the footer.
    pub status_message: String,
}

impl AppState {
    /// Create a new state for a training run of `n_total` samples.
    pub fn new(n_total: u64) -> Self {
        Self {
            n_samples: 0,
            n_total,
            metrics: Vec::new(),
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            feature_importances: Vec::new(),
            throughput: 0.0,
            elapsed_secs: 0.0,
            is_training: true,
            is_done: false,
            status_message: String::from("Initializing..."),
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

    /// Last N loss values as `u64` for the sparkline widget.
    /// Values are scaled to 0..100 range relative to the window's min/max.
    pub fn sparkline_data(&self, n: usize) -> Vec<u64> {
        let tail: &[f64] = if self.loss_history.len() <= n {
            &self.loss_history
        } else {
            &self.loss_history[self.loss_history.len() - n..]
        };

        if tail.is_empty() {
            return Vec::new();
        }

        let min = tail.iter().cloned().fold(f64::MAX, f64::min);
        let max = tail.iter().cloned().fold(0.0_f64, f64::max);
        let range = (max - min).max(f64::EPSILON);

        tail.iter()
            .map(|v| ((v - min) / range * 100.0) as u64)
            .collect()
    }
}

/// Thread-safe handle to the shared application state.
pub type SharedState = Arc<Mutex<AppState>>;
