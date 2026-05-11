//! TUI dashboard for live training/evaluation monitoring.
//!
//! Gated behind the `tui` feature. Provides a ratatui-based terminal UI
//! with a Catppuccin Mocha theme showing training progress, metrics,
//! and loss curves in real time.
//!
//! Shared state is wrapped in a [`parking_lot::RwLock`] so the renderer
//! (read-heavy, ~10 Hz) does not contend with the training thread (write
//! every update interval).

#[cfg(feature = "tui")]
mod app;
#[cfg(feature = "tui")]
pub mod demo;
#[cfg(feature = "tui")]
mod plot;
#[cfg(feature = "tui")]
mod theme;

#[cfg(feature = "tui")]
pub use app::{
    AppState, DriftEvent, ImageCache, MetricKind, ModelFamily, SharedState, Tab,
    PINBALL_DEFAULT_IDX, PINBALL_QUANTILES,
};
#[cfg(feature = "tui")]
pub use demo::{
    label_from_csv_path, refresh_family_diagnostics, run_eval_with_dataset, run_with_dataset,
    run_with_generator, DemoModel, TrainMode,
};

/// Run the TUI event loop, rendering state until the user presses 'q'.
///
/// The caller spawns this on a tokio task alongside the training loop.
/// Both sides share `state` through an `Arc<parking_lot::RwLock<AppState>>`.
///
/// Terminal state is restored via a panic hook + a Drop guard so that even
/// if the event loop panics — or the training thread panics and propagates
/// up through `await?` — the alternate screen exits and raw mode is
/// disabled. Without this, a panic mid-render leaves the terminal stuck in
/// the alternate buffer with sixel escape codes spewing as raw text.
#[cfg(feature = "tui")]
pub async fn run_tui(state: SharedState) -> color_eyre::Result<()> {
    use crossterm::{
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        ExecutableCommand,
    };
    use ratatui::prelude::*;
    use std::io::stdout;

    /// RAII guard: leaves the alternate screen + restores raw mode on Drop.
    /// Triggers on normal return AND on panic unwinding.
    struct TerminalGuard;
    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = stdout().execute(LeaveAlternateScreen);
        }
    }

    // Install a panic hook that runs the same cleanup BEFORE the default
    // hook prints the trace, so the panic message lands on a clean terminal
    // instead of inside the alt-screen buffer.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = stdout().execute(LeaveAlternateScreen);
        default_hook(info);
    }));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let _guard = TerminalGuard;

    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    event_loop(&mut terminal, &state).await
}

/// Inner event loop, factored out so cleanup always runs.
///
/// ## Lock decoupling
///
/// Three phases per loop iteration, none of which take `AppState::write()`
/// from the renderer side:
///
/// 1. **Cache regen** — [`cache_regen_pass`] briefly takes `read()` to build
///    a [`plot::MetricsSnapshot`], drops the lock, runs plotters lock-free
///    (~5–20 ms CPU), then briefly re-acquires `read()` + the
///    `image_cache` `Mutex` to publish the bitmap.
/// 2. **Render** — under a single `read()` lock for the duration of
///    `terminal.draw`. Read locks don't block other readers; the training
///    thread's `write()` only waits on the (very brief) write windows from
///    key handlers below.
/// 3. **Input poll** — 5 ms timeout (~200 Hz). Key handlers take `write()`
///    on `AppState` only when actually mutating shared fields.
async fn event_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    state: &SharedState,
) -> color_eyre::Result<()> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    use std::time::Duration;

    loop {
        // PHASE 1 — cache regen (lock-free CPU work + brief lock pings).
        cache_regen_pass(state, terminal);

        // PHASE 2 — render under a single read lock. The training thread's
        // per-sample `write()` only blocks during the brief window this
        // closure holds the lock; read locks don't serialize the hot path.
        terminal.draw(|frame| {
            let s = state.read();
            render(frame, &s);
        })?;

        // PHASE 3 — input poll. 5 ms gives ~200 Hz refresh; safe now that
        // the renderer no longer holds `AppState::write()` for plotters
        // rasterization.
        if event::poll(Duration::from_millis(5))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => break,
                    KeyCode::Char('?') => {
                        let mut s = state.write();
                        s.show_help = !s.show_help;
                    }
                    KeyCode::Char(' ') | KeyCode::Char('p') | KeyCode::Char('P') => {
                        let mut s = state.write();
                        s.is_paused = !s.is_paused;
                    }
                    KeyCode::Char('l') | KeyCode::Char('L') => {
                        let mut s = state.write();
                        s.log_scale = !s.log_scale;
                    }
                    KeyCode::Char('m') => {
                        let mut s = state.write();
                        s.active_metric = s.active_metric.next();
                    }
                    KeyCode::Char('M') => {
                        let mut s = state.write();
                        s.active_metric = s.active_metric.prev();
                    }
                    KeyCode::Char(',') => {
                        // Cycle pinball quantile down. Force cache regen so the
                        // active slot's series swaps in immediately.
                        {
                            let mut s = state.write();
                            let n_q = app::PINBALL_QUANTILES.len();
                            s.active_pinball_q_idx = (s.active_pinball_q_idx + n_q - 1) % n_q;
                        }
                        // Drop the cached image via the Mutex — separate lock,
                        // so this is independent of the `AppState` write above.
                        state.read().image_cache.lock().image = None;
                    }
                    KeyCode::Char('.') => {
                        {
                            let mut s = state.write();
                            let n_q = app::PINBALL_QUANTILES.len();
                            s.active_pinball_q_idx = (s.active_pinball_q_idx + 1) % n_q;
                        }
                        state.read().image_cache.lock().image = None;
                    }
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        // Toggle drift overlay on the Metrics tab loss view.
                        // Force cache regen this frame — `show_drift_overlay`
                        // lives on `ImageCache`, so this is purely a Mutex
                        // operation; no `AppState` write needed.
                        let s = state.read();
                        let mut cache = s.image_cache.lock();
                        cache.show_drift_overlay = !cache.show_drift_overlay;
                        cache.image = None;
                    }
                    KeyCode::Tab => {
                        let mut s = state.write();
                        s.active_tab = if key.modifiers.contains(KeyModifiers::SHIFT) {
                            s.active_tab.prev()
                        } else {
                            s.active_tab.next()
                        };
                    }
                    KeyCode::BackTab => {
                        let mut s = state.write();
                        s.active_tab = s.active_tab.prev();
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

/// Phase-1 cache regen: build the plotters bitmap off the `AppState` lock.
///
/// Pipeline:
/// 1. Compute the target plot pixel area from terminal size + Picker font
///    size. No `AppState` access yet.
/// 2. Briefly take `state.read()` + `image_cache.lock()` to evaluate
///    invalidation triggers and (if needed) construct a [`plot::MetricsSnapshot`].
///    Both locks dropped before plotters runs.
/// 3. Run [`plot::render_metric_from_snapshot`] lock-free — this is the
///    heavy CPU work (~5–20 ms at 2x supersample).
/// 4. Briefly take `state.read()` + `image_cache.lock()` again to publish
///    the new bitmap and update the cache-key fields.
///
/// The `AppState::write()` lock is **never** acquired here; training-thread
/// writes contend only with the two brief `read()` windows.
#[cfg(feature = "tui")]
fn cache_regen_pass(
    state: &SharedState,
    terminal: &ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
) {
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui_image::picker::Picker;

    // Step 1: terminal-size driven plot area derivation. Mirrors the Layout
    // math in `render_metrics_chart` so the cached image matches the area we
    // will hand to `StatefulImage`.
    let area = match terminal.size() {
        Ok(s) => ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: s.width,
            height: s.height,
        },
        Err(_) => return,
    };
    if area.width < 12 || area.height < 12 {
        return;
    }

    // Top-level: header(11) | tab-strip(1) | main(min 10) | footer(3)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11),
            Constraint::Length(1),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(area);
    let main = chunks[2];

    // Main: vital-signs panel (25%) | active tab (75%) — Metrics tab only.
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
        .split(main);
    let tab_area = main_chunks[1];

    // The Metrics tab area inside its block: subtract 2 for the border.
    if tab_area.width < 12 || tab_area.height < 4 {
        return;
    }
    let inner = ratatui::layout::Rect {
        x: tab_area.x + 1,
        y: tab_area.y + 1,
        width: tab_area.width.saturating_sub(2),
        height: tab_area.height.saturating_sub(2),
    };

    // Inner: title row(1) + body
    let outer_chunks = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(inner);
    let body = outer_chunks[1];
    // Body: y-labels(8) + canvas + right margin(2)
    let body_chunks = Layout::horizontal([
        Constraint::Length(8),
        Constraint::Min(10),
        Constraint::Length(2),
    ])
    .split(body);
    let canvas_col = body_chunks[1];
    // Canvas: plot + x-label row(1)
    let canvas_chunks =
        Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).split(canvas_col);
    let plot_area = canvas_chunks[0];

    // Pixel dimensions via terminal Picker (font metrics).
    let picker = match Picker::from_query_stdio() {
        Ok(p) => p,
        Err(_) => Picker::from_fontsize((8, 16)),
    };
    let (font_w, font_h) = picker.font_size();
    let pixel_w = (plot_area.width as u32).saturating_mul(font_w as u32);
    let pixel_h = (plot_area.height as u32).saturating_mul(font_h as u32);
    if pixel_w == 0 || pixel_h == 0 {
        return;
    }

    // Step 2: invalidation check + snapshot construction under brief `read()`.
    // We exit early without touching plotters if the cache is fresh.
    let regen_args = {
        let s = state.read();
        // Skip work entirely if the active tab isn't Metrics — Diagnostics
        // and Importances don't use the bitmap.
        if !matches!(s.active_tab, app::Tab::Metrics) {
            return;
        }
        // Skip if there's no data yet — render() will draw the "Waiting" pane.
        let history_len = match s.active_metric {
            MetricKind::LossEma => s.loss_history.len(),
            MetricKind::R2 => s.r2_history.len(),
            MetricKind::Accuracy => s.accuracy_history.len(),
            MetricKind::PinballLoss => s.pinball_history.len(),
            MetricKind::Mae => s.mae_history.len(),
        };
        if history_len == 0 {
            return;
        }

        let kind = s.active_metric;
        let log_scale = s.log_scale;
        let y_lo;
        let y_hi;
        let x_max;
        {
            let (lo, hi) = plot::compute_y_bounds(&s, log_scale);
            y_lo = lo;
            y_hi = hi;
            x_max = plot::compute_x_max(&s);
        }

        let cache = s.image_cache.lock();
        let now = std::time::Instant::now();
        let drift_n = s.drift_events.len();
        let throttle_elapsed = now.duration_since(cache.last_render).as_millis() >= 50;
        let metric_changed = cache.last_metric != Some(kind);
        let log_changed = cache.last_log_scale != log_scale;
        let drift_count_changed = cache.last_drift_count != drift_n;
        let sample_growth = s.n_samples.saturating_sub(cache.last_n_samples);
        let growth_threshold = (s.n_total / 50).max(1);
        let data_grew = sample_growth >= growth_threshold;
        let no_image = cache.image.is_none();

        let bounds_eps = 1e-9;
        let y_lo_changed = (cache.last_y_lo - y_lo).abs() > bounds_eps;
        let y_hi_changed = (cache.last_y_hi - y_hi).abs() > bounds_eps;
        let x_max_changed = (cache.last_x_max - x_max).abs() > bounds_eps;
        let bounds_changed = y_lo_changed || y_hi_changed || x_max_changed;

        let interaction_changed = no_image || metric_changed || log_changed;
        let training_active = !s.is_paused && !s.is_done;
        let training_changes = data_grew || drift_count_changed;

        let needs_regen = interaction_changed
            || bounds_changed
            || (training_active && throttle_elapsed && training_changes);

        if !needs_regen {
            return;
        }

        let show_drift = cache.show_drift_overlay;
        // Drop cache lock before snapshot construction so we don't hold both
        // locks across the clone (snapshot can take ~tens of µs on long histories).
        drop(cache);
        let snapshot = plot::MetricsSnapshot::from_state(&s, log_scale, show_drift);
        let cfg = plot::PlotConfig {
            width: pixel_w,
            height: pixel_h,
        };
        Some((snapshot, cfg, kind, log_scale, drift_n, y_lo, y_hi, x_max))
    };

    let (snapshot, cfg, kind, log_scale, drift_n, y_lo, y_hi, x_max) = match regen_args {
        Some(x) => x,
        None => return,
    };

    // Step 3: lock-free plotters work.
    let img = match plot::render_metric_from_snapshot(&snapshot, &cfg) {
        Ok(img) => img,
        Err(_) => return,
    };

    // Step 4: publish under brief `read()` + `Mutex`. No `write()` ever held.
    let s = state.read();
    let n_samples_now = s.n_samples;
    let mut cache = s.image_cache.lock();
    cache.image = Some(img);
    cache.last_render = std::time::Instant::now();
    cache.last_n_samples = n_samples_now;
    cache.last_metric = Some(kind);
    cache.last_log_scale = log_scale;
    cache.last_drift_count = drift_n;
    cache.last_y_lo = y_lo;
    cache.last_y_hi = y_hi;
    cache.last_x_max = x_max;
}

/// Render the full dashboard frame.
///
/// Takes `&AppState` (read-only) — the caller in `event_loop` holds
/// `state.read()` for the duration of `terminal.draw`. Cache regen happens
/// out-of-band in [`cache_regen_pass`] before this call, so render itself
/// never mutates `AppState` and never blocks the training thread's writes.
#[cfg(feature = "tui")]
fn render(frame: &mut ratatui::Frame, state: &app::AppState) {
    use ratatui::prelude::*;

    let area = frame.area();

    // Fill entire background with BASE color
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(theme::BASE)),
        area,
    );

    // Top-level vertical layout: header (progress bar + sparkline) | tabs | main | footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11), // Header: ASCII logo (6) + progress bar + sparkline + borders
            Constraint::Length(1),  // Tab strip
            Constraint::Min(10),    // Main
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    render_header(frame, state, chunks[0]);
    render_tab_strip(frame, state, chunks[1]);
    render_main(frame, state, chunks[2]);
    render_footer(frame, state, chunks[3]);

    if state.show_help {
        render_help_overlay(frame, area);
    }
}

/// Header: ASCII logo + stats row + progress gauge + sparkline.
#[cfg(feature = "tui")]
fn render_header(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::{prelude::*, symbols, widgets::*};

    const LOGO_RAW: &str = r#".__       .__  __  .__           .__  .__
│__│______│__│╱  │_│  │__ ___.__.│  │ │  │
│  ╲_  __ ╲  ╲   __╲  │  <   │  ││  │ │  │
│  ││  │ ╲╱  ││  │ │   Y  ╲___  ││  │_│  │__
│__││__│  │__││__│ │___│  ╱ ____││____╱____╱
                        ╲╱╲╱"#;

    // Pad each line to the longest line's display width with trailing spaces
    // so `Alignment::Center` operates on a uniform-width block and preserves
    // the ASCII art's shape.
    let max_len = LOGO_RAW
        .lines()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0);
    let logo_padded: String = LOGO_RAW
        .lines()
        .map(|l| {
            let pad = max_len.saturating_sub(l.chars().count());
            format!("{}{}", l, " ".repeat(pad))
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Run state glyph + label, color-keyed.
    let (state_glyph, state_text, state_color) = if state.is_done {
        ("■", "done", theme::GREEN)
    } else if state.is_paused {
        ("⏸", "paused", theme::YELLOW)
    } else {
        ("●", "running", theme::PEACH)
    };

    let version_label = format!(" irithyll v{} ", env!("CARGO_PKG_VERSION"));
    // Family belongs with the vital signs panel below (it labels the model
    // whose state the panel reports). Header title carries only run state +
    // version, which are universal regardless of model family.
    let mut title_spans = vec![Span::styled(
        format!(" {} {} ", state_glyph, state_text),
        Style::default()
            .fg(state_color)
            .add_modifier(Modifier::BOLD),
    )];

    if !state.dataset_label.is_empty() {
        title_spans.push(Span::styled(
            format!(" · {} · ", state.dataset_label),
            Style::default()
                .fg(theme::SUBTEXT0)
                .add_modifier(Modifier::ITALIC),
        ));
    }

    title_spans.push(Span::styled(
        version_label,
        Style::default()
            .fg(theme::SUBTEXT0)
            .add_modifier(Modifier::ITALIC),
    ));

    let block = Block::default()
        .title(Line::from(title_spans).right_aligned())
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Split inner: logo (6) | stats row (1) | gauge (1) | sparkline (rest)
    let header_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6), // ASCII logo
            Constraint::Length(1), // Stats row (4 cells)
            Constraint::Length(1), // Progress gauge (bare bar, no label)
            Constraint::Min(0),    // Sparkline section
        ])
        .split(inner);

    // -- ASCII Logo --
    let logo = Paragraph::new(logo_padded)
        .style(
            Style::default()
                .fg(theme::GREEN)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center);
    frame.render_widget(logo, header_chunks[0]);

    // -- Stats row: Samples | Throughput | Elapsed | ETA --
    let stat_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(25); 4])
        .split(header_chunks[1]);

    let render_stat =
        |frame: &mut ratatui::Frame, area: ratatui::layout::Rect, label: &str, value: String| {
            let line = Line::from(vec![
                Span::styled(format!(" {} ", label), Style::default().fg(theme::SUBTEXT0)),
                Span::styled(
                    value,
                    Style::default()
                        .fg(theme::GREEN)
                        .add_modifier(Modifier::BOLD),
                ),
            ]);
            frame.render_widget(Paragraph::new(line), area);
        };

    render_stat(
        frame,
        stat_chunks[0],
        "Samples",
        format!("{}", state.n_samples),
    );
    render_stat(
        frame,
        stat_chunks[1],
        "Throughput",
        format!("{:.0} samp/s", state.throughput),
    );
    render_stat(
        frame,
        stat_chunks[2],
        "Elapsed",
        format!("{:.1}s", state.elapsed_secs),
    );
    render_stat(frame, stat_chunks[3], "ETA", state.eta_display());

    // -- Progress bar (bare LineGauge, no inline label — stats row above carries the numbers) --
    let ratio = state.progress_ratio();
    let gauge_color = if state.is_done {
        theme::GREEN
    } else if state.is_paused {
        theme::YELLOW
    } else if ratio > 0.75 {
        theme::BLUE
    } else if ratio > 0.4 {
        theme::TEAL
    } else {
        theme::MAUVE
    };

    let gauge = LineGauge::default()
        .ratio(ratio)
        .filled_style(Style::default().fg(gauge_color))
        .unfilled_style(Style::default().fg(theme::SURFACE0))
        .line_set(symbols::line::THICK);

    frame.render_widget(gauge, header_chunks[2]);

    // -- Sparkline section: " ▎ Loss trend " label + sparkline --
    let sparkline_data = state.sparkline_data(50);
    if sparkline_data.is_empty() {
        let waiting = Paragraph::new(Span::styled(
            " ▎ Loss trend: waiting for data...",
            Style::default()
                .fg(theme::MAUVE)
                .add_modifier(Modifier::BOLD),
        ));
        frame.render_widget(waiting, header_chunks[3]);
    } else {
        let spark_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(15), // " ▎ Loss trend  " label
                Constraint::Min(10),    // Sparkline fill
            ])
            .split(header_chunks[3]);

        let label = Paragraph::new(Span::styled(
            " ▎ Loss trend  ",
            Style::default()
                .fg(theme::MAUVE)
                .add_modifier(Modifier::BOLD),
        ));
        frame.render_widget(label, spark_chunks[0]);

        let spark_max = sparkline_data.iter().copied().max().unwrap_or(1).max(1);
        let sparkline = Sparkline::default()
            .data(&sparkline_data)
            .bar_set(symbols::bar::NINE_LEVELS)
            .style(Style::default().fg(theme::GREEN).bg(theme::BASE))
            .max(spark_max);

        frame.render_widget(sparkline, spark_chunks[1]);
    }
}

/// Tab strip showing the active right-panel view.
#[cfg(feature = "tui")]
fn render_tab_strip(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    let mut spans = vec![Span::styled(" ", Style::default().fg(theme::SUBTEXT0))];
    for (i, tab) in app::Tab::ALL.iter().enumerate() {
        let style = if *tab == state.active_tab {
            Style::default()
                .fg(theme::PEACH)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::SUBTEXT0)
        };
        spans.push(Span::styled(format!("[{}] ", tab.label()), style));
        if i + 1 < app::Tab::ALL.len() {
            spans.push(Span::styled("· ", Style::default().fg(theme::SURFACE1)));
        }
    }
    let line = Line::from(spans);
    let para = Paragraph::new(line).style(Style::default().bg(theme::BASE));
    frame.render_widget(para, area);
}

/// Main area: vital-signs panel (Metrics tab only) + active-tab content.
///
/// Only the Metrics tab gets the left vital-signs column (25 %). Diagnostics
/// and Importances are content-dense and want the full width, so they bypass
/// the split and render directly into the whole `area`.
#[cfg(feature = "tui")]
fn render_main(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::prelude::*;

    if matches!(state.active_tab, app::Tab::Metrics) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
            .split(area);
        render_metrics_table(frame, state, chunks[0]);
        render_active_tab(frame, state, chunks[1]);
    } else {
        render_active_tab(frame, state, area);
    }
}

/// Right panel content driven by `state.active_tab`.
///
/// All three tabs are read-only now — the Metrics tab no longer mutates
/// `image_cache` here. Cache regen happens in [`cache_regen_pass`] before
/// the renderer is invoked; render reads the cached bitmap via the
/// `image_cache` `Mutex` (a separate lock from `AppState`).
#[cfg(feature = "tui")]
fn render_active_tab(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    match state.active_tab {
        app::Tab::Metrics => render_metrics_chart(frame, state, area),
        app::Tab::Diagnostics => render_diagnostics(frame, state, area),
        app::Tab::Importances => render_importances(frame, state, area),
    }
}

/// Left vital-signs panel on the Metrics tab — curated per model family.
///
/// Rows are chosen by [`AppState::active_family`] so each family surfaces its
/// own short list of meaningful indicators (state norm for Mamba, fast-weight
/// norm for TTT, spike rate for SpikeNet, etc.). The label/value/color tuples
/// are populated by the training loop in `main.rs::refresh_family_diagnostics`
/// and consumed here as `state.metrics` — keeping rendering free of any
/// concrete model handle and ensuring this function never has to format
/// numbers under the read lock.
#[cfg(feature = "tui")]
fn render_metrics_table(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, style::Color, widgets::*};

    // Block title carries the family identity — the panel below shows that
    // family's vital signs, so labelling the panel itself is the natural
    // home for the model name. Family is fixed per-run (set at startup via
    // `--family`), so this banner stays constant for the session.
    let block = Block::bordered()
        .title(Line::from(vec![
            Span::styled(
                format!(" {} ", state.active_family.label()),
                Style::default()
                    .fg(theme::TEAL)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "· live ",
                Style::default()
                    .fg(theme::SUBTEXT0)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    // Helper: pluck a value from `state.metrics` by name, default 0.0.
    let metric = |key: &str| -> f64 {
        state
            .metrics
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    };

    // Each family contributes its own ~3-5 vital signs. The Loss row is
    // universal — every model in the demo computes squared error and pushes
    // it into `state.metrics` under the key "Loss" — but the diagnostic rows
    // beside it are family-specific, picked from quantities that *change as
    // the model learns* and surface a structural property worth watching.
    let rows: Vec<(String, String, Color)> = match state.active_family {
        app::ModelFamily::Sgbt => {
            // SGBT trees grow and replace under drift. Loss + structural counters.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "Drift events".into(),
                    format!("{}", state.drift_events.len()),
                    theme::PEACH,
                ),
                (
                    "Replacements".into(),
                    format!("{}", state.total_replacements),
                    theme::PEACH,
                ),
            ]
        }
        app::ModelFamily::Mamba => {
            // Selective SSM. State norm shows recurrent activity; uncertainty
            // is the RLS readout's noise sigma. align EWMA tracks how often
            // prediction-change sign matches residual sign.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "State norm".into(),
                    format!("{:.4}", metric("StateNorm")),
                    theme::TEAL,
                ),
                (
                    "Align EWMA".into(),
                    format!("{:+.3}", metric("Alignment")),
                    theme::MAUVE,
                ),
                (
                    "Uncertainty".into(),
                    format!("{:.4}", metric("Uncertainty")),
                    theme::YELLOW,
                ),
            ]
        }
        app::ModelFamily::Ttt => {
            // Fast weights are the inner-loop optimizer's state; their L2
            // norm grows as the model integrates information.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "Fast w norm".into(),
                    format!("{:.4}", metric("FastNorm")),
                    theme::TEAL,
                ),
                (
                    "Output dim".into(),
                    format!("{:.0}", metric("OutputDim")),
                    theme::BLUE,
                ),
                (
                    "Uncertainty".into(),
                    format!("{:.4}", metric("Uncertainty")),
                    theme::YELLOW,
                ),
            ]
        }
        app::ModelFamily::Kan => {
            // B-spline coefficient count grows quadratic with grid_size; layer
            // sizes describe topology. Effective DoF tracks weight magnitude.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "Spline params".into(),
                    format!("{:.0}", metric("NParams")),
                    theme::TEAL,
                ),
                (
                    "Layers".into(),
                    format!("{:.0}", metric("NLayers")),
                    theme::BLUE,
                ),
                (
                    "Eff. DoF".into(),
                    format!("{:.4}", metric("EffDoF")),
                    theme::MAUVE,
                ),
            ]
        }
        app::ModelFamily::Esn => {
            // Echo state reservoir state norm + spectral radius (config) +
            // RLS readout uncertainty.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "Reservoir".into(),
                    format!("{:.0}", metric("NReservoir")),
                    theme::TEAL,
                ),
                (
                    "Spec radius".into(),
                    format!("{:.3}", metric("SpectralRadius")),
                    theme::BLUE,
                ),
                (
                    "State norm".into(),
                    format!("{:.4}", metric("StateNorm")),
                    theme::MAUVE,
                ),
            ]
        }
        app::ModelFamily::Ngrc => {
            // NG-RC is a delay-buffer + polynomial features pipeline. k/s are
            // structural; "warm" indicates the buffer has filled enough for
            // predictions to be meaningful.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                ("Delay k".into(), format!("{:.0}", metric("K")), theme::TEAL),
                ("Skip s".into(), format!("{:.0}", metric("S")), theme::BLUE),
                (
                    "Degree".into(),
                    format!("{:.0}", metric("Degree")),
                    theme::MAUVE,
                ),
                (
                    "Warm".into(),
                    if metric("Warm") > 0.5 {
                        "yes".into()
                    } else {
                        "no".into()
                    },
                    if metric("Warm") > 0.5 {
                        theme::GREEN
                    } else {
                        theme::YELLOW
                    },
                ),
            ]
        }
        app::ModelFamily::SpikeNet => {
            // SNN: hidden spike rate (fraction of neurons firing each step) is
            // the central activity signal; membrane mean tracks subthreshold
            // drive. n_hidden is the network width.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "Hidden".into(),
                    format!("{:.0}", metric("NHidden")),
                    theme::TEAL,
                ),
                (
                    "Spike rate".into(),
                    format!("{:.4}", metric("SpikeRate")),
                    theme::BLUE,
                ),
                (
                    "Membrane".into(),
                    format!("{:.4}", metric("Membrane")),
                    theme::MAUVE,
                ),
            ]
        }
        app::ModelFamily::Linear => {
            // Plain SGD linear regressor. Weight norm + bias + sample count.
            vec![
                ("Loss".into(), format!("{:.4}", metric("Loss")), theme::RED),
                (
                    "Weight norm".into(),
                    format!("{:.4}", metric("WeightNorm")),
                    theme::TEAL,
                ),
                (
                    "Bias".into(),
                    format!("{:+.4}", metric("Bias")),
                    theme::BLUE,
                ),
                (
                    "Features".into(),
                    format!("{:.0}", metric("NFeatures")),
                    theme::MAUVE,
                ),
            ]
        }
    };

    // Render as a bordered Paragraph with one Line per row.
    let lines: Vec<Line> = rows
        .into_iter()
        .map(|(label, value, color)| {
            Line::from(vec![
                Span::styled(
                    format!(" {:<14}", label),
                    Style::default().fg(theme::SUBTEXT0),
                ),
                Span::styled(
                    value,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ])
        })
        .collect();

    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, area);
}

/// Color-code metric values based on their name and magnitude.
///
/// Retained for potential use in future model-specific panels.
#[cfg(feature = "tui")]
#[allow(dead_code)]
fn color_for_metric(name: &str, value: f64) -> ratatui::style::Color {
    let lower = name.to_lowercase();

    // R-squared: >0.5 good, >0.25 mediocre, else poor
    if lower.contains("r2") || lower.contains("r_squared") || lower.contains("r²") {
        return if value > 0.5 {
            theme::GREEN
        } else if value > 0.25 {
            theme::YELLOW
        } else {
            theme::RED
        };
    }

    // Accuracy: >0.7 good, >0.4 mediocre, else poor
    if lower.contains("accuracy") || lower.contains("acc") {
        return if value > 0.7 {
            theme::GREEN
        } else if value > 0.4 {
            theme::YELLOW
        } else {
            theme::RED
        };
    }

    // Loss/error metrics: lower is better. <0.1 good, <0.5 mediocre, else poor
    if lower.contains("loss")
        || lower.contains("mse")
        || lower.contains("mae")
        || lower.contains("rmse")
        || lower.contains("error")
    {
        return if value < 0.1 {
            theme::GREEN
        } else if value < 0.5 {
            theme::YELLOW
        } else {
            theme::RED
        };
    }

    // Default: neutral green
    theme::GREEN
}

/// Bucketed-mean downsampling for line charts.
///
/// Per-sample squared losses on a streaming regressor have very high variance
/// — outliers blow out the Y-range and a min-max envelope fills the chart with
/// vertical bars that obscure the descending learning trend. Mean-per-bucket
/// produces a smooth curve that is the actual quantity readers care about.
///
/// Used only by the braille fallback [`render_metrics_chart_native`].
#[cfg(feature = "tui")]
#[allow(dead_code)]
fn downsample_mean(history: &[f64], max_points: usize) -> Vec<(f64, f64)> {
    if history.is_empty() {
        return Vec::new();
    }
    if history.len() <= max_points {
        return history
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64, *v))
            .collect();
    }

    let bucket_size = history.len().div_ceil(max_points);
    history
        .chunks(bucket_size)
        .enumerate()
        .map(|(bi, chunk)| {
            let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            (bi as f64, mean)
        })
        .collect()
}

/// Generate Y-axis labels with intermediate ticks.
///
/// Used only by the braille fallback [`render_metrics_chart_native`].
#[cfg(feature = "tui")]
#[allow(dead_code)]
fn y_axis_labels(y_min: f64, y_max: f64) -> Vec<ratatui::text::Line<'static>> {
    let range = y_max - y_min;
    if range < f64::EPSILON {
        return vec![ratatui::text::Line::from(format!("{:.4}", y_min))];
    }

    // 5 labels: min, 25%, 50%, 75%, max
    (0..=4)
        .map(|i| {
            let v = y_min + range * (i as f64 / 4.0);
            ratatui::text::Line::from(format!("{:.4}", v))
        })
        .collect()
}

/// Generate X-axis labels at 0%, 25%, 50%, 75%, 100%.
///
/// Used only by the braille fallback [`render_metrics_chart_native`].
#[cfg(feature = "tui")]
#[allow(dead_code)]
fn x_axis_labels(_x_max: f64, total_samples: u64) -> Vec<ratatui::text::Span<'static>> {
    if total_samples == 0 {
        return vec![ratatui::text::Span::from("0")];
    }
    (0..=4)
        .map(|i| {
            let sample = (total_samples as f64 * i as f64 / 4.0) as u64;
            ratatui::text::Span::from(format!("{}", sample))
        })
        .collect()
}

/// Right panel: academic-quality metrics chart, rendered via plotters into an
/// RGB image and embedded inline through `ratatui-image` (sixel on Windows
/// Terminal 1.22+, Kitty on Kitty/Konsole, iTerm2 on iTerm2/WezTerm,
/// half-blocks fallback elsewhere).
///
/// Cache strategy — see [`app::ImageCache`] for the full invalidation rule
/// set. Briefly: plotters re-rasterizes only when (a) more than 200ms has
/// passed, (b) the user toggled metric/log/drift, or (c) sample count grew
/// by more than 2% of the run total. The cached `DynamicImage` is then
/// handed to `ratatui-image` which encodes it for the terminal each frame
/// (cheap relative to plotters rasterization).
///
/// On detection failure (no terminal protocol queryable, e.g. headless or
/// piped stdout) or a plotters render error, falls back to the old ratatui
/// braille `Chart` via [`render_metrics_chart_native`].
#[cfg(feature = "tui")]
fn render_metrics_chart(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};
    use ratatui_image::{picker::Picker, Resize, StatefulImage};

    let kind = state.active_metric;
    // Log toggle now applies to every metric, not just loss. The transform
    // floors at 1e-12 so log10(0) and log10(<0) stay finite — useful for
    // surfacing late-stage convergence detail in MAE/Pinball, and for showing
    // the log-error band on accuracy near 1.0.
    let log_scale = state.log_scale;
    let log_suffix = if log_scale { " (log)" } else { "" };

    // Block title is intentionally minimal — the in-pane title row inside
    // `inner` carries the metric+sample count headline. The bordered block
    // still owns its outer border; we put no text in the corner.
    let block = Block::bordered()
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    // Empty data → display "Waiting for data..." regardless of backend.
    let history_len = match kind {
        MetricKind::LossEma => state.loss_history.len(),
        MetricKind::R2 => state.r2_history.len(),
        MetricKind::Accuracy => state.accuracy_history.len(),
        MetricKind::PinballLoss => state.pinball_history.len(),
        MetricKind::Mae => state.mae_history.len(),
    };
    if history_len == 0 {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // Detect terminal protocol once. `from_query_stdio` is the
    // standard path; on Windows it can fail (CONIN$ access) — fall back to
    // `from_fontsize` with a reasonable default so Windows Terminal still
    // gets a sixel-quality plot. If both paths fail (truly no protocol
    // available, or stdio is piped), drop to the native ratatui renderer.
    let picker = match Picker::from_query_stdio() {
        Ok(p) => p,
        Err(_) => Picker::from_fontsize((8, 16)),
    };

    // -- Layout the inner area --
    //
    // Inside the bordered block:
    //   row 0:        title (1 row, ratatui Paragraph)
    //   row 1..:      body, split horizontally:
    //                   col 0..8 :  Y-axis tick labels (ratatui)
    //                   col 8..-2:  canvas — split vertically:
    //                                 plot_area  (sixel image — geometry)
    //                                 x_label_row (ratatui Paragraphs)
    //                   col -2.. :  right margin (breathing room)
    let inner = block.inner(area);

    // Bail early if the inner area is too small to host title + body.
    if inner.width < 12 || inner.height < 4 {
        frame.render_widget(block, area);
        return;
    }

    let outer_chunks = Layout::vertical([
        Constraint::Length(1), // Title row
        Constraint::Min(3),    // Body (Y-labels + canvas + right margin)
    ])
    .split(inner);
    let title_area = outer_chunks[0];
    let body = outer_chunks[1];

    let body_chunks = Layout::horizontal([
        Constraint::Length(8), // Y-axis labels (e.g. "1.000")
        Constraint::Min(10),   // Canvas: plot + X-axis labels
        Constraint::Length(2), // Right margin (axis breathing room)
    ])
    .split(body);

    let y_label_col = body_chunks[0];
    let canvas_col = body_chunks[1];
    // body_chunks[2] is the right margin — intentionally empty.

    let canvas_chunks = Layout::vertical([
        Constraint::Min(3),    // Plot area (sixel)
        Constraint::Length(1), // X-axis labels row
    ])
    .split(canvas_col);

    let plot_area = canvas_chunks[0];
    let x_label_row = canvas_chunks[1];

    // -- Compute bounds (must match the values used in cache_regen_pass so
    // the ratatui-rendered tick labels align with the cached geometry) --
    let (y_lo, y_hi) = plot::compute_y_bounds(state, log_scale);
    let x_max = plot::compute_x_max(state);

    // -- Pixel dimensions for the sixel image --
    let (font_w, font_h) = picker.font_size();
    let pixel_w = (plot_area.width as u32).saturating_mul(font_w as u32);
    let pixel_h = (plot_area.height as u32).saturating_mul(font_h as u32);

    if pixel_w == 0 || pixel_h == 0 {
        // Plot area collapsed — render the block and bail.
        frame.render_widget(block, area);
        return;
    }

    // Cache regen lives in `cache_regen_pass`, called once per event-loop
    // iteration before this render runs. Here we only consume the cached
    // bitmap. Briefly take the `image_cache` Mutex to clone the image out
    // for `ratatui-image`; the lock is held for at most a few hundred µs.

    // -- Render block + title --
    frame.render_widget(block, area);

    // For pinball, append the active quantile so cycling gives visible feedback.
    let pinball_q_suffix = if matches!(kind, MetricKind::PinballLoss) {
        let q = app::PINBALL_QUANTILES
            .get(state.active_pinball_q_idx)
            .copied()
            .unwrap_or(0.5);
        format!(" q={}", q)
    } else {
        String::new()
    };
    let title_text = format!(
        "{}{}{} - n={}",
        kind.label(),
        pinball_q_suffix,
        log_suffix,
        state.n_samples
    );
    let title = Paragraph::new(Span::styled(
        title_text,
        Style::default()
            .fg(theme::TEXT)
            .add_modifier(Modifier::BOLD),
    ))
    .alignment(Alignment::Center);
    frame.render_widget(title, title_area);

    // -- Render Y-axis tick labels --
    //
    // Each tick is placed at the row that corresponds to its value in
    // [y_lo, y_hi] mapped to [bottom, top] of `plot_area`. The fraction
    // `frac` runs 0 (y_lo, bottom) → 1 (y_hi, top); `row_offset` is the
    // distance from the top of `plot_area`, so we subtract `frac` from 1.
    let y_fmt = plot::y_label_formatter(kind, log_scale, y_lo, y_hi);
    let n_y_ticks = 5usize;
    let plot_h = plot_area.height as i32;
    if plot_h > 0 {
        for i in 0..n_y_ticks {
            let frac = i as f64 / (n_y_ticks - 1) as f64;
            let value = y_lo + frac * (y_hi - y_lo);
            let row_offset = ((1.0 - frac) * (plot_h - 1) as f64).round() as i32;
            let y = plot_area.y as i32 + row_offset;
            // Defensive bounds check — `plot_area.y` is u16 from ratatui,
            // adding row_offset can in principle overflow if plot_h is 0.
            if y < 0 || y > u16::MAX as i32 {
                continue;
            }
            let label_rect = Rect {
                x: y_label_col.x,
                y: y as u16,
                width: y_label_col.width,
                height: 1,
            };
            // Right-align so the label ends just before the canvas edge —
            // this is how scientific plots align numeric tick labels.
            let label = Paragraph::new(Span::styled(
                y_fmt(value),
                Style::default().fg(theme::SUBTEXT0),
            ))
            .alignment(Alignment::Right);
            frame.render_widget(label, label_rect);
        }
    }

    // -- Render X-axis tick labels --
    //
    // 5 ticks evenly spaced from 0 to x_max. Each label is placed at the
    // column corresponding to its sample-count value. The first tick is
    // left-aligned to the canvas edge; the last is right-aligned so it
    // doesn't bleed into the right margin. Middle ticks are centered on
    // their column.
    let n_x_ticks = 5usize;
    let plot_w = plot_area.width as i32;
    if plot_w > 0 && x_label_row.height > 0 {
        // Reserve a small slot per tick label — keep them from colliding.
        // Width-3 each is enough for "20k" / "1.5M". For first/last, anchor
        // to the edges; for the middle three, center on their column.
        for i in 0..n_x_ticks {
            let frac = i as f64 / (n_x_ticks - 1) as f64;
            let value = frac * x_max;
            let col_offset = (frac * (plot_w - 1) as f64).round() as i32;
            let label_text = plot::format_x_label(value);
            let label_w = (label_text.chars().count() as u16).max(1);

            // Anchor: left for first tick, right for last, centered otherwise.
            let label_x = if i == 0 {
                plot_area.x
            } else if i == n_x_ticks - 1 {
                plot_area
                    .x
                    .saturating_add(plot_area.width)
                    .saturating_sub(label_w)
            } else {
                let center = plot_area.x as i32 + col_offset;
                let half = (label_w / 2) as i32;
                let raw = center - half;
                let max_x = plot_area.x as i32 + plot_area.width as i32 - label_w as i32;
                raw.clamp(plot_area.x as i32, max_x.max(plot_area.x as i32)) as u16
            };
            let label_rect = Rect {
                x: label_x,
                y: x_label_row.y,
                width: label_w,
                height: 1,
            };
            let label = Paragraph::new(Span::styled(
                label_text,
                Style::default().fg(theme::SUBTEXT0),
            ));
            frame.render_widget(label, label_rect);
        }
    }

    // -- Render the cached bitmap into plot_area --
    //
    // Brief lock on `image_cache` to clone out the cached image. The clone
    // is needed because `ratatui-image`'s `new_resize_protocol` takes
    // ownership; we cannot hand it a borrow tied to the lifetime of the
    // Mutex guard. Clone cost on a `DynamicImage` is bounded by the bitmap
    // size (~MB at 1080p) but is still under a millisecond.
    let cached_img = {
        let cache = state.image_cache.lock();
        cache.image.clone()
    };
    let Some(img) = cached_img else {
        // Cache hasn't been populated yet (first frame, or regen failed).
        // The "Waiting for data..." path above handles the empty-history
        // case; here we just leave the plot area blank for one frame and
        // expect cache_regen_pass to fill it on the next iteration.
        return;
    };
    let mut protocol = picker.new_resize_protocol(img);
    let widget = StatefulImage::default().resize(Resize::Fit(None));
    frame.render_stateful_widget(widget, plot_area, &mut protocol);
}

/// Native ratatui braille fallback for `render_metrics_chart`. Used when
/// terminal protocol detection fails (e.g. piped stdout) or when plotters
/// rasterization errors. This is the original `render_metrics_chart` impl,
/// preserved verbatim to keep behavior bit-for-bit identical in fallback mode.
///
/// Currently unreferenced: the lock-decoupled cache regen path runs
/// out-of-band and silently leaves the bitmap empty on plotters error
/// (the next regen will retry). Kept for the headless / piped-stdout case
/// where a future caller may dispatch here when `Picker::from_query_stdio`
/// fails AND the fontsize fallback is undesired.
#[cfg(feature = "tui")]
#[allow(dead_code)]
fn render_metrics_chart_native(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, symbols::Marker, widgets::*};

    let kind = state.active_metric;
    let is_loss = matches!(kind, MetricKind::LossEma);

    // (data source, fixed Y bounds). Pinball reads the active quantile slot;
    // fallback braille keeps log toggle local to loss for legacy parity.
    let pinball_slot = state
        .pinball_history
        .get(state.active_pinball_q_idx)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let (history, y_bounds_fixed): (&[f64], Option<[f64; 2]>) = match kind {
        MetricKind::LossEma => (state.loss_history.as_slice(), None),
        MetricKind::R2 => (state.r2_history.as_slice(), Some([-0.05, 1.05])),
        MetricKind::Accuracy => (state.accuracy_history.as_slice(), Some([0.0, 1.0])),
        MetricKind::PinballLoss => (pinball_slot, None),
        MetricKind::Mae => (state.mae_history.as_slice(), None),
    };

    let log_suffix = if is_loss && state.log_scale {
        " (log)"
    } else {
        ""
    };
    let title_text = format!(" {}{} ", kind.label(), log_suffix);

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            title_text,
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    if history.is_empty() {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // log10 transform only meaningful for LossEma; other metrics ignore the toggle.
    let transform = |v: f64| -> f64 {
        if is_loss && state.log_scale {
            v.max(1e-12).log10()
        } else {
            v
        }
    };
    let transformed: Vec<f64> = history.iter().copied().map(transform).collect();

    // Bucketed-mean downsampling — 100 points gives braille cells breathing room.
    let max_points = 100;
    let data = downsample_mean(&transformed, max_points);

    let x_max = data
        .iter()
        .map(|(x, _)| *x)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    let (y_lo, y_hi) = if let Some(bounds) = y_bounds_fixed {
        (bounds[0], bounds[1])
    } else {
        let y_max = data
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let range = (y_max - y_min).max(0.001);
        let padding = range * 0.15;
        (y_min - padding, y_max + padding)
    };

    // Braille fallback: line only, no drift markers. Vertical-line markers at
    // braille resolution paint over the curve they're meant to annotate. Drift
    // visualization is reserved for the plotters/sixel path where alpha-blended
    // overlay actually reads. See `plot.rs` for the high-resolution variant.
    let datasets: Vec<Dataset> = vec![Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(
            Style::default()
                .fg(theme::GREEN)
                .add_modifier(Modifier::BOLD),
        )
        .data(&data)];

    let x_labels = x_axis_labels(x_max, state.n_samples);
    let y_labels = y_axis_labels(y_lo, y_hi);

    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .labels(x_labels)
                .style(Style::default().fg(theme::SUBTEXT0)),
        )
        .y_axis(
            Axis::default()
                .bounds([y_lo, y_hi])
                .labels(y_labels)
                .style(Style::default().fg(theme::SUBTEXT0)),
        );

    frame.render_widget(chart, area);
}

/// Importances tab: horizontal bar chart of feature importances with semantic rank colors.
///
/// Color stratification (rank order, descending importance):
///   rank 0 → GREEN  (top feature)
///   rank 1 → MAUVE
///   rank 2 → BLUE
///   rank 3 → TEAL
///   rank 4+ → LAVENDER
///
/// Bar widths are already max-normalized in `main.rs` (largest = 1.0), so the
/// leading bar is always full-width. The BarChart value scale is set to 1000
/// to give fine resolution within that unit range.
#[cfg(feature = "tui")]
fn render_importances(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    // Filter out noise-floor features (< 5% of max importance) so the chart
    // shows the model's actual signal, not split-gain quirks on irrelevant
    // features. Friedman is a clean test: 5 causal + 5 noise features, the
    // noise should converge to ≈0 importance over training.
    let max_imp = state
        .feature_importances
        .iter()
        .map(|(_, v)| *v)
        .fold(0.0_f64, f64::max);
    let threshold = (max_imp * 0.05).max(1e-9);
    let n_signal = state
        .feature_importances
        .iter()
        .filter(|(_, v)| *v >= threshold)
        .count();
    let n_total = state.feature_importances.len();
    let title_text = format!(
        " Feature Importances — {} of {} features above noise floor ",
        n_signal, n_total
    );
    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            title_text,
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    if state.feature_importances.is_empty() {
        let msg = if state.active_family.supports_feature_importance() {
            "Waiting for data...".to_string()
        } else {
            format!(
                "Per-feature attribution not exposed for {}.\nUse SGBT, KAN, or Linear for feature importance.",
                state.active_family.label()
            )
        };
        let empty = Paragraph::new(msg)
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // AppState already stores pairs sorted descending and max-normalized;
    // re-sort here as a defensive guarantee (other code paths may populate
    // the field without that invariant).
    let mut sorted: Vec<(&str, f64)> = state
        .feature_importances
        .iter()
        .map(|(name, val)| (name.as_str(), *val))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Drop entries below 5% of max — noise floor for SGBT split-gain on
    // irrelevant features. Floor min at one entry so an empty chart never shows.
    let max_v = sorted.first().map(|(_, v)| *v).unwrap_or(0.0);
    let cutoff = (max_v * 0.05).max(1e-9);
    sorted.retain(|(_, v)| *v >= cutoff);
    if sorted.is_empty() {
        // Safety: keep at least one row so the chart isn't blank.
        let mut all: Vec<(&str, f64)> = state
            .feature_importances
            .iter()
            .map(|(name, val)| (name.as_str(), *val))
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted = all.into_iter().take(1).collect();
    }
    sorted.truncate(10);

    // Distinct-hue palette, NOT a gradient. Each rank gets its own categorical
    // color so the eye reads features as different things, not a smooth scale.
    // 10 entries cover the top-N truncation cap; later ranks reuse via modulo
    // wrap if a future model exposes >10 features.
    let rank_colors = [
        Color::Rgb(255, 70, 70),   // rank 0: red       (most important)
        Color::Rgb(255, 150, 30),  // rank 1: orange
        Color::Rgb(57, 255, 20),   // rank 2: neon green
        Color::Rgb(70, 130, 255),  // rank 3: electric blue
        Color::Rgb(190, 100, 255), // rank 4: neon purple
        Color::Rgb(255, 220, 50),  // rank 5: yellow
        Color::Rgb(0, 230, 220),   // rank 6: cyan
        Color::Rgb(255, 80, 200),  // rank 7: hot pink
        Color::Rgb(140, 255, 110), // rank 8: lime
        Color::Rgb(180, 180, 220), // rank 9: steel
    ];

    // Importances are sum-normalized (Σ = 1.0). The top feature's share might
    // be 0.27, not 1.0. We scale bar widths so the top bar is full-width
    // (visual comparison stays legible) but the LABEL shows the true share —
    // a sum-to-1 probability that a reader can compare across runs.
    let top_val = sorted.iter().map(|(_, v)| *v).fold(0.0_f64, f64::max);
    let bar_max_scaled: u64 = ((top_val.max(1e-9)) * 1000.0).ceil() as u64;

    let bars: Vec<Bar> = sorted
        .iter()
        .enumerate()
        .map(|(i, (name, val))| {
            let color = rank_colors[i.min(rank_colors.len() - 1)];
            let bar_val = (val * 1000.0) as u64;
            Bar::default()
                .label(Line::from(Span::styled(
                    *name,
                    Style::default().fg(theme::SUBTEXT0),
                )))
                .value(bar_val)
                .style(Style::default().fg(color))
                .value_style(
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                )
                .text_value(format!("{:.3}", val))
        })
        .collect();

    let bar_group = BarGroup::default().bars(&bars);

    let chart = BarChart::default()
        .data(bar_group)
        .block(block)
        .bar_width(1)
        .bar_gap(0)
        .max(bar_max_scaled.max(1))
        .label_style(Style::default().fg(theme::SUBTEXT0))
        .direction(Direction::Horizontal);

    frame.render_widget(chart, area);
}

/// Diagnostics tab: comprehensive ensemble and drift diagnostics from `diagnostic_rows`.
///
/// Rows come from `state.diagnostic_rows` — each is `(label, value, color_class)`.
/// Section-header rows have an empty value string and render as a single bold MAUVE span.
/// Regular rows render label in SUBTEXT0 (left col, 24 chars) + value styled by color_class:
///   "good" → GREEN, "warn" → YELLOW, "error" → RED, "neutral" → TEXT.
#[cfg(feature = "tui")]
fn render_diagnostics(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            " Diagnostics ",
            Style::default()
                .fg(theme::MAUVE)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::MAUVE))
        .style(Style::default().bg(theme::BASE));

    if state.diagnostic_rows.is_empty() {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // Map color_class string to a theme color.
    let value_color = |class: &str| -> Color {
        match class {
            "good" => theme::GREEN,
            "warn" => theme::YELLOW,
            "error" => theme::RED,
            _ => theme::TEXT,
        }
    };

    // Label column width (chars). Wide enough to gap "drift rate (events/sample)"
    // (28 chars) from its value column without bleeding into it.
    const LABEL_W: usize = 32;

    let lines: Vec<Line> = state
        .diagnostic_rows
        .iter()
        .map(|(label, value, color_class)| {
            if value.is_empty() {
                // Section header: full-width MAUVE bold span.
                Line::from(Span::styled(
                    format!(" {}", label),
                    Style::default()
                        .fg(theme::MAUVE)
                        .add_modifier(Modifier::BOLD),
                ))
            } else {
                // Data row: padded label + value.
                let label_padded = format!("  {:<width$}", label, width = LABEL_W);
                Line::from(vec![
                    Span::styled(label_padded, Style::default().fg(theme::SUBTEXT0)),
                    Span::styled(value.clone(), Style::default().fg(value_color(color_class))),
                ])
            }
        })
        .collect();

    let para = Paragraph::new(lines)
        .block(block)
        .style(Style::default().bg(theme::BASE));

    frame.render_widget(para, area);
}

/// Footer: keybinding hints + status.
#[cfg(feature = "tui")]
fn render_footer(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE1))
        .style(Style::default().bg(theme::BASE));

    let status = if state.is_done {
        Span::styled(
            " complete ",
            Style::default()
                .fg(theme::GREEN)
                .add_modifier(Modifier::BOLD),
        )
    } else if state.is_paused {
        Span::styled(
            " paused ",
            Style::default()
                .fg(theme::YELLOW)
                .add_modifier(Modifier::BOLD),
        )
    } else if state.is_training {
        Span::styled(" training... ", Style::default().fg(theme::PEACH))
    } else {
        Span::styled(
            state.status_message.clone(),
            Style::default().fg(theme::TEXT),
        )
    };

    let key = |k: &str| {
        Span::styled(
            k.to_string(),
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        )
    };
    let lbl = |t: &str| Span::styled(t.to_string(), Style::default().fg(theme::SUBTEXT0));
    let sep = || Span::styled(" · ", Style::default().fg(theme::SURFACE1));

    let line = Line::from(vec![
        Span::styled(" ", Style::default()),
        key("?"),
        lbl(" help"),
        sep(),
        key("space"),
        lbl(" pause"),
        sep(),
        key("l"),
        lbl(" log"),
        sep(),
        key("Tab"),
        lbl(" cycle"),
        sep(),
        key("m"),
        lbl(" metric"),
        sep(),
        key(", ."),
        lbl(" pinball q"),
        sep(),
        key("d"),
        lbl(" drift"),
        sep(),
        key("q"),
        lbl(" quit"),
        sep(),
        status,
    ]);

    let paragraph = Paragraph::new(line).block(block);
    frame.render_widget(paragraph, area);
}

/// Help overlay drawn over the rest of the UI when toggled with `?`.
#[cfg(feature = "tui")]
fn render_help_overlay(frame: &mut ratatui::Frame, area: ratatui::layout::Rect) {
    use ratatui::{prelude::*, widgets::*};

    // Center a fixed-size popup (60x18) within the area, capped to area size.
    let popup_w = area.width.min(60);
    let popup_h = area.height.min(18);
    let x = area.x + (area.width.saturating_sub(popup_w)) / 2;
    let y = area.y + (area.height.saturating_sub(popup_h)) / 2;
    let popup = Rect::new(x, y, popup_w, popup_h);

    // Clear underneath the popup so we don't render-over chart pixels.
    frame.render_widget(Clear, popup);

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            " Help ",
            Style::default()
                .fg(theme::PEACH)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::PEACH))
        .style(Style::default().bg(theme::BASE));

    let line = |key: &str, desc: &str| {
        Line::from(vec![
            Span::styled(
                format!(" {:<10}", key),
                Style::default()
                    .fg(theme::BLUE)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(desc.to_string(), Style::default().fg(theme::TEXT)),
        ])
    };

    let body = vec![
        Line::from(""),
        line("?", "Toggle this help overlay"),
        line("Space / p", "Pause / resume training"),
        line("l", "Toggle log10 scale on loss chart"),
        line("d", "Toggle drift overlay on loss chart"),
        line("Tab", "Cycle to next tab"),
        line("Shift+Tab", "Cycle to previous tab"),
        line("m / M", "Cycle metric (loss/R²/acc/pinball/MAE)"),
        line(", / .", "Cycle pinball quantile"),
        line("q", "Quit"),
        Line::from(""),
        Line::from(Span::styled(
            " Family is fixed per-run; pick at launch with --family <name> ",
            Style::default().fg(theme::SUBTEXT0),
        )),
    ];
    let paragraph = Paragraph::new(body).block(block);
    frame.render_widget(paragraph, popup);
}
