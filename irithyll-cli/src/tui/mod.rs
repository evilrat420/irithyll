//! TUI dashboard for live training/evaluation monitoring.
//!
//! Gated behind the `tui` feature. Provides a ratatui-based terminal UI
//! with a Catppuccin Mocha theme showing training progress, metrics,
//! and loss curves in real time.

#[cfg(feature = "tui")]
mod app;
#[cfg(feature = "tui")]
mod theme;

#[cfg(feature = "tui")]
pub use app::{AppState, SharedState};

/// Run the TUI event loop, rendering state until the user presses 'q'.
///
/// The caller spawns this on a tokio task alongside the training loop.
/// Both sides share `state` through an `Arc<Mutex<AppState>>`.
#[cfg(feature = "tui")]
pub async fn run_tui(state: SharedState) -> color_eyre::Result<()> {
    use crossterm::{
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        ExecutableCommand,
    };
    use ratatui::prelude::*;
    use std::io::stdout;

    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let result = event_loop(&mut terminal, &state).await;

    // Cleanup — always restore terminal state
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    result
}

/// Inner event loop, factored out so cleanup always runs.
#[cfg(feature = "tui")]
async fn event_loop(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    state: &SharedState,
) -> color_eyre::Result<()> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind};
    use std::time::Duration;

    loop {
        terminal.draw(|frame| render(frame, state))?;

        // Poll for keyboard events with a 100ms timeout so the UI refreshes ~10 Hz.
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        // If training is done, keep rendering but only exit on 'q'.
        // (The loop naturally continues.)
    }

    Ok(())
}

/// Render the full dashboard frame.
#[cfg(feature = "tui")]
fn render(frame: &mut ratatui::Frame, state: &SharedState) {
    use ratatui::prelude::*;

    let state = state.lock().unwrap();
    let area = frame.area();

    // Fill entire background with BASE color
    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::default().bg(theme::BASE)),
        area,
    );

    // Top-level vertical layout: header (progress bar + sparkline) | main | footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Header: title + progress bar + sparkline
            Constraint::Min(10),   // Main
            Constraint::Length(3), // Footer
        ])
        .split(area);

    render_header(frame, &state, chunks[0]);
    render_main(frame, &state, chunks[1]);
    render_footer(frame, &state, chunks[2]);
}

/// Header: title + progress gauge + sparkline.
#[cfg(feature = "tui")]
fn render_header(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::{prelude::*, symbols, widgets::*};

    let block = Block::default()
        .title(
            Line::from(vec![Span::styled(
                " irithyll ",
                Style::default()
                    .fg(theme::BLUE)
                    .add_modifier(Modifier::BOLD),
            )])
            .alignment(Alignment::Center),
        )
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Split inner into: progress bar row | sparkline + info row
    let header_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Progress gauge
            Constraint::Length(1), // Sparkline + info text
        ])
        .split(inner);

    // -- Progress bar (LineGauge) --
    let ratio = state.progress_ratio();
    let gauge_color = if state.is_done {
        theme::GREEN
    } else if ratio > 0.75 {
        theme::BLUE
    } else if ratio > 0.4 {
        theme::TEAL
    } else {
        theme::MAUVE
    };

    let eta_str = state.eta_display();
    let gauge_label = format!(
        " {}/{} ({:.1}%) | {:.0} samp/s | {:.1}s | ETA: {}",
        state.n_samples,
        state.n_total,
        state.progress_pct(),
        state.throughput,
        state.elapsed_secs,
        eta_str,
    );

    let gauge = LineGauge::default()
        .ratio(ratio)
        .filled_style(Style::default().fg(gauge_color))
        .unfilled_style(Style::default().fg(theme::SURFACE0))
        .label(Span::styled(gauge_label, Style::default().fg(theme::TEXT)))
        .line_set(symbols::line::THICK);

    frame.render_widget(gauge, header_chunks[0]);

    // -- Sparkline (last 50 loss values) --
    let sparkline_data = state.sparkline_data(50);
    if sparkline_data.is_empty() {
        let waiting = Paragraph::new(Span::styled(
            "  Loss trend: waiting for data...",
            Style::default().fg(theme::SUBTEXT0),
        ));
        frame.render_widget(waiting, header_chunks[1]);
    } else {
        // Split row: label | sparkline
        let spark_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(14), // "  Loss trend: " label
                Constraint::Min(10),    // Sparkline fill
            ])
            .split(header_chunks[1]);

        let label = Paragraph::new(Span::styled(
            "  Loss trend: ",
            Style::default().fg(theme::SUBTEXT0),
        ));
        frame.render_widget(label, spark_chunks[0]);

        let spark_max = sparkline_data.iter().copied().max().unwrap_or(1).max(1);
        let sparkline = Sparkline::default()
            .data(&sparkline_data)
            .bar_set(symbols::bar::NINE_LEVELS)
            .style(Style::default().fg(theme::PEACH).bg(theme::BASE))
            .max(spark_max);

        frame.render_widget(sparkline, spark_chunks[1]);
    }
}

/// Main area: metrics + importances (left) + loss/accuracy charts (right).
#[cfg(feature = "tui")]
fn render_main(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::prelude::*;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Left: metrics + importances
            Constraint::Percentage(60), // Right: charts
        ])
        .split(area);

    // Left panel: metrics table, optionally split with feature importances.
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if state.feature_importances.is_empty() {
            vec![Constraint::Percentage(100)]
        } else {
            vec![Constraint::Percentage(55), Constraint::Percentage(45)]
        })
        .split(chunks[0]);

    render_metrics_table(frame, state, left_chunks[0]);
    if !state.feature_importances.is_empty() {
        render_importances(frame, state, left_chunks[1]);
    }

    // Right panel: loss chart + optional accuracy chart + diagnostics panel.
    let has_accuracy = !state.accuracy_history.is_empty();
    let has_diagnostics =
        state.diagnostics_array.iter().any(|v| v.abs() > 1e-15) || state.total_replacements > 0;

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(match (has_accuracy, has_diagnostics) {
            (true, true) => vec![
                Constraint::Percentage(35),
                Constraint::Percentage(35),
                Constraint::Percentage(30),
            ],
            (true, false) => vec![Constraint::Percentage(50), Constraint::Percentage(50)],
            (false, true) => vec![Constraint::Percentage(60), Constraint::Percentage(40)],
            (false, false) => vec![Constraint::Percentage(100)],
        })
        .split(chunks[1]);

    render_loss_chart(frame, state, right_chunks[0]);

    let mut next_idx = 1;
    if has_accuracy && right_chunks.len() > next_idx {
        render_accuracy_chart(frame, state, right_chunks[next_idx]);
        next_idx += 1;
    }
    if has_diagnostics && right_chunks.len() > next_idx {
        render_diagnostics(frame, state, right_chunks[next_idx]);
    }
}

/// Left panel: metrics table with color-coded values.
#[cfg(feature = "tui")]
fn render_metrics_table(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            " Metrics ",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    let header = Row::new(vec!["Metric", "Value"]).style(
        Style::default()
            .fg(theme::BLUE)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = state
        .metrics
        .iter()
        .map(|(name, value)| {
            let value_color = color_for_metric(name, *value);
            Row::new(vec![
                Cell::from(Span::styled(
                    name.as_str(),
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                )),
                Cell::from(Span::styled(
                    format!("{:.6}", value),
                    Style::default().fg(value_color),
                )),
            ])
        })
        .collect();

    // Add ETA row at the bottom
    let mut all_rows = rows;
    all_rows.push(Row::new(vec![
        Cell::from(Span::styled(
            "ETA",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )),
        Cell::from(Span::styled(
            state.eta_display(),
            Style::default().fg(theme::LAVENDER),
        )),
    ]));
    all_rows.push(Row::new(vec![
        Cell::from(Span::styled(
            "Throughput",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )),
        Cell::from(Span::styled(
            format!("{:.0} samp/s", state.throughput),
            Style::default().fg(theme::TEAL),
        )),
    ]));

    let table = Table::new(
        all_rows,
        [Constraint::Percentage(50), Constraint::Percentage(50)],
    )
    .block(block)
    .header(header);

    frame.render_widget(table, area);
}

/// Color-code metric values based on their name and magnitude.
#[cfg(feature = "tui")]
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

/// Min-max envelope downsampling: preserves peaks and valleys better than averaging.
/// For each bucket, emits (bucket_start_x, min) and (bucket_start_x, max) as two points.
#[cfg(feature = "tui")]
fn downsample_minmax(history: &[f64], max_points: usize) -> Vec<(f64, f64)> {
    if history.len() <= max_points {
        return history
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64, *v))
            .collect();
    }

    let bucket_size = history.len() / max_points;
    let mut data = Vec::with_capacity(max_points * 2);

    for (bi, chunk) in history.chunks(bucket_size).enumerate() {
        let x = bi as f64;
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for &v in chunk {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        // Emit min first, then max — this creates an envelope
        data.push((x, lo));
        if (hi - lo).abs() > f64::EPSILON {
            data.push((x, hi));
        }
    }

    data
}

/// Generate Y-axis labels with intermediate ticks.
#[cfg(feature = "tui")]
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
#[cfg(feature = "tui")]
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

/// Right panel: loss curve chart with min-max envelope and improved axes.
#[cfg(feature = "tui")]
fn render_loss_chart(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, symbols::Marker, widgets::*};

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            " Loss Curve ",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::SURFACE0));

    if state.loss_history.is_empty() {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // Min-max envelope downsampling for dense rendering
    let max_points = 200;
    let data = downsample_minmax(&state.loss_history, max_points);

    let x_max = data
        .iter()
        .map(|(x, _)| *x)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let y_max = state.loss_history.iter().cloned().fold(0.0_f64, f64::max);
    let y_min = state.loss_history.iter().cloned().fold(f64::MAX, f64::min);

    // Pad Y-axis so dots near min/max are never clipped
    let range = (y_max - y_min).max(0.001);
    let padding = range * 0.15;
    let (y_lo, y_hi) = (y_min - padding, y_max + padding);

    let dataset = Dataset::default()
        .marker(Marker::Dot)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(theme::PEACH))
        .data(&data);

    let x_labels = x_axis_labels(x_max, state.n_samples);
    let y_labels = y_axis_labels(y_lo, y_hi);

    let chart = Chart::new(vec![dataset])
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

/// Left sub-panel: horizontal bar chart of feature importances with cycling colors.
#[cfg(feature = "tui")]
fn render_importances(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            " Feature Importances ",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    // Sort descending by importance, take top 10.
    let mut sorted: Vec<(&str, f64)> = state
        .feature_importances
        .iter()
        .map(|(name, val)| (name.as_str(), *val))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(10);

    // Build individual Bar items with cycling colors and value labels
    let colors = [theme::MAUVE, theme::BLUE, theme::TEAL, theme::LAVENDER];

    let bars: Vec<Bar> = sorted
        .iter()
        .enumerate()
        .map(|(i, (name, val))| {
            let color = colors[i % colors.len()];
            Bar::default()
                .label(Line::from(Span::styled(
                    *name,
                    Style::default().fg(theme::SUBTEXT0),
                )))
                .value((val * 1000.0) as u64)
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
        .label_style(Style::default().fg(theme::SUBTEXT0))
        .direction(Direction::Horizontal);

    frame.render_widget(chart, area);
}

/// Right sub-panel: accuracy curve chart with improved rendering.
#[cfg(feature = "tui")]
fn render_accuracy_chart(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, symbols::Marker, widgets::*};

    let block = Block::bordered()
        .title(Line::from(vec![Span::styled(
            " Accuracy Curve ",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::SURFACE0));

    if state.accuracy_history.is_empty() {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // Min-max envelope downsampling
    let max_points = 200;
    let data = downsample_minmax(&state.accuracy_history, max_points);

    let x_max = data
        .iter()
        .map(|(x, _)| *x)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let y_max = state
        .accuracy_history
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    let y_min = state
        .accuracy_history
        .iter()
        .cloned()
        .fold(f64::MAX, f64::min);

    let range = (y_max - y_min).max(0.001);
    let padding = range * 0.15;
    let (y_lo, y_hi) = (y_min - padding, y_max + padding);

    let dataset = Dataset::default()
        .marker(Marker::Dot)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(theme::GREEN))
        .data(&data);

    let x_labels = x_axis_labels(x_max, state.n_samples);
    let y_labels = y_axis_labels(y_lo, y_hi);

    let chart = Chart::new(vec![dataset])
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

/// Right sub-panel: model diagnostics (replacements, diagnostic signals, honest_sigma).
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
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )]))
        .border_style(Style::default().fg(theme::MAUVE))
        .style(Style::default().bg(theme::BASE));

    let d = state.diagnostics_array;
    let labels = [
        ("Residual Align", d[0]),
        ("Reg Sensitivity", d[1]),
        ("Depth Sufficiency", d[2]),
        ("Effective DOF", d[3]),
        ("Uncertainty", d[4]),
    ];

    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from(Span::styled(
            "Replacements",
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )),
        Cell::from(Span::styled(
            format!("{}", state.total_replacements),
            Style::default().fg(theme::PEACH),
        )),
    ])];

    for (name, val) in &labels {
        let color = if val.abs() < 1e-15 {
            theme::SUBTEXT0
        } else if *val > 0.5 {
            theme::GREEN
        } else {
            theme::TEAL
        };
        rows.push(Row::new(vec![
            Cell::from(Span::styled(
                *name,
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                format!("{:.6}", val),
                Style::default().fg(color),
            )),
        ]));
    }

    if state.model_type == "distributional" && state.honest_sigma > 0.0 {
        rows.push(Row::new(vec![
            Cell::from(Span::styled(
                "Honest Sigma",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                format!("{:.6}", state.honest_sigma),
                Style::default().fg(theme::LAVENDER),
            )),
        ]));
    }

    let table = Table::new(
        rows,
        [Constraint::Percentage(50), Constraint::Percentage(50)],
    )
    .block(block);

    frame.render_widget(table, area);
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
    } else if state.is_training {
        Span::styled(" training... ", Style::default().fg(theme::PEACH))
    } else {
        Span::styled(&state.status_message, Style::default().fg(theme::TEXT))
    };

    let line = Line::from(vec![
        Span::styled(
            " q",
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(": quit", Style::default().fg(theme::SUBTEXT0)),
        Span::styled(" | ", Style::default().fg(theme::SURFACE1)),
        status,
    ]);

    let paragraph = Paragraph::new(line).block(block);

    frame.render_widget(paragraph, area);
}
