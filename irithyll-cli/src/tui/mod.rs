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

    // Top-level vertical layout: header | main | footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Main
            Constraint::Length(3), // Footer
        ])
        .split(area);

    render_header(frame, &state, chunks[0]);
    render_main(frame, &state, chunks[1]);
    render_footer(frame, &state, chunks[2]);
}

/// Header: title + progress summary line.
#[cfg(feature = "tui")]
fn render_header(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::default()
        .title(" irithyll ")
        .title_alignment(Alignment::Center)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BLUE))
        .style(Style::default().bg(theme::BASE));

    let text = format!(
        "Training: {}/{} ({:.1}%) | {:.0} samples/sec | {:.1}s elapsed",
        state.n_samples,
        state.n_total,
        state.progress_pct(),
        state.throughput,
        state.elapsed_secs,
    );

    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(theme::TEXT))
        .block(block)
        .alignment(Alignment::Center);

    frame.render_widget(paragraph, area);
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

    // Right panel: loss chart, optionally split with accuracy chart.
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(if state.accuracy_history.is_empty() {
            vec![Constraint::Percentage(100)]
        } else {
            vec![Constraint::Percentage(50), Constraint::Percentage(50)]
        })
        .split(chunks[1]);

    render_loss_chart(frame, state, right_chunks[0]);
    if !state.accuracy_history.is_empty() {
        render_accuracy_chart(frame, state, right_chunks[1]);
    }
}

/// Left panel: scrollable metrics table.
#[cfg(feature = "tui")]
fn render_metrics_table(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::default()
        .title(" Metrics ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE1))
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
            Row::new(vec![
                Cell::from(name.as_str()).style(Style::default().fg(theme::SUBTEXT0)),
                Cell::from(format!("{:.6}", value)).style(Style::default().fg(theme::GREEN)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [Constraint::Percentage(50), Constraint::Percentage(50)],
    )
    .block(block)
    .header(header);

    frame.render_widget(table, area);
}

/// Right panel: Braille-style loss curve chart.
#[cfg(feature = "tui")]
fn render_loss_chart(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, symbols::Marker, widgets::*};

    let block = Block::default()
        .title(" Loss Curve ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE1))
        .style(Style::default().bg(theme::BASE));

    if state.loss_history.is_empty() {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // Downsample to at most 200 points by averaging buckets so the chart
    // stays readable even when loss_history grows very large.
    let max_points = 200;
    let data: Vec<(f64, f64)> = if state.loss_history.len() > max_points {
        let bucket_size = state.loss_history.len() / max_points;
        state
            .loss_history
            .chunks(bucket_size)
            .enumerate()
            .map(|(i, chunk)| {
                let avg = chunk.iter().sum::<f64>() / chunk.len() as f64;
                (i as f64, avg)
            })
            .collect()
    } else {
        state
            .loss_history
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64, *v))
            .collect()
    };

    let x_max = (data.len() as f64).max(1.0);
    let y_max = state.loss_history.iter().cloned().fold(0.0_f64, f64::max);
    let y_min = state.loss_history.iter().cloned().fold(f64::MAX, f64::min);

    // Pad the Y-axis symmetrically so dots near min/max are never clipped.
    let range = (y_max - y_min).max(0.001);
    let padding = range * 0.15;
    let (y_lo, y_hi) = (y_min - padding, y_max + padding);

    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(theme::PEACH))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(block)
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .style(Style::default().fg(theme::SUBTEXT0)),
        )
        .y_axis(
            Axis::default()
                .bounds([y_lo, y_hi])
                .labels(vec![
                    Line::from(format!("{:.4}", y_min)),
                    Line::from(format!("{:.4}", y_max)),
                ])
                .style(Style::default().fg(theme::SUBTEXT0)),
        );

    frame.render_widget(chart, area);
}

/// Left sub-panel: horizontal bar chart of feature importances.
#[cfg(feature = "tui")]
fn render_importances(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::default()
        .title(" Feature Importances ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE1))
        .style(Style::default().bg(theme::BASE));

    // Scale f64 importances to u64 (multiply by 1000) for BarChart.
    // Take the top entries that fit; sort descending by importance.
    let mut sorted: Vec<(&str, f64)> = state
        .feature_importances
        .iter()
        .map(|(name, val)| (name.as_str(), *val))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Limit to at most 10 features so it fits the panel.
    sorted.truncate(10);

    let bar_data: Vec<(&str, u64)> = sorted
        .iter()
        .map(|(name, val)| (*name, (val * 1000.0) as u64))
        .collect();

    let chart = BarChart::default()
        .data(&bar_data)
        .block(block)
        .bar_width(1)
        .bar_gap(0)
        .bar_style(Style::default().fg(theme::MAUVE))
        .value_style(
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        )
        .label_style(Style::default().fg(theme::SUBTEXT0))
        .direction(Direction::Horizontal);

    frame.render_widget(chart, area);
}

/// Right sub-panel: Braille-style accuracy curve chart.
#[cfg(feature = "tui")]
fn render_accuracy_chart(
    frame: &mut ratatui::Frame,
    state: &app::AppState,
    area: ratatui::layout::Rect,
) {
    use ratatui::{prelude::*, symbols::Marker, widgets::*};

    let block = Block::default()
        .title(" Accuracy Curve ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE1))
        .style(Style::default().bg(theme::BASE));

    if state.accuracy_history.is_empty() {
        let empty = Paragraph::new("Waiting for data...")
            .style(Style::default().fg(theme::SUBTEXT0))
            .block(block)
            .alignment(Alignment::Center);
        frame.render_widget(empty, area);
        return;
    }

    // Downsample accuracy history the same way as loss.
    let max_points = 200;
    let data: Vec<(f64, f64)> = if state.accuracy_history.len() > max_points {
        let bucket_size = state.accuracy_history.len() / max_points;
        state
            .accuracy_history
            .chunks(bucket_size)
            .enumerate()
            .map(|(i, chunk)| {
                let avg = chunk.iter().sum::<f64>() / chunk.len() as f64;
                (i as f64, avg)
            })
            .collect()
    } else {
        state
            .accuracy_history
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f64, *v))
            .collect()
    };

    let x_max = (data.len() as f64).max(1.0);
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
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(theme::GREEN))
        .data(&data);

    let chart = Chart::new(vec![dataset])
        .block(block)
        .x_axis(
            Axis::default()
                .bounds([0.0, x_max])
                .style(Style::default().fg(theme::SUBTEXT0)),
        )
        .y_axis(
            Axis::default()
                .bounds([y_lo, y_hi])
                .labels(vec![
                    Line::from(format!("{:.4}", y_min)),
                    Line::from(format!("{:.4}", y_max)),
                ])
                .style(Style::default().fg(theme::SUBTEXT0)),
        );

    frame.render_widget(chart, area);
}

/// Footer: status message + keybinding hints.
#[cfg(feature = "tui")]
fn render_footer(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::{prelude::*, widgets::*};

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE1))
        .style(Style::default().bg(theme::BASE));

    let status = if state.is_done {
        "Training complete. Press 'q' to exit."
    } else {
        &state.status_message
    };

    let paragraph = Paragraph::new(format!(" {} | q: quit", status))
        .style(Style::default().fg(theme::SUBTEXT0))
        .block(block);

    frame.render_widget(paragraph, area);
}
