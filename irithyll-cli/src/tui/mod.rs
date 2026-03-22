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

/// Main area: metrics table (left) + loss chart (right).
#[cfg(feature = "tui")]
fn render_main(frame: &mut ratatui::Frame, state: &app::AppState, area: ratatui::layout::Rect) {
    use ratatui::prelude::*;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // Metrics
            Constraint::Percentage(60), // Chart
        ])
        .split(area);

    render_metrics_table(frame, state, chunks[0]);
    render_loss_chart(frame, state, chunks[1]);
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

    let data: Vec<(f64, f64)> = state
        .loss_history
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v))
        .collect();

    let x_max = (data.len() as f64).max(1.0);
    let y_max = state.loss_history.iter().cloned().fold(0.0_f64, f64::max);
    let y_min = state.loss_history.iter().cloned().fold(f64::MAX, f64::min);

    // Avoid zero-height axis when all losses are identical.
    let (y_lo, y_hi) = if (y_max - y_min).abs() < f64::EPSILON {
        (y_min - 0.1, y_max + 0.1)
    } else {
        (y_min * 0.9, y_max * 1.1)
    };

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
