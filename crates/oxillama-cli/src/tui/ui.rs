//! TUI rendering — converts [`TuiApp`] state into a ratatui frame.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::app::TuiApp;

/// Draw the complete TUI frame.
///
/// Layout (vertical):
/// ```text
/// ┌────────────────────────────────────────┐
/// │  Conversation  (fills available height) │
/// ├──────────────────────┬─────────────────┤
/// │                      │  Stats sidebar  │
/// ├──────────────────────┴─────────────────┤
/// │  Status bar (1 line)                   │
/// ├────────────────────────────────────────┤
/// │  Input (≥4 lines)                      │
/// └────────────────────────────────────────┘
/// ```
pub fn draw(frame: &mut Frame, app: &TuiApp) {
    let area = frame.area();

    // Main vertical split
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Fill(1),   // conversation + stats
            Constraint::Length(1), // status bar
            Constraint::Min(4),    // input area
        ])
        .split(area);

    // Horizontal split for the conversation row
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Fill(1), Constraint::Length(28)])
        .split(vertical[0]);

    draw_conversation(frame, app, horizontal[0]);
    draw_stats_sidebar(frame, app, horizontal[1]);
    draw_status_bar(frame, app, vertical[1]);
    draw_input(frame, app, vertical[2]);
}

/// Render the conversation history pane.
fn draw_conversation(frame: &mut Frame, app: &TuiApp, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.session.messages {
        let (label, style) = if msg.role == "user" {
            (
                "You",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            (
                "Assistant",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        };

        lines.push(Line::from(Span::styled(format!("{label}:"), style)));
        for content_line in msg.content.lines() {
            lines.push(Line::from(format!("  {content_line}")));
        }
        // Blank separator between turns
        lines.push(Line::from(""));
    }

    let para = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title("Conversation"))
        .scroll((app.scroll_offset, 0));

    frame.render_widget(para, area);
}

/// Render the stats sidebar.
fn draw_stats_sidebar(frame: &mut Frame, app: &TuiApp, area: Rect) {
    // Truncate model id so it fits in the sidebar width.
    let model_short: String = app.model_id.chars().take(20).collect();

    let content = format!(
        "Model:\n  {model_short}\n\nTokens/s:\n  {:.1}\n\nTokens:\n  {}\n\nKV usage:\n  {:.0}%\n\nHelp: /help",
        app.tokens_per_sec, app.token_count, app.kv_usage_pct
    );

    let para = Paragraph::new(content)
        .block(Block::default().borders(Borders::ALL).title("Stats"))
        .style(Style::default().fg(Color::Yellow));

    frame.render_widget(para, area);
}

/// Render the one-line status bar.
fn draw_status_bar(frame: &mut Frame, app: &TuiApp, area: Rect) {
    let msg = app
        .status_msg
        .as_deref()
        .unwrap_or("Ready · Commands: /save /load /clear /quit");

    let para = Paragraph::new(msg).style(Style::default().fg(Color::DarkGray));

    frame.render_widget(para, area);
}

/// Render the user input box and position the cursor.
fn draw_input(frame: &mut Frame, app: &TuiApp, area: Rect) {
    let display = if app.input_buffer.is_empty() {
        "[Type your message, Enter to send, Shift+Enter for newline]".to_string()
    } else {
        app.input_buffer.clone()
    };

    let para = Paragraph::new(display)
        .block(Block::default().borders(Borders::ALL).title("Input"))
        .wrap(ratatui::widgets::Wrap { trim: false });

    frame.render_widget(para, area);

    // Only position cursor when the user has typed something.
    if !app.input_buffer.is_empty() {
        // area.x + 1 skips the left border; clamp to stay inside the box.
        let cursor_x =
            (area.x + 1 + app.cursor_pos as u16).min(area.x + area.width.saturating_sub(2));
        let cursor_y = area.y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}
