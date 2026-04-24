//! TUI event types for the OxiLLaMa TUI chat interface.

/// Events that the TUI loop can receive.
///
/// Currently unused — event dispatch is handled inline in [`crate::tui::app::TuiApp::run`].
/// Reserved for a future dedicated event thread that decouples input from rendering.
#[allow(dead_code)]
pub enum TuiEvent {
    /// A keyboard event from the terminal.
    Key(crossterm::event::KeyEvent),
    /// Terminal was resized to the given (width, height).
    Resize(u16, u16),
    /// A periodic tick — used for polling live stats.
    Tick,
}
