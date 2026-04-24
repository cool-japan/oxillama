//! Full-screen TUI chat interface for OxiLLaMa — powered by ratatui + crossterm.
//!
//! Gate the entire module behind `feature = "tui"`.  The entry point is
//! [`run_tui`] which sets up the terminal, hands off to [`TuiApp`], and
//! restores the terminal on exit (even on error).

pub mod app;
pub mod events;
pub mod ui;
pub mod widgets;

pub use app::TuiApp;

/// Launch the full-screen TUI for a chat session.
///
/// `model_path` is the path to the GGUF file.
/// `model_id` is the short name shown in the stats sidebar.
///
/// The function blocks until the user quits (`Ctrl+C`, `Ctrl+Q`, or `/quit`).
/// The terminal is restored to its previous state before returning, even when
/// an error occurs.
pub fn run_tui(model_path: std::path::PathBuf, model_id: String) -> anyhow::Result<()> {
    use crossterm::{
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        ExecutableCommand,
    };
    use ratatui::prelude::CrosstermBackend;
    use std::io::stdout;

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout());
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = TuiApp::new(model_path, model_id);
    let result = app.run(&mut terminal);

    // Always restore the terminal, regardless of success or error.
    let _ = disable_raw_mode();
    let _ = stdout().execute(LeaveAlternateScreen);

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use ratatui::{backend::TestBackend, Terminal};
    use std::path::PathBuf;

    use crate::session::ChatMessage;

    use super::app::{AppState, TuiApp};

    fn make_test_terminal(width: u16, height: u16) -> Terminal<TestBackend> {
        let backend = TestBackend::new(width, height);
        Terminal::new(backend).expect("TestBackend::new is infallible")
    }

    #[test]
    fn tui_renders_empty_state() {
        let mut terminal = make_test_terminal(120, 40);
        let app = TuiApp::new(PathBuf::from("test.gguf"), "test-model".to_string());
        terminal
            .draw(|f| super::ui::draw(f, &app))
            .expect("draw should not fail");

        let buffer = terminal.backend().buffer().clone();
        let content: String = buffer
            .content
            .iter()
            .map(|c| c.symbol().to_string())
            .collect();

        assert!(
            content.contains("Conversation"),
            "should render conversation block, got: {content}"
        );
        assert!(
            content.contains("Stats"),
            "should render stats sidebar, got: {content}"
        );
        assert!(
            content.contains("Input"),
            "should render input block, got: {content}"
        );
    }

    #[test]
    fn tui_appends_user_message_on_submit() {
        let mut app = TuiApp::new(PathBuf::from("test.gguf"), "test-model".to_string());
        app.input_buffer = "Hello world".to_string();
        app.cursor_pos = 11;
        app.submit_prompt().expect("submit should not fail");

        assert_eq!(app.session.messages[0].role, "user");
        assert_eq!(app.session.messages[0].content, "Hello world");
        assert!(
            app.input_buffer.is_empty(),
            "input buffer should be cleared after submit"
        );
    }

    #[test]
    fn tui_slash_save_invokes_session_save() {
        let mut app = TuiApp::new(PathBuf::from("test.gguf"), "test-model".to_string());
        let tmp = std::env::temp_dir().join("tui_test_save.bin");
        app.handle_slash_command(&format!("/save {}", tmp.display()))
            .expect("slash save should not fail");
        assert!(tmp.exists(), "save should create the file");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn tui_slash_clear_empties_messages() {
        let mut app = TuiApp::new(PathBuf::from("test.gguf"), "test-model".to_string());
        app.session.messages.push(ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        });
        app.handle_slash_command("/clear")
            .expect("slash clear should not fail");
        assert!(
            app.session.messages.is_empty(),
            "messages should be empty after /clear"
        );
    }

    #[test]
    fn tui_ctrlc_sets_quitting_state() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let mut app = TuiApp::new(PathBuf::from("test.gguf"), "test-model".to_string());
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .expect("handle_key should not fail");
        assert!(
            matches!(app.state, AppState::Quitting),
            "state should be Quitting after Ctrl+C"
        );
    }

    #[test]
    fn tui_slash_load_fails_gracefully_on_missing_file() {
        let mut app = TuiApp::new(PathBuf::from("test.gguf"), "test-model".to_string());
        // Loading a non-existent file should set a status message, not panic.
        app.handle_slash_command("/load /tmp/nonexistent_tui_test_abc123.bin")
            .expect("handle_slash_command should not propagate IO errors as anyhow::Error");
        assert!(
            app.status_msg.is_some(),
            "status_msg should be set on load error"
        );
        let msg = app.status_msg.as_deref().unwrap_or("");
        assert!(
            msg.starts_with("Load error:"),
            "status should report load error, got: {msg}"
        );
    }
}
