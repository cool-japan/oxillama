//! TUI application state machine for OxiLLaMa chat.

use std::path::PathBuf;

use crate::session::{ChatMessage, SessionSnapshot};

/// The lifecycle state of the TUI application.
pub enum AppState {
    /// Waiting for user input.
    Idle,
    /// Model is currently generating a response (reserved for future async generation).
    #[allow(dead_code)]
    Generating,
    /// The user has requested to quit.
    Quitting,
}

/// All mutable state for the full-screen TUI chat interface.
pub struct TuiApp {
    /// Path to the GGUF model file (retained for future async engine integration).
    #[allow(dead_code)]
    pub model_path: PathBuf,
    /// Short identifier for the model (derived from file stem).
    pub model_id: String,
    /// The live conversation session.
    pub session: SessionSnapshot,
    /// Current contents of the user's input box.
    pub input_buffer: String,
    /// Byte offset of the cursor within `input_buffer`.
    pub cursor_pos: usize,
    /// Vertical scroll offset for the conversation view (in lines).
    pub scroll_offset: u16,
    /// Current lifecycle state.
    pub state: AppState,
    /// Transient status message shown in the status bar.
    pub status_msg: Option<String>,
    /// Live generation speed in tokens per second.
    pub tokens_per_sec: f64,
    /// Total tokens generated in this session.
    pub token_count: u64,
    /// KV-cache utilisation as a percentage (0–100).
    pub kv_usage_pct: f64,
}

impl TuiApp {
    /// Create a new [`TuiApp`] ready to begin a fresh session.
    pub fn new(model_path: PathBuf, model_id: String) -> Self {
        Self {
            session: SessionSnapshot::new(model_id.as_str()),
            model_path,
            model_id,
            input_buffer: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            state: AppState::Idle,
            status_msg: None,
            tokens_per_sec: 0.0,
            token_count: 0,
            kv_usage_pct: 0.0,
        }
    }

    /// Run the main event loop, drawing to `terminal` on each iteration.
    pub fn run<B>(&mut self, terminal: &mut ratatui::Terminal<B>) -> anyhow::Result<()>
    where
        B: ratatui::backend::Backend,
        B::Error: Send + Sync + 'static,
    {
        use crossterm::event::{self, Event};
        use std::time::Duration;

        loop {
            terminal.draw(|frame| super::ui::draw(frame, self))?;

            if matches!(self.state, AppState::Quitting) {
                break;
            }

            if event::poll(Duration::from_millis(50))? {
                match event::read()? {
                    Event::Key(key) => self.handle_key(key)?,
                    // ratatui handles resize automatically on next draw
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Dispatch a keyboard event to the appropriate handler.
    pub fn handle_key(&mut self, key: crossterm::event::KeyEvent) -> anyhow::Result<()> {
        use crossterm::event::{KeyCode, KeyModifiers};

        match (key.modifiers, key.code) {
            // Ctrl+C or Ctrl+Q — quit
            (KeyModifiers::CONTROL, KeyCode::Char('c'))
            | (KeyModifiers::CONTROL, KeyCode::Char('q')) => {
                self.state = AppState::Quitting;
            }

            // Enter — submit prompt
            (KeyModifiers::NONE, KeyCode::Enter) => {
                self.submit_prompt()?;
            }

            // Shift+Enter — insert newline in input buffer
            (KeyModifiers::SHIFT, KeyCode::Enter) => {
                self.input_buffer.insert(self.cursor_pos, '\n');
                self.cursor_pos += 1;
            }

            // Backspace — delete character before cursor
            (KeyModifiers::NONE, KeyCode::Backspace) if self.cursor_pos > 0 => {
                self.cursor_pos -= 1;
                self.input_buffer.remove(self.cursor_pos);
            }
            (KeyModifiers::NONE, KeyCode::Backspace) => {}

            // Left/Right — move cursor
            (KeyModifiers::NONE, KeyCode::Left) => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            (KeyModifiers::NONE, KeyCode::Right) => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.input_buffer.len());
            }

            // Up/Down — scroll conversation view
            (KeyModifiers::NONE, KeyCode::Up) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            (KeyModifiers::NONE, KeyCode::Down) => {
                self.scroll_offset += 1;
            }

            // Regular character input (unmodified or shift-modified)
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => {
                self.input_buffer.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }

            _ => {}
        }
        Ok(())
    }

    /// Process the current contents of `input_buffer` as a user submission.
    pub fn submit_prompt(&mut self) -> anyhow::Result<()> {
        let input = self.input_buffer.trim().to_string();
        if input.is_empty() {
            return Ok(());
        }

        // Slash commands are handled separately
        if input.starts_with('/') {
            self.handle_slash_command(&input)?;
            self.input_buffer.clear();
            self.cursor_pos = 0;
            return Ok(());
        }

        // Record user turn
        self.session.messages.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });
        self.input_buffer.clear();
        self.cursor_pos = 0;

        // TODO: spawn async generation task against InferenceEngine.
        // Full streaming inference is wired in the `chat` REPL (non-TUI path).
        // The TUI path shows a placeholder until the engine can be handed off
        // asynchronously without blocking the draw loop.
        self.session.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: "[Generation requires engine integration — \
                use `oxillama chat` without --tui for full inference]"
                .to_string(),
        });

        Ok(())
    }

    /// Execute an in-chat slash command such as `/save`, `/load`, `/clear`.
    pub fn handle_slash_command(&mut self, cmd: &str) -> anyhow::Result<()> {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        match parts[0] {
            "/save" => {
                if let Some(path_str) = parts.get(1) {
                    let p = std::path::Path::new(path_str.trim());
                    crate::session::save(&self.session, p)?;
                    self.status_msg = Some(format!("Session saved to {}", path_str.trim()));
                } else {
                    self.status_msg = Some("Usage: /save <path>".to_string());
                }
            }
            "/load" => {
                if let Some(path_str) = parts.get(1) {
                    let p = std::path::Path::new(path_str.trim());
                    match crate::session::load_for_model(p, &self.model_id) {
                        Ok(snap) => {
                            self.session = snap;
                            self.status_msg =
                                Some(format!("Session loaded from {}", path_str.trim()));
                        }
                        Err(e) => {
                            self.status_msg = Some(format!("Load error: {e}"));
                        }
                    }
                } else {
                    self.status_msg = Some("Usage: /load <path>".to_string());
                }
            }
            "/clear" => {
                self.session.messages.clear();
                self.status_msg = Some("Conversation cleared".to_string());
            }
            "/quit" | "/q" => {
                self.state = AppState::Quitting;
            }
            "/help" => {
                self.status_msg =
                    Some("Commands: /save <path>, /load <path>, /clear, /quit".to_string());
            }
            _ => {
                self.status_msg = Some(format!("Unknown command: {}", parts[0]));
            }
        }
        Ok(())
    }
}
