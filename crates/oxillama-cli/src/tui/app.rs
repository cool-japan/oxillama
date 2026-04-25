//! TUI application state machine for OxiLLaMa chat.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::events::TuiEvent;
use crate::session::{ChatMessage, SessionSnapshot};

/// The lifecycle state of the TUI application.
pub enum AppState {
    /// Waiting for user input.
    Idle,
    /// Model is currently generating a response.
    Generating,
    /// The user has requested to quit.
    Quitting,
}

/// All mutable state for the full-screen TUI chat interface.
pub struct TuiApp {
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

    /// Channel to send prompt strings to the inference worker.
    request_tx: std::sync::mpsc::SyncSender<String>,
    /// Channel to receive token/done/error events from the inference worker.
    event_rx: std::sync::mpsc::Receiver<TuiEvent>,

    /// Accumulating streamed response (None when not generating).
    partial_assistant: Option<String>,
    /// Wall-clock start of the current generation (for tokens/sec).
    gen_start: Option<Instant>,
    /// Token count at the start of the current generation (unused directly
    /// but kept for delta-counting symmetry with the server worker).
    last_token_count: u64,
}

impl TuiApp {
    /// Create a new [`TuiApp`] with a real inference engine.
    ///
    /// Spawns a background blocking worker on the current Tokio runtime that
    /// owns the engine and processes one prompt at a time.
    pub fn new(
        _model_path: PathBuf,
        model_id: String,
        engine: Arc<Mutex<oxillama_runtime::InferenceEngine>>,
        sampler: oxillama_runtime::SamplerConfig,
        max_tokens: usize,
    ) -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<String>(1);
        let (event_tx, event_rx) = std::sync::mpsc::channel::<TuiEvent>();

        let engine_clone = engine.clone();
        let event_tx_clone = event_tx.clone();

        // Spawn the inference worker on a dedicated Tokio blocking thread so
        // that the heavy generate() call never stalls the draw loop.
        tokio::task::spawn_blocking(move || {
            while let Ok(prompt) = request_rx.recv() {
                let result = {
                    let mut eng = engine_clone.lock().unwrap_or_else(|e| e.into_inner());
                    // Reset KV cache between turns so the full conversation
                    // is prefilled fresh each time (avoids double-prefill).
                    eng.reset();
                    eng.generate_with_config(&prompt, max_tokens, sampler.clone(), |tok| {
                        // Ignore send errors (TUI might have quit).
                        let _ = event_tx_clone.send(TuiEvent::Token(tok.to_string()));
                    })
                };

                match result {
                    Ok(_) => {
                        let _ = event_tx_clone.send(TuiEvent::GenerationDone);
                    }
                    Err(e) => {
                        let _ = event_tx_clone.send(TuiEvent::GenerationError(e.to_string()));
                    }
                }
            }
        });

        Self {
            session: SessionSnapshot::new(model_id.as_str()),
            model_id,
            input_buffer: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            state: AppState::Idle,
            status_msg: None,
            tokens_per_sec: 0.0,
            token_count: 0,
            kv_usage_pct: 0.0,
            request_tx,
            event_rx,
            partial_assistant: None,
            gen_start: None,
            last_token_count: 0,
        }
    }

    /// Create a [`TuiApp`] for unit tests using caller-supplied channels.
    ///
    /// No real engine is loaded; the caller drives the `event_tx` side directly.
    #[cfg(test)]
    pub fn new_for_testing(
        event_rx: std::sync::mpsc::Receiver<TuiEvent>,
        request_tx: std::sync::mpsc::SyncSender<String>,
    ) -> Self {
        Self {
            session: SessionSnapshot::new("test-model"),
            model_id: "test-model".to_string(),
            input_buffer: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            state: AppState::Idle,
            status_msg: None,
            tokens_per_sec: 0.0,
            token_count: 0,
            kv_usage_pct: 0.0,
            request_tx,
            event_rx,
            partial_assistant: None,
            gen_start: None,
            last_token_count: 0,
        }
    }

    /// Create a [`TuiApp`] for UI-rendering tests (no engine, no channels wired).
    ///
    /// The idle channels are never written to; the app stays in `Idle` state.
    #[cfg(test)]
    pub fn new_ui_test(_model_path: PathBuf, model_id: String) -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<String>(1);
        let (_event_tx, event_rx) = std::sync::mpsc::channel::<TuiEvent>();

        // Keep the request receiver alive on a background thread so that
        // submit_prompt's try_send does not fail with Disconnected in UI tests.
        // The thread exits cleanly when request_tx is dropped (i.e., when the
        // TuiApp is dropped).
        std::thread::spawn(move || while request_rx.recv().is_ok() {});

        Self {
            session: SessionSnapshot::new(model_id.as_str()),
            model_id,
            input_buffer: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            state: AppState::Idle,
            status_msg: None,
            tokens_per_sec: 0.0,
            token_count: 0,
            kv_usage_pct: 0.0,
            request_tx,
            event_rx,
            partial_assistant: None,
            gen_start: None,
            last_token_count: 0,
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

            // Drain all pending token / done / error events from the worker.
            self.drain_worker_events();
        }
        Ok(())
    }

    /// Drain all pending events from the inference worker channel without blocking.
    fn drain_worker_events(&mut self) {
        loop {
            match self.event_rx.try_recv() {
                Ok(TuiEvent::Token(tok)) => {
                    if let Some(partial) = &mut self.partial_assistant {
                        partial.push_str(&tok);
                    } else {
                        self.partial_assistant = Some(tok.clone());
                    }
                    self.token_count += 1;

                    // Update live tokens/sec.
                    if let Some(start) = self.gen_start {
                        let elapsed = start.elapsed().as_secs_f64();
                        if elapsed > 0.0 {
                            let delta = self.token_count - self.last_token_count;
                            self.tokens_per_sec = delta as f64 / elapsed;
                        }
                    }

                    // Patch the last (placeholder) assistant message in-place.
                    if let Some(last) = self.session.messages.last_mut() {
                        if last.role == "assistant" {
                            last.content = self.partial_assistant.clone().unwrap_or_default();
                        }
                    }
                }

                Ok(TuiEvent::GenerationDone) => {
                    self.state = AppState::Idle;
                    self.partial_assistant = None;
                    self.gen_start = None;
                    self.status_msg = None;
                }

                Ok(TuiEvent::GenerationError(e)) => {
                    self.state = AppState::Idle;
                    if let Some(last) = self.session.messages.last_mut() {
                        if last.role == "assistant" {
                            last.content = format!("[Error: {e}]");
                        }
                    }
                    self.partial_assistant = None;
                    self.gen_start = None;
                    self.status_msg = Some(format!("Error: {e}"));
                }

                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.state = AppState::Idle;
                    break;
                }
            }
        }
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

        // Block new submissions while the worker is busy.
        if matches!(self.state, AppState::Generating) {
            self.status_msg =
                Some("Already generating. Wait for response to complete.".to_string());
            return Ok(());
        }

        // Push user message, then an empty placeholder for the assistant reply.
        self.session.messages.push(ChatMessage {
            role: "user".to_string(),
            content: input.clone(),
        });
        self.session.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
        });
        self.input_buffer.clear();
        self.cursor_pos = 0;

        // Transition to Generating state.
        self.state = AppState::Generating;
        self.partial_assistant = Some(String::new());
        self.gen_start = Some(Instant::now());
        self.last_token_count = self.token_count;

        // Build full prompt from session history (excluding the empty placeholder).
        let prompt = build_prompt_from_session(&self.session);

        // Send the prompt to the worker.  On failure, roll back state.
        if let Err(e) = self.request_tx.try_send(prompt) {
            self.state = AppState::Idle;
            // Remove the placeholder assistant message and the user message.
            self.session.messages.pop();
            self.session.messages.pop();
            self.partial_assistant = None;
            self.gen_start = None;
            self.status_msg = Some(format!("Failed to start generation: {e}"));
        }

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

/// Build a single prompt string from the full session history.
///
/// The empty assistant placeholder pushed at the tail of `submit_prompt` is
/// intentionally excluded (it has an empty `content` string and is the target
/// for streaming, not an input).  All prior complete turns are included.
fn build_prompt_from_session(session: &SessionSnapshot) -> String {
    let mut prompt = String::new();
    for msg in &session.messages {
        // Skip empty assistant placeholders (the streaming target).
        if msg.role == "assistant" && msg.content.is_empty() {
            continue;
        }
        match msg.role.as_str() {
            "user" => {
                prompt.push_str("User: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            "assistant" => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            "system" => {
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            _ => {}
        }
    }
    // Elicit the assistant response.
    prompt.push_str("Assistant:");
    prompt
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::events::TuiEvent;

    /// Helper: create a TuiApp with owned channels so tests can drive events.
    fn make_test_app() -> (
        TuiApp,
        std::sync::mpsc::Sender<TuiEvent>,
        std::sync::mpsc::Receiver<String>,
    ) {
        let (event_tx, event_rx) = std::sync::mpsc::channel::<TuiEvent>();
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<String>(4);
        let app = TuiApp::new_for_testing(event_rx, request_tx);
        (app, event_tx, request_rx)
    }

    #[test]
    fn tui_token_event_appends_to_partial() {
        let (mut app, event_tx, _req_rx) = make_test_app();

        // Put app in Generating state with a placeholder assistant message.
        app.state = AppState::Generating;
        app.partial_assistant = Some(String::new());
        app.gen_start = Some(Instant::now());
        app.session.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
        });

        event_tx.send(TuiEvent::Token("Hello".to_string())).unwrap();
        event_tx
            .send(TuiEvent::Token(" world".to_string()))
            .unwrap();

        app.drain_worker_events();

        assert_eq!(
            app.partial_assistant.as_deref(),
            Some("Hello world"),
            "partial should accumulate tokens"
        );
        assert_eq!(app.token_count, 2, "token_count should increase");
        assert_eq!(
            app.session.messages.last().map(|m| m.content.as_str()),
            Some("Hello world"),
            "last assistant message content should be updated"
        );
    }

    #[test]
    fn tui_generation_done_resets_to_idle() {
        let (mut app, event_tx, _req_rx) = make_test_app();

        app.state = AppState::Generating;
        app.partial_assistant = Some("half".to_string());
        app.gen_start = Some(Instant::now());

        event_tx.send(TuiEvent::GenerationDone).unwrap();
        app.drain_worker_events();

        assert!(
            matches!(app.state, AppState::Idle),
            "state should be Idle after GenerationDone"
        );
        assert!(
            app.partial_assistant.is_none(),
            "partial_assistant should be cleared"
        );
        assert!(app.gen_start.is_none(), "gen_start should be cleared");
        assert!(app.status_msg.is_none(), "status_msg should be cleared");
    }

    #[test]
    fn tui_blocks_submit_during_generating() {
        let (mut app, _event_tx, _req_rx) = make_test_app();

        app.state = AppState::Generating;
        app.input_buffer = "some input".to_string();
        app.cursor_pos = 10;

        app.submit_prompt().expect("submit should not fail");

        assert!(
            matches!(app.state, AppState::Generating),
            "state should still be Generating"
        );
        assert!(
            app.status_msg.is_some(),
            "status_msg should be set when blocked"
        );
        let msg = app.status_msg.as_deref().unwrap_or("");
        assert!(
            msg.contains("Already generating"),
            "status should mention Already generating, got: {msg}"
        );
    }

    #[test]
    fn tui_generation_error_shows_status_msg() {
        let (mut app, event_tx, _req_rx) = make_test_app();

        app.state = AppState::Generating;
        app.partial_assistant = Some(String::new());
        app.gen_start = Some(Instant::now());
        app.session.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
        });

        event_tx
            .send(TuiEvent::GenerationError("oops".to_string()))
            .unwrap();
        app.drain_worker_events();

        assert!(
            matches!(app.state, AppState::Idle),
            "state should be Idle after GenerationError"
        );
        let status = app.status_msg.as_deref().unwrap_or("");
        assert!(
            status.contains("Error: oops"),
            "status_msg should contain error text, got: {status}"
        );
    }

    #[test]
    fn build_prompt_skips_empty_assistant_placeholder() {
        let mut session = SessionSnapshot::new("test");
        session.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        });
        // Empty assistant placeholder (streaming target — must be excluded).
        session.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: String::new(),
        });

        let prompt = build_prompt_from_session(&session);
        assert!(
            prompt.contains("User: Hi"),
            "prompt should include user turn"
        );
        assert!(
            !prompt.contains("Assistant: \n"),
            "empty assistant placeholder should not appear in prompt"
        );
        assert!(
            prompt.ends_with("Assistant:"),
            "prompt should end with 'Assistant:' to elicit the response"
        );
    }

    #[test]
    fn build_prompt_includes_completed_turns() {
        let mut session = SessionSnapshot::new("test");
        session.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "Turn 1".to_string(),
        });
        session.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: "Reply 1".to_string(),
        });
        session.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "Turn 2".to_string(),
        });

        let prompt = build_prompt_from_session(&session);
        assert!(prompt.contains("User: Turn 1"));
        assert!(prompt.contains("Assistant: Reply 1"));
        assert!(prompt.contains("User: Turn 2"));
        assert!(prompt.ends_with("Assistant:"));
    }
}
