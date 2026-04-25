//! TUI event types for the OxiLLaMa TUI chat interface.

/// Events that the TUI loop can receive.
///
/// Token, GenerationDone, and GenerationError are produced by the background
/// inference worker and consumed in the main draw loop via `event_rx.try_recv()`.
pub enum TuiEvent {
    /// A decoded token string emitted by the inference worker.
    Token(String),
    /// Inference completed successfully.
    GenerationDone,
    /// Inference terminated with an error.
    GenerationError(String),
}
