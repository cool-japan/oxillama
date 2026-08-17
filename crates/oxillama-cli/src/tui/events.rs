//! TUI event types for the OxiLLaMa TUI chat interface.

/// Events that the TUI loop can receive.
///
/// Token, GenerationDone, and GenerationError are produced by the background
/// inference worker and consumed in the main draw loop via `event_rx.try_recv()`.
pub enum TuiEvent {
    /// A decoded token string emitted by the inference worker.
    Token(String),
    /// Inference completed successfully.
    ///
    /// Carries the KV cache occupancy observed immediately after the worker
    /// finished this turn, so the stats sidebar can show a real "KV usage"
    /// percentage instead of a value that is never assigned.
    GenerationDone {
        /// Current KV cache sequence length (tokens occupied).
        kv_seq_len: usize,
        /// The model's effective max context length (0 if unknown).
        max_context: usize,
    },
    /// Inference terminated with an error.
    GenerationError(String),
}
