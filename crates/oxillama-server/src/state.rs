//! Shared application state for the API server.
//!
//! `AppState` no longer holds the `InferenceEngine` directly; instead it
//! carries a sender end of the request queue plus read-only fields that
//! were previously obtained by locking the engine at request time.

use std::sync::Arc;
use std::time::SystemTime;

use tokio::sync::mpsc;

use oxillama_runtime::sampling::SamplerConfig;

use crate::metrics::Metrics;
use crate::queue::{BatchRequest, VocabBytes};

/// Shared application state accessible by all route handlers.
///
/// All inference is delegated to the single background worker via `queue`.
/// Read-only metadata (model ID, default sampler, vocabulary, hidden size)
/// is cached here so handlers never need to reach into the engine.
pub struct AppState {
    /// Channel to send inference requests to the worker.
    pub queue: mpsc::Sender<BatchRequest>,

    /// The model name/identifier for API responses.
    pub model_id: String,

    /// Unix timestamp (seconds) when the model was loaded.
    pub loaded_at: u64,

    /// Default sampler configuration read from `EngineConfig` at startup.
    ///
    /// Route handlers clone this and apply per-request overrides on top.
    pub default_sampler: SamplerConfig,

    /// Vocabulary byte table used for grammar-constrained sampling.
    ///
    /// `None` when the model has no tokenizer (should not happen at serve time).
    pub vocab_bytes: Option<VocabBytes>,

    /// Hidden-state dimension for the `/v1/embeddings` endpoint.
    pub hidden_size: usize,

    /// Shared metrics store.
    pub metrics: Arc<Metrics>,
}

impl AppState {
    /// Create new app state from all required fields.
    ///
    /// `queue` must be connected to a live inference worker.
    pub fn new(
        queue: mpsc::Sender<BatchRequest>,
        model_id: String,
        default_sampler: SamplerConfig,
        vocab_bytes: Option<VocabBytes>,
        hidden_size: usize,
    ) -> Self {
        let loaded_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            queue,
            model_id,
            loaded_at,
            default_sampler,
            vocab_bytes,
            hidden_size,
            metrics: Arc::new(Metrics::new()),
        }
    }
}
