//! Shared application state for the API server.
//!
//! `AppState` carries all read/write shared data needed by route handlers:
//! - The inference request queue (mpsc sender).
//! - Cached model metadata (id, sampler, vocab, hidden size).
//! - Metrics store.
//! - In-memory batch store (legacy).
//! - Disk-backed batch store + queue sender (C3).
//! - Multi-model LRU pool (C1), protected by a `Mutex` for admin mutations.

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use tokio::sync::mpsc;

use crate::batch::{new_batch_store, BatchStore};
use crate::batch_spool::{BatchQueueSender, BatchStore as DiskBatchStore};
use crate::metrics::Metrics;
use crate::queue::{BatchRequest, VocabBytes};
use crate::router::ModelPool;

use oxillama_runtime::sampling::SamplerConfig;

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

    /// In-memory batch job registry (legacy OpenAI batch compat layer).
    pub batch_store: BatchStore,

    /// Disk-backed batch job store (C3: disk-spool backend).
    pub batch_disk_store: Arc<DiskBatchStore>,

    /// Sender into the disk-backed batch processing queue (C3).
    pub batch_queue_tx: BatchQueueSender,

    /// Multi-model LRU warm-pool (C1).
    ///
    /// Wrapped in `Mutex` so admin routes can mutate it without blocking the
    /// inference worker. In the current single-worker design the worker also
    /// holds the pool; admin mutations use `try_lock` to avoid deadlocks.
    pub model_pool: Mutex<ModelPool>,
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

        // Default disk store goes to a temp-dir location when not configured.
        let spool_dir = std::env::temp_dir().join("oxillama_batch_spool");
        let batch_disk_store = Arc::new(DiskBatchStore::new(spool_dir).unwrap_or_else(|_| {
            DiskBatchStore::new(std::env::temp_dir()).expect("fallback spool dir")
        }));

        // Create a no-op batch queue (capacity 0 → sends will fail gracefully).
        let (batch_queue_tx, _) =
            tokio::sync::mpsc::channel::<crate::batch_spool::BatchWorkItem>(1);

        Self {
            queue,
            model_id,
            loaded_at,
            default_sampler,
            vocab_bytes,
            hidden_size,
            metrics: Arc::new(Metrics::new()),
            batch_store: new_batch_store(),
            batch_disk_store,
            batch_queue_tx,
            model_pool: Mutex::new(ModelPool::new(4, 0)),
        }
    }

    /// Create app state with an explicit disk batch store and queue sender.
    ///
    /// Used by the server startup code to wire up the full batch pipeline.
    pub fn with_batch_pipeline(
        queue: mpsc::Sender<BatchRequest>,
        model_id: String,
        default_sampler: SamplerConfig,
        vocab_bytes: Option<VocabBytes>,
        hidden_size: usize,
        batch_disk_store: Arc<DiskBatchStore>,
        batch_queue_tx: BatchQueueSender,
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
            batch_store: new_batch_store(),
            batch_disk_store,
            batch_queue_tx,
            model_pool: Mutex::new(ModelPool::new(4, 0)),
        }
    }
}
