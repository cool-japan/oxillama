//! Shared application state for the API server.
//!
//! `AppState` carries all read/write shared data needed by route handlers:
//! - The inference request queue (mpsc sender).
//! - Cached model metadata (id, sampler, vocab, hidden size).
//! - Metrics store.
//! - In-memory batch store (legacy).
//! - Disk-backed batch store + queue sender (C3).
//! - Multi-model LRU pool (C1), protected by a `Mutex` for admin mutations.
//! - Prefix KV cache for system-prompt reuse across requests.
//! - LoRA adapter registry (name → `Arc<LoadedLora>`).
//! - Persistent thread store + run queue (Assistants API).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use tokio::sync::mpsc;

use crate::batch::{new_batch_store, BatchStore};
use crate::batch_spool::{BatchQueueSender, BatchStore as DiskBatchStore};
use crate::error::{ServerError, ServerResult};
use crate::files_store::FilesStore;
use crate::metrics::Metrics;
use crate::prefix_registry::PrefixCacheRegistry;
use crate::queue::{BatchRequest, VocabBytes};
use crate::rate_limit::PerKeyRateLimiter;
use crate::responses_store::ResponseStore;
use crate::router::ModelPool;
use crate::threads::stream::RunEventSender;
use crate::threads::{RunQueueSender, ThreadStore};

use oxillama_runtime::sampling::SamplerConfig;
use oxillama_runtime::{ChatTemplate, GpuStatus, LoadedLora};

/// Compute-backend facts reported by `GET /admin/health`.
///
/// Reporting only: nothing here influences inference. The engine is loaded by
/// the CLI before the server starts, so the server can describe the backend
/// but never selects it.
///
/// A `None` field means "not applicable / unknown", which is the shape the CPU
/// path always produces (`gpu_enabled == false`, everything else `None`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendInfo {
    /// Whether a GPU device was successfully initialised for the loaded model.
    ///
    /// `true` with `resident_tensors == Some(0)` is a valid state: a device
    /// exists but nothing met the offload eligibility rules.
    pub gpu_enabled: bool,
    /// Adapter name as reported by the GPU backend.
    pub device_name: Option<String>,
    /// Adapter backend family (`Metal`, `Vulkan`, `Dx12`, …).
    pub backend: Option<String>,
    /// Tensors whose weights reside on the device.
    pub resident_tensors: Option<usize>,
    /// Device bytes those tensors occupy, quantised (not dequantised).
    pub resident_bytes: Option<u64>,
}

impl From<&GpuStatus> for BackendInfo {
    /// A `GpuStatus` only exists when a device was initialised, so
    /// `gpu_enabled` is unconditionally `true`.
    ///
    /// This conversion lives here because the orphan rule forbids any crate
    /// downstream of both `oxillama-server` and `oxillama-runtime` from
    /// writing it.
    fn from(status: &GpuStatus) -> Self {
        Self {
            gpu_enabled: true,
            device_name: Some(status.device_name.clone()),
            backend: Some(status.backend.clone()),
            resident_tensors: Some(status.resident_tensors),
            resident_bytes: Some(status.resident_bytes),
        }
    }
}

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

    /// The loaded model's chat template family, detected **once** at model
    /// load time from its own GGUF metadata
    /// ([`InferenceEngine::chat_template`](oxillama_runtime::InferenceEngine::chat_template)).
    ///
    /// Every chat-shaped route renders its turns through this rather than the
    /// fabricated `<|system|>…<|end|>` skeleton the server used to hardcode
    /// for every model. That skeleton belonged to no real model: Qwen3, whose
    /// real markers are ChatML's `<|im_start|>`/`<|im_end|>`, would tokenize
    /// `<|end|>` as ordinary text and imitate it back as literal output
    /// (`"Hello! <|end|#>"`).
    ///
    /// Detection is a pure function of the model file, so caching it here
    /// costs one resolve at startup and nothing per request.
    pub chat_template: ChatTemplate,

    /// Hidden-state dimension for the `/v1/embeddings` endpoint.
    pub hidden_size: usize,

    /// Shared metrics store.
    pub metrics: Arc<Metrics>,

    /// In-memory batch job registry (legacy OpenAI batch compat layer).
    pub batch_store: BatchStore,

    /// Disk-backed batch job store (C3: disk-spool backend).
    pub batch_disk_store: Arc<DiskBatchStore>,

    /// Sender into the disk-backed batch processing queue (C3).
    ///
    /// `None` when batch processing has not been wired up (D10 fix): the
    /// previous design always constructed a capacity-1 channel and
    /// immediately dropped the receiver, so every send silently failed
    /// with a permanently-closed channel — this distinguishes "not
    /// configured" (503 for the caller) from "actually broken".
    pub batch_queue_tx: Option<BatchQueueSender>,

    /// Multi-model LRU warm-pool (C1).
    ///
    /// Wrapped in `Mutex` so admin routes can mutate it without blocking the
    /// inference worker. In the current single-worker design the worker also
    /// holds the pool; admin mutations use `try_lock` to avoid deadlocks.
    pub model_pool: Mutex<ModelPool>,

    /// Namespaced, bounded registry of prefix KV caches for system-prompt
    /// reuse across requests (D6 fix).
    ///
    /// When a new request shares a long prefix with a previously-cached
    /// sequence (e.g. a fixed system prompt) *within the same namespace*
    /// (model + exact LoRA adapter set/scale), the matching KV state is
    /// restored and only the suffix tokens need a fresh prefill pass.
    /// Namespacing prevents a LoRA'd request's cache entry from ever being
    /// served to a plain (or differently-LoRA'd) request — see
    /// [`crate::prefix_registry`].
    pub prefix_cache_registry: Arc<PrefixCacheRegistry>,

    /// Loaded LoRA adapter registry: stable name → `Arc<LoadedLora>`.
    ///
    /// Populated via `POST /admin/loras`.  Request handlers look up adapters
    /// by name and pass them to the worker via `BatchRequest::Generate`.
    pub loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>,

    /// Persistent thread/message/run store for the Assistants API.
    ///
    /// `None` when the Assistants API has not been configured (no `--threads-dir`
    /// flag was passed at startup).  Route handlers return 503 in this case.
    pub threads_store: Option<Arc<ThreadStore>>,

    /// Sender into the run processing queue for the Assistants API.
    ///
    /// `None` when `threads_store` is `None`.
    pub run_queue_tx: Option<RunQueueSender>,

    /// Persistent files store for the Files API (`/v1/files`).
    ///
    /// `None` when the Files API has not been configured.
    pub files_store: Option<Arc<FilesStore>>,

    /// Broadcast sender for run lifecycle events (SSE streaming).
    ///
    /// `None` when SSE streaming is not enabled.
    pub run_event_tx_broadcast: Option<RunEventSender>,

    /// In-memory store for Responses API objects.
    ///
    /// `None` when the Responses API has not been enabled.  Route handlers
    /// return 503 (`ModelNotReady`) in this case.
    pub responses_store: Option<Arc<ResponseStore>>,

    /// Per-API-key token-bucket rate limiter.
    ///
    /// `None` when per-key rate limiting has not been configured.
    pub per_key_rate_limiter: Option<Arc<PerKeyRateLimiter>>,

    /// Liveness flag for the inference worker (D7 fix).
    ///
    /// Set to `true` while `run_worker` is executing its main loop, and to
    /// `false` only when the worker's receiver channel closes (i.e. the
    /// worker has actually exited, not merely caught a panic mid-request —
    /// panics are caught per-request and the loop continues). `GET /ready`
    /// reads this to report 503 instead of always returning 200 regardless
    /// of whether anything is actually able to serve requests.
    pub worker_alive: Arc<AtomicBool>,

    /// Directories that admin-supplied model/LoRA `path` fields are
    /// allowed to resolve into (D1 fix). Empty = no restriction
    /// (backward-compatible default); see `crate::admin::path_guard`.
    pub allowed_model_dirs: Vec<PathBuf>,

    /// Compute-backend facts for `GET /admin/health` (B3).
    ///
    /// `None` means the process never reported a backend; the admin route
    /// then reports the CPU shape (`{"gpu_enabled": false}`). Populated by
    /// the CLI via [`AppState::with_backend_info`] after it loads the engine.
    backend_info: Option<BackendInfo>,
}

impl AppState {
    /// Create new app state from all required fields.
    ///
    /// `queue` must be connected to a live inference worker.
    ///
    /// `prefix_cache_registry` and `worker_alive` should be the *same*
    /// `Arc`s passed to [`crate::worker::spawn_inference_worker`], so the
    /// worker and the HTTP layer observe/mutate shared state rather than
    /// independent copies.
    ///
    /// `batch_spool_dir` selects where disk-spooled batch jobs are stored;
    /// `None` falls back to `std::env::temp_dir().join("oxillama_batch_spool")`.
    /// Fails with `ServerError::IoError` if that directory cannot be created,
    /// instead of panicking (D10 fix — this used to `.expect()`).
    ///
    /// `chat_template` is a **required** argument rather than a builder
    /// method on purpose: getting it wrong silently corrupts every chat
    /// request (the model is prompted in a format it was never trained on),
    /// so a caller that loads a model must be forced by the compiler to say
    /// which template that model uses. Pass
    /// `engine.chat_template().unwrap_or_default()`; a caller with no model
    /// at all (tests, embedders) can pass [`ChatTemplate::default`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queue: mpsc::Sender<BatchRequest>,
        model_id: String,
        default_sampler: SamplerConfig,
        vocab_bytes: Option<VocabBytes>,
        hidden_size: usize,
        chat_template: ChatTemplate,
        prefix_cache_registry: Arc<PrefixCacheRegistry>,
        worker_alive: Arc<AtomicBool>,
        batch_spool_dir: Option<PathBuf>,
    ) -> ServerResult<Self> {
        let loaded_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let spool_dir =
            batch_spool_dir.unwrap_or_else(|| std::env::temp_dir().join("oxillama_batch_spool"));
        let batch_disk_store =
            Arc::new(
                DiskBatchStore::new(spool_dir.clone()).map_err(|e| ServerError::IoError {
                    context: format!("create batch spool directory {}", spool_dir.display()),
                    source: e,
                })?,
            );

        Ok(Self {
            queue,
            model_id,
            loaded_at,
            default_sampler,
            vocab_bytes,
            chat_template,
            hidden_size,
            metrics: Arc::new(Metrics::new()),
            batch_store: new_batch_store(),
            batch_disk_store,
            batch_queue_tx: None,
            model_pool: Mutex::new(ModelPool::new(4, 0)),
            prefix_cache_registry,
            loras: Arc::new(RwLock::new(HashMap::new())),
            threads_store: None,
            run_queue_tx: None,
            files_store: None,
            run_event_tx_broadcast: None,
            responses_store: None,
            per_key_rate_limiter: None,
            worker_alive,
            allowed_model_dirs: Vec::new(),
            backend_info: None,
        })
    }

    /// Attach a threads store and run queue to this `AppState`.
    ///
    /// Returns `self` with the `threads_store` and `run_queue_tx` fields
    /// populated.  Designed for use in a builder chain:
    ///
    /// ```text
    /// let state = AppState::new(...).with_threads(store, tx);
    /// ```
    pub fn with_threads(mut self, store: Arc<ThreadStore>, tx: RunQueueSender) -> Self {
        self.threads_store = Some(store);
        self.run_queue_tx = Some(tx);
        self
    }

    /// Attach a files store to this `AppState`.
    pub fn with_files(mut self, store: Arc<FilesStore>) -> Self {
        self.files_store = Some(store);
        self
    }

    /// Attach a run-event broadcast sender to this `AppState`.
    ///
    /// When set, the run worker broadcasts lifecycle events that SSE handlers
    /// can subscribe to.
    pub fn with_run_event_sender(mut self, tx: RunEventSender) -> Self {
        self.run_event_tx_broadcast = Some(tx);
        self
    }

    /// Attach a Responses API store to this `AppState`.
    ///
    /// When set, the `/v1/responses` routes are fully operational.
    pub fn with_responses_store(mut self, store: Arc<ResponseStore>) -> Self {
        self.responses_store = Some(store);
        self
    }

    /// Attach a per-API-key rate limiter to this `AppState`.
    ///
    /// When set, the `per_key_rate_limit_middleware` is applied to all routes
    /// in `build_app_with_config`.
    pub fn with_per_key_rate_limiter(mut self, limiter: Arc<PerKeyRateLimiter>) -> Self {
        self.per_key_rate_limiter = Some(limiter);
        self
    }

    /// Attach a batch-processing queue sender to this `AppState` (D10 fix).
    ///
    /// Until this is called, `batch_queue_tx` stays `None` and
    /// `POST /v1/batch_jobs` returns 503 rather than accepting jobs that a
    /// permanently-closed channel can never deliver to a worker. Call this
    /// after spawning the disk-batch worker (see `batch_spool::spawn_batch_worker`).
    pub fn with_batch_queue(mut self, tx: BatchQueueSender) -> Self {
        self.batch_queue_tx = Some(tx);
        self
    }

    /// Attach an admin path allow-list to this `AppState` (D1 fix).
    ///
    /// When set (non-empty), `path` fields in admin requests
    /// (`POST /admin/models/load`, `POST /admin/loras`) must canonicalize
    /// into one of these directories or the request is rejected with 400.
    pub fn with_allowed_model_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.allowed_model_dirs = dirs;
        self
    }

    /// Attach compute-backend facts to this `AppState` (B3).
    ///
    /// Purely descriptive: `GET /admin/health` reports this and nothing else
    /// reads it. Callers pass what the already-loaded engine actually got,
    /// not what was requested.
    pub fn with_backend_info(mut self, info: BackendInfo) -> Self {
        self.backend_info = Some(info);
        self
    }

    /// Compute-backend facts, or `None` when none were reported.
    pub fn backend_info(&self) -> Option<&BackendInfo> {
        self.backend_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix_registry::DEFAULT_MAX_NAMESPACES;
    use oxillama_runtime::PrefixCacheConfig;
    use uuid::Uuid;

    fn fresh_state() -> ServerResult<AppState> {
        let (tx, _rx) = mpsc::channel::<BatchRequest>(8);
        let registry = Arc::new(PrefixCacheRegistry::new(
            PrefixCacheConfig::default(),
            DEFAULT_MAX_NAMESPACES,
        ));
        let worker_alive = Arc::new(AtomicBool::new(true));
        let spool_dir = std::env::temp_dir().join(format!(
            "oxillama_state_test_{}",
            Uuid::new_v4().as_simple()
        ));
        AppState::new(
            tx,
            "test-model".to_string(),
            SamplerConfig::default(),
            None,
            0,
            ChatTemplate::default(),
            registry,
            worker_alive,
            Some(spool_dir),
        )
    }

    #[test]
    fn app_state_new_succeeds() {
        assert!(fresh_state().is_ok());
    }

    #[test]
    fn app_state_new_has_no_batch_queue_by_default() {
        let state = fresh_state().expect("fresh_state should succeed");
        assert!(
            state.batch_queue_tx.is_none(),
            "batch_queue_tx must start unset — D10 fix"
        );
    }

    #[test]
    fn with_batch_queue_sets_sender() {
        let state = fresh_state().expect("fresh_state should succeed");
        let (tx, _rx) = tokio::sync::mpsc::channel::<crate::batch_spool::BatchWorkItem>(4);
        let state = state.with_batch_queue(tx);
        assert!(state.batch_queue_tx.is_some());
    }

    #[test]
    fn app_state_new_honours_batch_spool_dir() {
        let dir = std::env::temp_dir().join(format!(
            "oxillama_state_test_honour_{}",
            Uuid::new_v4().as_simple()
        ));
        let (tx, _rx) = mpsc::channel::<BatchRequest>(8);
        let registry = Arc::new(PrefixCacheRegistry::new(
            PrefixCacheConfig::default(),
            DEFAULT_MAX_NAMESPACES,
        ));
        let worker_alive = Arc::new(AtomicBool::new(true));
        let _state = AppState::new(
            tx,
            "test-model".to_string(),
            SamplerConfig::default(),
            None,
            0,
            ChatTemplate::default(),
            registry,
            worker_alive,
            Some(dir.clone()),
        )
        .expect("AppState::new should succeed");
        assert!(dir.exists(), "configured batch_spool_dir must be created");
    }

    #[test]
    fn app_state_new_has_no_backend_info_by_default() {
        let state = fresh_state().expect("fresh_state should succeed");
        assert!(state.backend_info().is_none());
    }

    #[test]
    fn with_backend_info_is_readable_through_accessor() {
        let state = fresh_state()
            .expect("fresh_state should succeed")
            .with_backend_info(BackendInfo {
                gpu_enabled: true,
                device_name: Some("Adapter".to_string()),
                backend: Some("Vulkan".to_string()),
                resident_tensors: Some(3),
                resident_bytes: Some(1024),
            });
        let info = state.backend_info().expect("backend info should be set");
        assert!(info.gpu_enabled);
        assert_eq!(info.backend.as_deref(), Some("Vulkan"));
        assert_eq!(info.resident_bytes, Some(1024));
    }

    #[test]
    fn backend_info_from_gpu_status_is_enabled_even_when_idle() {
        let status = GpuStatus {
            device_name: "Adapter".to_string(),
            backend: "Metal".to_string(),
            resident_tensors: 0,
            resident_bytes: 0,
            cpu_tensors: 12,
            upload_failures: 0,
        };
        let info = BackendInfo::from(&status);
        assert!(
            info.gpu_enabled,
            "a GpuStatus means a device was initialised, even with 0 resident tensors"
        );
        assert_eq!(info.resident_tensors, Some(0));
        assert_eq!(info.device_name.as_deref(), Some("Adapter"));
    }

    #[test]
    fn builder_chain_composes() {
        let state = fresh_state()
            .expect("fresh_state should succeed")
            .with_files(Arc::new(
                FilesStore::new(std::env::temp_dir().join(format!(
                    "oxillama_state_test_files_{}",
                    Uuid::new_v4().as_simple()
                )))
                .expect("FilesStore::new"),
            ));
        assert!(state.files_store.is_some());
    }
}
