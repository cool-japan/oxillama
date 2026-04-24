//! Multi-model LRU warm-pool.
//!
//! The `ModelPool` holds up to `capacity` `InferenceEngine` instances in a
//! `HashMap` keyed by model identifier. When a request arrives the pool's
//! `acquire` method is called: the model is returned immediately if already
//! loaded; otherwise it is loaded from disk (evicting the LRU entry if the
//! pool is full or over its memory budget) and then returned.
//!
//! Architecture note: The pool itself is owned by the single inference worker
//! thread.  All mutations happen on that thread — no `Arc<Mutex<...>>` is
//! needed around the pool itself.  `Arc<RwLock<LoadedModel>>` is used for the
//! per-model handle so that future multi-worker scenarios can share the engine
//! while one caller holds a read lock.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use oxillama_runtime::engine::{EngineConfig, InferenceEngine};

use crate::error::{ServerError, ServerResult};
use crate::router::eviction::LruQueue;

/// Identifier type alias for model IDs.
pub type ModelId = String;

/// A single loaded model with its engine and bookkeeping data.
pub struct LoadedModel {
    /// The owned inference engine.
    pub engine: InferenceEngine,
    /// Monotonic timestamp of the last request that used this model.
    pub last_used: Instant,
    /// Estimated resident memory in bytes:
    /// `weights_size + max_batch * (kv_size_per_seq + state_size_per_seq)`.
    ///
    /// Used by the pool to enforce the memory budget.
    pub mem_bytes: usize,
    /// Number of requests currently using this model.
    pub inflight: u64,
}

// Manual Debug impl because InferenceEngine does not derive Debug.
impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("last_used", &self.last_used)
            .field("mem_bytes", &self.mem_bytes)
            .field("inflight", &self.inflight)
            .finish_non_exhaustive()
    }
}

/// Status of a model in the pool (used by the Admin API).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelStatus {
    /// Model identifier.
    pub id: String,
    /// Load state.
    pub status: ModelLoadStatus,
    /// Estimated memory footprint in bytes (0 if not yet loaded).
    pub mem_bytes: usize,
    /// Last-used timestamp (seconds since UNIX epoch, 0 if never used).
    pub last_used_secs: u64,
    /// Number of requests currently using this model.
    pub inflight: u64,
}

/// Load state of a model entry.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStatus {
    /// The model is being loaded in a background task.
    Loading,
    /// The model is loaded and ready for inference.
    Ready,
    /// Loading failed; the model cannot be used.
    Failed,
}

/// Resolved model path + optional quantisation hint for a named model.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Filesystem path to the `.gguf` file.
    pub path: PathBuf,
    /// Quantisation hint (e.g. `"q4_0"`).  Currently informational only.
    pub quant: Option<String>,
}

/// A registry that maps model IDs to filesystem specs.
///
/// Used by `ModelPool::acquire` to locate models it has not yet loaded.
pub struct ModelLoader {
    registry: HashMap<ModelId, ModelSpec>,
    /// Default context size to pass to the engine.
    pub default_context_size: Option<usize>,
    /// Default thread count.
    pub default_num_threads: usize,
}

impl ModelLoader {
    /// Create a new loader with no registered models.
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            default_context_size: None,
            default_num_threads: 4,
        }
    }

    /// Register a model ID → spec mapping so the pool can load it on demand.
    pub fn register(&mut self, id: impl Into<String>, spec: ModelSpec) {
        self.registry.insert(id.into(), spec);
    }

    /// Look up the spec for a model ID.
    pub fn lookup(&self, id: &str) -> Option<&ModelSpec> {
        self.registry.get(id)
    }

    /// Build an `EngineConfig` for the given model spec.
    pub fn build_engine_config(&self, id: &str, spec: &ModelSpec) -> EngineConfig {
        tracing::debug!(model_id = id, path = %spec.path.display(), "building engine config");
        EngineConfig {
            model_path: spec.path.to_string_lossy().into_owned(),
            context_size: self.default_context_size,
            num_threads: self.default_num_threads,
            ..EngineConfig::default()
        }
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-model pending-load state (used when async background loading is active).
#[derive(Debug, Clone, PartialEq)]
pub enum PendingStatus {
    Loading,
    Failed(String),
}

/// Entry in the pool for a model that is still loading (or failed).
pub struct PendingEntry {
    pub status: PendingStatus,
    pub mem_bytes: usize,
}

/// The multi-model LRU warm-pool.
///
/// Owned by the inference worker thread; no `Send + Sync` requirement for the
/// `HashMap` internals because all accesses happen on one thread.
pub struct ModelPool {
    /// Live engines, keyed by model ID.
    loaded: HashMap<ModelId, Arc<RwLock<LoadedModel>>>,
    /// Models being loaded in background or that failed to load.
    pending: HashMap<ModelId, PendingEntry>,
    /// LRU ordering of loaded model IDs.
    lru: Mutex<LruQueue>,
    /// Maximum number of concurrently loaded models.
    capacity: usize,
    /// Maximum total memory budget in bytes (0 = unlimited).
    mem_budget_bytes: usize,
    /// Internal model loader (registry of id → spec mappings).
    loader: ModelLoader,
}

impl ModelPool {
    /// Create a new empty pool.
    ///
    /// - `capacity`: maximum number of models that may be resident at once.
    /// - `mem_budget_mb`: memory budget in MiB (0 = unlimited).
    pub fn new(capacity: usize, mem_budget_mb: usize) -> Self {
        Self {
            loaded: HashMap::with_capacity(capacity),
            pending: HashMap::new(),
            lru: Mutex::new(LruQueue::with_capacity(capacity)),
            capacity,
            mem_budget_bytes: mem_budget_mb.saturating_mul(1024 * 1024),
            loader: ModelLoader::new(),
        }
    }

    /// Register a model spec so it can be loaded on demand via `acquire`.
    ///
    /// Called by the admin `POST /admin/models/load` route before initiating a
    /// background load, and at startup for models listed in `[router] preload`.
    pub fn loader_register(&mut self, id: impl Into<String>, spec: ModelSpec) {
        self.loader.register(id, spec);
    }

    /// Access the embedded loader (read-only).
    pub fn loader(&self) -> &ModelLoader {
        &self.loader
    }

    /// Acquire an engine for `model_id`.
    ///
    /// If the model is already loaded it is promoted to the MRU position and
    /// its `Arc<RwLock<LoadedModel>>` is returned immediately.
    ///
    /// Otherwise the model is loaded synchronously (blocking the calling thread)
    /// after evicting LRU entries as needed.
    ///
    /// The optional `loader` parameter allows callers (like tests) to supply
    /// an external loader; `None` uses the pool's embedded loader.
    pub fn acquire(
        &mut self,
        model_id: &str,
        ext_loader: Option<&ModelLoader>,
    ) -> ServerResult<Arc<RwLock<LoadedModel>>> {
        // Fast path — already loaded.
        if let Some(handle) = self.loaded.get(model_id) {
            self.touch_lru(model_id);
            // Update inflight counter and last_used.
            if let Ok(mut guard) = handle.write() {
                guard.inflight = guard.inflight.saturating_add(1);
                guard.last_used = Instant::now();
            }
            return Ok(Arc::clone(handle));
        }

        // Choose loader: external takes precedence over embedded.
        // SAFETY: we know `self.loader` lives as long as `self`; to avoid
        // the borrow issue we clone the spec out.
        let spec = {
            let ldr = ext_loader.unwrap_or(&self.loader);
            ldr.lookup(model_id)
                .cloned()
                .ok_or_else(|| ServerError::InvalidRequest {
                    message: format!("model '{model_id}' is not registered"),
                })?
        };

        // Estimate memory before eviction so the budget check is accurate.
        let estimated_mem = estimate_mem_bytes(&spec.path);

        // Evict LRU models until we have room.
        self.evict_until_budget(estimated_mem)?;
        if self.loaded.len() >= self.capacity {
            self.evict_one()?;
        }

        // Load the engine synchronously.
        tracing::info!(model_id, "loading model into pool");
        let engine_config = self.loader.build_engine_config(model_id, &spec);
        let mut engine = InferenceEngine::new(engine_config);
        engine.load_model().map_err(ServerError::Runtime)?;
        tracing::info!(model_id, mem_bytes = estimated_mem, "model loaded");

        let handle = Arc::new(RwLock::new(LoadedModel {
            engine,
            last_used: Instant::now(),
            mem_bytes: estimated_mem,
            inflight: 1,
        }));

        self.loaded
            .insert(model_id.to_string(), Arc::clone(&handle));
        self.touch_lru(model_id);

        Ok(handle)
    }

    /// Decrement inflight count for a model when the caller is done with it.
    pub fn release(&self, model_id: &str) {
        if let Some(handle) = self.loaded.get(model_id) {
            if let Ok(mut guard) = handle.write() {
                guard.inflight = guard.inflight.saturating_sub(1);
            }
        }
    }

    /// Explicitly unload a model, freeing its memory.
    ///
    /// Returns an error if the model ID is not currently loaded.
    pub fn unload(&mut self, model_id: &str) -> ServerResult<()> {
        if self.loaded.remove(model_id).is_none() {
            return Err(ServerError::InvalidRequest {
                message: format!("model '{model_id}' is not loaded"),
            });
        }
        self.pending.remove(model_id);
        if let Ok(mut lru) = self.lru.lock() {
            lru.remove(model_id);
        }
        tracing::info!(model_id, "model unloaded from pool");
        Ok(())
    }

    /// List the status of all known models (loaded + pending).
    pub fn list(&self) -> Vec<ModelStatus> {
        let mut out = Vec::with_capacity(self.loaded.len() + self.pending.len());

        for (id, handle) in &self.loaded {
            let (mem_bytes, last_used_secs, inflight) = if let Ok(guard) = handle.read() {
                let secs = guard.last_used.elapsed().as_secs();
                (guard.mem_bytes, secs, guard.inflight)
            } else {
                (0, 0, 0)
            };
            out.push(ModelStatus {
                id: id.clone(),
                status: ModelLoadStatus::Ready,
                mem_bytes,
                last_used_secs,
                inflight,
            });
        }

        for (id, entry) in &self.pending {
            let status = match &entry.status {
                PendingStatus::Loading => ModelLoadStatus::Loading,
                PendingStatus::Failed(_) => ModelLoadStatus::Failed,
            };
            out.push(ModelStatus {
                id: id.clone(),
                status,
                mem_bytes: entry.mem_bytes,
                last_used_secs: 0,
                inflight: 0,
            });
        }

        out
    }

    /// Mark a model as being loaded in a background task.
    pub fn mark_loading(&mut self, model_id: impl Into<String>) {
        let id = model_id.into();
        self.pending.insert(
            id,
            PendingEntry {
                status: PendingStatus::Loading,
                mem_bytes: 0,
            },
        );
    }

    /// Mark a pending model as ready (called after a background load succeeds).
    ///
    /// Moves the engine from the temporary pending state into the loaded map.
    pub fn mark_ready(
        &mut self,
        model_id: &str,
        engine: InferenceEngine,
        mem_bytes: usize,
    ) -> ServerResult<()> {
        // Evict if needed.
        self.evict_until_budget(mem_bytes)?;
        if self.loaded.len() >= self.capacity {
            self.evict_one()?;
        }

        let handle = Arc::new(RwLock::new(LoadedModel {
            engine,
            last_used: Instant::now(),
            mem_bytes,
            inflight: 0,
        }));
        self.loaded
            .insert(model_id.to_string(), Arc::clone(&handle));
        self.pending.remove(model_id);
        self.touch_lru(model_id);
        Ok(())
    }

    /// Mark a pending model as failed to load.
    pub fn mark_failed(&mut self, model_id: &str, reason: String) {
        if let Some(entry) = self.pending.get_mut(model_id) {
            entry.status = PendingStatus::Failed(reason);
        }
    }

    /// Total estimated bytes currently consumed by loaded models.
    pub fn current_mem_bytes(&self) -> usize {
        self.loaded
            .values()
            .filter_map(|h| h.read().ok().map(|g| g.mem_bytes))
            .sum()
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn touch_lru(&self, model_id: &str) {
        if let Ok(mut lru) = self.lru.lock() {
            lru.touch(model_id);
        }
    }

    /// Evict LRU models until `current + needed <= budget` (or budget is 0).
    fn evict_until_budget(&mut self, needed_bytes: usize) -> ServerResult<()> {
        if self.mem_budget_bytes == 0 {
            return Ok(());
        }
        while self.current_mem_bytes() + needed_bytes > self.mem_budget_bytes {
            self.evict_one().map_err(|_| ServerError::InvalidRequest {
                message: "memory budget exceeded and no evictable model found".to_string(),
            })?;
        }
        Ok(())
    }

    /// Evict the single LRU model.
    fn evict_one(&mut self) -> ServerResult<()> {
        let victim = {
            let mut lru = self.lru.lock().map_err(|_| ServerError::InvalidRequest {
                message: "LRU queue lock poisoned".to_string(),
            })?;
            lru.evict_lru()
        };

        let victim = victim.ok_or_else(|| ServerError::InvalidRequest {
            message: "no model to evict — pool is empty".to_string(),
        })?;

        // Don't evict a model that has in-flight requests.
        let inflight = self
            .loaded
            .get(&victim)
            .and_then(|h| h.read().ok().map(|g| g.inflight))
            .unwrap_or(0);

        if inflight > 0 {
            // Return the model to the LRU queue so it isn't lost.
            self.touch_lru(&victim);
            return Err(ServerError::InvalidRequest {
                message: format!("cannot evict '{victim}': {inflight} request(s) in flight"),
            });
        }

        tracing::info!(model_id = %victim, "evicting model from pool (LRU)");
        self.loaded.remove(&victim);
        Ok(())
    }
}

/// Rough memory estimate based on the file size of the GGUF on disk.
///
/// The weights are memory-mapped, so the on-disk size approximates
/// resident memory.  We add a 64 MiB overhead for KV cache + buffers.
fn estimate_mem_bytes(path: &std::path::Path) -> usize {
    const KV_OVERHEAD: usize = 64 * 1024 * 1024;
    let file_size = std::fs::metadata(path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    file_size.saturating_add(KV_OVERHEAD)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) pool_single_model_routes: manual insert; acquire same model twice;
    ///     second call returns same Arc (pointer equality).
    #[test]
    fn pool_single_model_routes() {
        let mut pool = ModelPool::new(2, 0); // 0 = unlimited budget

        // Manually insert a fake loaded model so we don't need a real file.
        let engine = InferenceEngine::new(EngineConfig::default());
        let handle = Arc::new(RwLock::new(LoadedModel {
            engine,
            last_used: Instant::now(),
            mem_bytes: 0,
            inflight: 0,
        }));
        pool.loaded
            .insert("model-a".to_string(), Arc::clone(&handle));
        pool.touch_lru("model-a");

        // First acquire (no external loader needed — already loaded)
        let h1 = pool.acquire("model-a", None).expect("first acquire");
        // Second acquire — should be the same Arc
        let h2 = pool.acquire("model-a", None).expect("second acquire");

        assert!(
            Arc::ptr_eq(&h1, &h2),
            "both acquires should return the same Arc"
        );
    }

    /// (b) pool_evicts_when_over_capacity: capacity=1; insert model-a;
    ///     insert model-b manually; model-a should be evicted.
    #[test]
    fn pool_evicts_when_over_capacity() {
        let mut pool = ModelPool::new(1, 0); // capacity = 1

        // Insert model-a
        let engine_a = InferenceEngine::new(EngineConfig::default());
        let handle_a = Arc::new(RwLock::new(LoadedModel {
            engine: engine_a,
            last_used: Instant::now(),
            mem_bytes: 0,
            inflight: 0,
        }));
        pool.loaded.insert("model-a".to_string(), handle_a);
        pool.touch_lru("model-a");

        // Now insert model-b via mark_ready — should evict model-a.
        let engine_b = InferenceEngine::new(EngineConfig::default());
        pool.mark_ready("model-b", engine_b, 0)
            .expect("mark_ready should succeed after evicting model-a");

        assert!(
            !pool.loaded.contains_key("model-a"),
            "model-a should have been evicted"
        );
        assert!(
            pool.loaded.contains_key("model-b"),
            "model-b should now be loaded"
        );
    }

    /// (c) pool_unknown_model_returns_error: acquire a model that was never
    ///     registered; expect Err with a descriptive message.
    #[test]
    fn pool_unknown_model_returns_error() {
        let mut pool = ModelPool::new(4, 0);
        // No spec registered in pool's embedded loader, no external loader.

        let err = pool.acquire("unknown-model", None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not registered"),
            "error should mention 'not registered': {msg}"
        );
    }

    /// (d) pool_list_shows_loaded: insert two models; pool.list() returns both.
    #[test]
    fn pool_list_shows_loaded() {
        let mut pool = ModelPool::new(4, 0);

        for name in ["model-x", "model-y"] {
            let engine = InferenceEngine::new(EngineConfig::default());
            let handle = Arc::new(RwLock::new(LoadedModel {
                engine,
                last_used: Instant::now(),
                mem_bytes: 1024,
                inflight: 0,
            }));
            pool.loaded.insert(name.to_string(), handle);
            pool.touch_lru(name);
        }

        let statuses = pool.list();
        assert_eq!(statuses.len(), 2, "list should report both models");
        let ids: Vec<&str> = statuses.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"model-x"), "model-x should appear in list");
        assert!(ids.contains(&"model-y"), "model-y should appear in list");
        for s in &statuses {
            assert_eq!(s.status, ModelLoadStatus::Ready);
            assert_eq!(s.mem_bytes, 1024);
        }
    }

    /// LRU eviction order test: insert 3 models with capacity=3; touch the
    /// first two; the third (oldest) should be evicted first.
    #[test]
    fn pool_lru_ordering() {
        let mut pool = ModelPool::new(3, 0);

        for name in ["alpha", "beta", "gamma"] {
            let engine = InferenceEngine::new(EngineConfig::default());
            let handle = Arc::new(RwLock::new(LoadedModel {
                engine,
                last_used: Instant::now(),
                mem_bytes: 0,
                inflight: 0,
            }));
            pool.loaded.insert(name.to_string(), handle);
            pool.touch_lru(name);
        }

        // Touch alpha and beta → gamma is now LRU.
        pool.touch_lru("alpha");
        pool.touch_lru("beta");

        pool.evict_one().expect("should evict gamma");
        assert!(
            !pool.loaded.contains_key("gamma"),
            "gamma should have been evicted"
        );
        assert!(pool.loaded.contains_key("alpha"), "alpha should remain");
        assert!(pool.loaded.contains_key("beta"), "beta should remain");
    }

    /// Memory-budget eviction: use mark_ready then mark_ready a second model
    /// to exceed capacity=1 which forces LRU eviction.
    ///
    /// We exercise the budget path by setting a tiny budget and using
    /// mark_ready (which calls evict_until_budget + evict_one internally).
    #[test]
    fn pool_evicts_when_over_budget() {
        // Budget = 1 MiB; capacity = 1 so first mark_ready fills the slot.
        let mut pool = ModelPool::new(1, 1); // 1-MiB budget, capacity=1

        // Insert "big-model" using mark_ready with 0 bytes (fits in 1 MiB).
        let engine_a = InferenceEngine::new(EngineConfig::default());
        pool.mark_ready("big-model", engine_a, 0)
            .expect("first mark_ready should succeed");

        assert!(
            pool.loaded.contains_key("big-model"),
            "big-model should be in pool after mark_ready"
        );

        // Now insert a second model — capacity=1 forces eviction of big-model.
        let engine_b = InferenceEngine::new(EngineConfig::default());
        pool.mark_ready("small-model", engine_b, 0)
            .expect("second mark_ready should evict big-model and succeed");

        assert!(
            !pool.loaded.contains_key("big-model"),
            "big-model should have been evicted when small-model was loaded"
        );
        assert!(
            pool.loaded.contains_key("small-model"),
            "small-model should now be in the pool"
        );
    }
}
