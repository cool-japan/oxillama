//! Multi-model LRU warm-pool router.
//!
//! The router module provides:
//!
//! - [`ModelPool`] — a fixed-capacity cache of loaded [`oxillama_runtime::InferenceEngine`]
//!   instances, with LRU eviction under memory pressure.
//! - [`ModelLoader`] — resolves model identifiers to filesystem paths and
//!   builds [`oxillama_runtime::EngineConfig`] values.
//! - [`eviction::LruQueue`] — a standalone LRU ordering helper.
//!
//! ## Design
//!
//! The pool is owned entirely by the inference worker thread (see
//! `worker.rs`). Route handlers place a `BatchRequest` (with an attached
//! `model_id`) onto the `mpsc` queue; the worker calls `pool.acquire` on
//! its thread, which may block while loading from disk.
//!
//! This keeps all mutable state off the async runtime threads and avoids
//! any `Mutex` around `InferenceEngine` itself — the only locks are the
//! per-model `RwLock<LoadedModel>` (for inflight tracking) and the
//! `Mutex<LruQueue>` (for LRU ordering).

pub mod eviction;
pub mod pool;

pub use pool::{
    LoadedModel, ModelId, ModelLoadStatus, ModelLoader, ModelPool, ModelSpec, ModelStatus,
    PendingStatus,
};
