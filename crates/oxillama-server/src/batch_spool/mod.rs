//! Disk-spooled OpenAI Batch API backend.
//!
//! This module provides the production batch processing implementation:
//! - `store` — atomic disk I/O for job metadata and results.
//! - `queue` — tokio mpsc work queue connecting route handlers to the worker.
//! - `worker` — background task that processes jobs line-by-line.
//! - `routes` — axum HTTP handlers for `/v1/batches/*`.

pub mod queue;
pub mod routes;
pub mod store;
pub mod worker;

pub use queue::{new_batch_queue, BatchQueueSender, BatchWorkItem};
pub use routes::{cancel_batch, create_batch, get_batch, get_batch_output, list_batches};
pub use store::{BatchJobMeta, BatchJobStatus, BatchStore};
pub use worker::spawn_batch_worker;
