//! OpenAI Assistants v2 API — threads, messages, runs, steps, and SSE streaming.
//!
//! This module provides persistent thread/message/run/step storage and the HTTP
//! handlers that expose them as OpenAI-compatible endpoints.
//!
//! ## Sub-modules
//!
//! | Module | Role |
//! |--------|------|
//! | `types`  | Wire-format types (`Thread`, `ThreadMessage`, `Run`, `RunStep`, request bodies) |
//! | `store`  | Atomic disk I/O (`ThreadStore`) |
//! | `queue`  | Tokio mpsc work queue (`RunQueueSender`, `RunQueueReceiver`) |
//! | `worker` | Background run processing task |
//! | `routes` | axum HTTP handlers for threads/messages/runs |
//! | `steps`  | axum HTTP handlers for run steps |
//! | `stream` | SSE streaming (`RunEvent`, broadcast channel) |

pub mod queue;
pub mod routes;
pub mod steps;
pub mod store;
pub mod stream;
pub mod types;
pub mod worker;

pub use queue::{new_run_queue, RunQueueSender};
pub use store::ThreadStore;
pub use types::{
    ContentBlock, CreateMessageRequest, CreateRunRequest, CreateThreadRequest,
    MessageCreationStepDetails, MessageRole, Run, RunError, RunStatus, RunStep, RunStepStatus,
    RunStepType, TextContent, Thread, ThreadMessage,
};
