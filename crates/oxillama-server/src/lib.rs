//! # oxillama-server
//!
//! OpenAI-compatible HTTP API server for OxiLLaMa.
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/v1/chat/completions` | Chat completion |
//! | POST | `/v1/completions` | Text completion |
//! | POST | `/v1/embeddings` | Text embeddings |
//! | GET | `/v1/models` | List loaded models |
//! | GET | `/health` | Health check |
//! | POST | `/v1/batches` | Create batch job (disk-spooled) |
//! | GET | `/v1/batches/:id` | Retrieve batch job |
//! | GET | `/v1/batches/:id/output` | Stream batch output JSONL |
//! | POST | `/v1/batches/:id/cancel` | Cancel batch job |
//! | GET | `/v1/batches` | List batch jobs |
//! | POST | `/admin/models/load` | Background-load model (admin) |
//! | POST | `/admin/models/unload` | Unload model (admin) |
//! | GET | `/admin/models` | List model pool (admin) |
//! | GET | `/admin/stats` | Server stats (admin) |
//! | GET | `/admin/health` | Extended health (admin) |

pub mod admin;
pub mod app;
pub mod auth;
#[cfg(feature = "jwt")]
pub mod jwt_auth;
pub mod batch;
pub mod batch_spool;
pub mod body_limit;
pub mod config;
pub mod error;
pub mod metrics;
pub mod queue;
pub mod rate_limit;
pub mod router;
pub mod routes;
pub mod shutdown;
pub mod sse;
pub mod state;
pub mod tracing_layer;
pub mod worker;
pub mod ws;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use app::build_app;
pub use auth::ApiKeys;
pub use config::ServerConfig;
pub use error::{ServerError, ServerResult};
pub use metrics::Metrics;
pub use queue::{BatchRequest, VocabBytes};
pub use rate_limit::RateLimiter;
pub use router::{ModelLoader, ModelPool, ModelSpec};
pub use shutdown::{shutdown_signal, ShutdownSignal, ShutdownTrigger};
pub use state::AppState;
pub use worker::spawn_inference_worker;
