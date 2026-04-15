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

pub mod app;
pub mod auth;
pub mod body_limit;
pub mod config;
pub mod error;
pub mod metrics;
pub mod queue;
pub mod rate_limit;
pub mod routes;
pub mod shutdown;
pub mod sse;
pub mod state;
pub mod tracing_layer;
pub mod worker;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use app::build_app;
pub use auth::ApiKeys;
pub use config::ServerConfig;
pub use error::{ServerError, ServerResult};
pub use metrics::Metrics;
pub use queue::{BatchRequest, VocabBytes};
pub use rate_limit::RateLimiter;
pub use shutdown::{shutdown_signal, ShutdownSignal, ShutdownTrigger};
pub use state::AppState;
pub use worker::spawn_inference_worker;
