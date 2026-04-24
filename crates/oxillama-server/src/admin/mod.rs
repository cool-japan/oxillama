//! Admin API — fleet management endpoints under `/admin/*`.
//!
//! Includes bearer-token or loopback-only authentication, background model
//! loading, unloading, and server stats.

pub mod auth;
pub mod routes;
pub mod stats;

pub use auth::{admin_auth_middleware, ensure_admin_security, AdminAuth};
pub use routes::{
    admin_health, admin_list_models, admin_load_model, admin_stats, admin_unload_model,
};
pub use stats::AdminStats;
