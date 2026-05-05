//! Admin API — fleet management endpoints under `/admin/*`.
//!
//! Includes bearer-token or loopback-only authentication, background model
//! loading, unloading, server stats, and LoRA adapter registry management.

pub mod auth;
pub mod loras;
pub mod routes;
pub mod stats;

pub use auth::{admin_auth_middleware, ensure_admin_security, AdminAuth};
pub use loras::{admin_list_loras, admin_register_lora, admin_unregister_lora};
pub use routes::{
    admin_health, admin_list_models, admin_load_model, admin_stats, admin_unload_model,
};
pub use stats::AdminStats;
