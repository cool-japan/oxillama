//! Process exit codes for OxiLLaMa CLI.

#![allow(dead_code)]

/// Successful execution.
pub const SUCCESS: i32 = 0;

/// Model file not found or path is invalid.
pub const ERR_MODEL_NOT_FOUND: i32 = 2;

/// Invalid or malformed configuration / arguments.
pub const ERR_INVALID_CONFIG: i32 = 3;

/// Inference engine failure (load, generate, etc.).
pub const ERR_INFERENCE_FAILED: i32 = 4;

/// HTTP server startup or runtime failure.
pub const ERR_SERVER_FAILED: i32 = 5;

/// Generic I/O error (file read, stdin, etc.).
pub const ERR_IO: i32 = 6;

/// Map an `anyhow::Error` to the most appropriate exit code by inspecting its
/// message / chain.
pub fn classify(err: &anyhow::Error) -> i32 {
    let msg = format!("{err:#}");
    if msg.contains("model file not found") || msg.contains("No such file") {
        ERR_MODEL_NOT_FOUND
    } else if msg.contains("invalid config")
        || msg.contains("parsing TOML")
        || msg.contains("loading config")
        || msg.contains("OXILLAMA_CONFIG")
    {
        ERR_INVALID_CONFIG
    } else if msg.contains("inference")
        || msg.contains("generate")
        || msg.contains("load_model")
        || msg.contains("engine")
    {
        ERR_INFERENCE_FAILED
    } else if msg.contains("server") || msg.contains("listen") || msg.contains("bind") {
        ERR_SERVER_FAILED
    } else if msg.contains("IO") || msg.contains("io") || msg.contains("read") {
        ERR_IO
    } else {
        // Default: treat as config / args error.
        ERR_INVALID_CONFIG
    }
}
