//! Example: start the OpenAI-compatible HTTP server programmatically.
//!
//! Usage:
//! ```text
//! cargo run --example openai_server --features server -- --model /path/to/model.gguf --port 8080
//! ```
//!
//! Then query with:
//! ```text
//! curl -X POST http://localhost:8080/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"oxillama","messages":[{"role":"user","content":"Hello!"}]}'
//! ```
//!
//! If the `server` feature is not enabled the example prints a notice and exits
//! cleanly.

fn parse_args() -> Option<(String, u16)> {
    let mut args = std::env::args().skip(1);
    let mut model_path: Option<String> = None;
    let mut port: u16 = 8080;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_path = args.next();
            }
            "--port" => {
                if let Some(p) = args.next() {
                    port = p.parse().unwrap_or_else(|_| {
                        eprintln!("Invalid port '{p}', using 8080");
                        8080
                    });
                }
            }
            other => {
                eprintln!("Unknown argument: {other}");
            }
        }
    }

    model_path.map(|m| (m, port))
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("The `server` feature is not enabled.");
    eprintln!(
        "Re-run with: cargo run --example openai_server --features server -- --model <model.gguf>"
    );
}

#[cfg(feature = "server")]
fn main() -> anyhow::Result<()> {
    let (model_path, port) = match parse_args() {
        Some(v) => v,
        None => {
            eprintln!("Usage: openai_server --model <model.gguf> [--port <port>]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    run_server(model_path, port)
}

#[cfg(feature = "server")]
fn run_server(model_path: String, port: u16) -> anyhow::Result<()> {
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    use oxillama_runtime::{EngineConfig, InferenceEngine};
    use oxillama_server::{
        build_app_with_config, spawn_inference_worker, AppState, PrefixCacheRegistry, ServerConfig,
        DEFAULT_MAX_NAMESPACES,
    };

    // ── Build engine ──────────────────────────────────────────────────────────
    let engine_config = EngineConfig {
        model_path: model_path.clone(),
        ..Default::default()
    };
    let mut engine = InferenceEngine::new(engine_config);

    eprintln!("Loading model: {model_path}");
    engine.load_model()?;
    eprintln!("Model loaded.");

    // Cache read-only metadata before the engine moves into the worker.
    let cached_sampler = engine.config().sampler.clone();
    let hidden_size = engine.hidden_size().unwrap_or(0);
    let vocab_bytes = engine.vocab_bytes().map(std::sync::Arc::new);
    // The model's real chat template, resolved from the GGUF metadata the
    // engine already holds. Every chat-shaped route renders through it.
    let chat_template = engine.chat_template().unwrap_or_default();

    // ── Build shared state ────────────────────────────────────────────────────
    let server_cfg = ServerConfig {
        host: "127.0.0.1".to_string(),
        port,
        ..Default::default()
    };

    // ── Wire up the worker queue ──────────────────────────────────────────────
    // The worker owns the engine exclusively; route handlers communicate with
    // it through an mpsc channel. `prefix_registry` and `worker_alive` are
    // shared `Arc`s between the worker and `AppState` so both sides observe
    // the same prefix-cache/liveness state.
    let (tx, rx) = mpsc::channel(64);
    let prefix_registry = Arc::new(PrefixCacheRegistry::new(
        oxillama_runtime::PrefixCacheConfig::default(),
        DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = Arc::new(AtomicBool::new(false));
    let loras = std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::<
        String,
        std::sync::Arc<oxillama_runtime::LoadedLora>,
    >::new()));
    spawn_inference_worker(
        engine,
        rx,
        model_path.clone(),
        Arc::clone(&prefix_registry),
        loras,
        Arc::clone(&worker_alive),
    );

    let state = Arc::new(AppState::new(
        tx,
        model_path.clone(),
        cached_sampler,
        vocab_bytes,
        hidden_size,
        chat_template,
        prefix_registry,
        worker_alive,
        server_cfg.batch_spool_dir.clone().map(PathBuf::from),
    )?);

    // ── Build the axum router ─────────────────────────────────────────────────
    // `build_app_with_config` (not the test-only `build_app`) applies the full
    // production layer stack: admin auth, rate limiting, timeouts, CORS,
    // metrics, and tracing, all driven by `server_cfg`.
    let app = build_app_with_config(Arc::clone(&state), &server_cfg)?;

    let bind_addr = format!("{}:{}", server_cfg.host, server_cfg.port);
    eprintln!("Server listening on http://{bind_addr}");
    eprintln!("  POST /v1/chat/completions");
    eprintln!("  POST /v1/completions");
    eprintln!("  GET  /health");

    // ── Run the Tokio runtime ─────────────────────────────────────────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        // `with_connect_info` is required: the loopback-only admin fallback
        // (no bearer token configured) reads `ConnectInfo<SocketAddr>` and
        // fails closed — denying even legitimate loopback requests — if the
        // service is built with the plain `into_make_service()` instead.
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(oxillama_server::shutdown_signal())
        .await
        .map_err(anyhow::Error::from)
    })
}
