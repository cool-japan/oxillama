//! `oxillama serve --model <model.gguf>` — OpenAI-compatible API server.
//!
//! Loads the model, hands it to the single inference worker that owns it, and
//! serves the `oxillama-server` router until the graceful-shutdown signal
//! fires. Unlike `run`, this subcommand does not consult the TOML config for
//! context/thread defaults — only the command line and the named model
//! profile.
//!
//! The whole module is gated on the `server` feature; the `Serve` variant and
//! its dispatch arm carry the same `#[cfg]`.

use std::path::PathBuf;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli_args::KvDtypeArg;
use crate::config;

/// Arguments for the `serve` subcommand.
#[derive(clap::Args)]
pub struct ServeArgs {
    /// Path to the GGUF model file.
    #[arg(short, long)]
    pub model: String,

    /// Host address to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port number.
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// Context size (max sequence length; 0 = auto: min(model's
    /// trained context, 4096)).
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: usize,

    /// llama.cpp alias for --ctx-size.
    #[arg(short = 'c', long = "n-ctx", conflicts_with = "ctx_size")]
    pub n_ctx: Option<usize>,

    /// Number of threads (0 = auto: one per logical CPU).
    #[arg(short = 't', long, default_value_t = 0)]
    pub threads: usize,

    /// Temperature for default sampler.
    #[arg(long, default_value_t = 0.7f32)]
    pub temp: f32,

    /// Top-P for default sampler.
    #[arg(long, default_value_t = 0.9f32)]
    pub top_p: f32,

    /// Top-K for default sampler.
    #[arg(long, default_value_t = 40usize)]
    pub top_k: usize,

    /// Seed for the default sampler (0 = random).
    #[arg(short = 's', long, default_value_t = 0u64)]
    pub seed: u64,

    /// Override the model identifier reported to clients (defaults to the GGUF file stem).
    #[arg(long, value_name = "ID")]
    pub model_id: Option<String>,

    /// Named sampler profile to apply (`~/.config/oxillama/models/<name>.toml`).
    /// Falls back to a profile matching the model's file stem when omitted.
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// Bearer API key required on `/v1/*` routes (repeatable). No keys = no auth.
    #[arg(long = "api-key", value_name = "KEY")]
    pub api_keys: Vec<String>,

    /// Bearer token required on `/admin/*` routes. Required unless the
    /// server only ever listens on loopback.
    #[arg(long, value_name = "TOKEN")]
    pub admin_token: Option<String>,

    /// Directories that admin-supplied model/LoRA paths may resolve into
    /// (repeatable). Empty = no restriction.
    #[arg(long = "allowed-model-dir", value_name = "DIR")]
    pub allowed_model_dirs: Vec<PathBuf>,

    /// Disable permissive CORS headers.
    #[arg(long)]
    pub disable_cors: bool,

    /// Maximum concurrent in-flight requests (0 = unbounded).
    #[arg(long, default_value_t = 64usize)]
    pub max_concurrent: usize,

    /// Per-request timeout in seconds (0 = unbounded).
    #[arg(long, default_value_t = 300u64)]
    pub request_timeout_secs: u64,

    /// Maximum request body size in bytes (0 = unbounded).
    #[arg(long, default_value_t = 10 * 1024 * 1024)]
    pub body_limit_bytes: usize,

    /// Directory for disk-spooled batch jobs (defaults to a temp dir).
    #[arg(long, value_name = "DIR")]
    pub batch_spool_dir: Option<PathBuf>,

    /// KV cache storage type: `f32` (lossless, default) or `f16` (half the
    /// KV memory). Attention still accumulates in f32 either way.
    #[arg(long = "kv-dtype", value_enum, default_value_t = KvDtypeArg::F32)]
    pub kv_dtype: KvDtypeArg,
}

/// Execute the `serve` subcommand; returns only once the server has shut down.
///
/// # Errors
///
/// Propagates model-loading failures, `AppState`/router construction failures,
/// a failure to bind `host:port`, and any error surfaced by `axum::serve`.
pub async fn run_serve(args: ServeArgs) -> Result<()> {
    let ServeArgs {
        model,
        host,
        port,
        ctx_size,
        n_ctx,
        threads,
        temp,
        top_p,
        top_k,
        seed,
        model_id: model_id_override,
        profile,
        api_keys,
        admin_token,
        allowed_model_dirs,
        disable_cors,
        max_concurrent,
        request_timeout_secs,
        body_limit_bytes,
        batch_spool_dir,
        kv_dtype,
    } = args;

    let effective_ctx = n_ctx.unwrap_or(ctx_size);
    let effective_seed = if seed == 0 { None } else { Some(seed) };

    // Extract model name from filename for the API model ID,
    // unless the caller supplied an explicit override via --model-id.
    let model_id = model_id_override.unwrap_or_else(|| {
        std::path::Path::new(&model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("oxillama-model")
            .to_string()
    });

    // Resolve a per-model sampler profile: --profile <name> wins,
    // otherwise fall back to a profile matching the model's file stem.
    let model_stem = std::path::Path::new(&model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let profile_name = profile.unwrap_or_else(|| model_stem.to_string());
    let model_profile = config::load_model_profile(&profile_name).unwrap_or_default();

    let final_temp = model_profile.as_ref().and_then(|p| p.temp).unwrap_or(temp);
    let final_top_p = model_profile
        .as_ref()
        .and_then(|p| p.top_p)
        .unwrap_or(top_p);
    let final_top_k = model_profile
        .as_ref()
        .and_then(|p| p.top_k.map(|k| k as usize))
        .unwrap_or(top_k);
    let final_seed = effective_seed.or_else(|| model_profile.as_ref().and_then(|p| p.seed));
    let final_ctx = model_profile
        .as_ref()
        .and_then(|p| p.ctx_size)
        .unwrap_or(effective_ctx);
    let final_threads = model_profile
        .as_ref()
        .and_then(|p| p.threads)
        .unwrap_or(threads);

    tracing::info!(
        model = %model,
        host = %host,
        port,
        ctx_size = final_ctx,
        threads = final_threads,
        kv_dtype = ?kv_dtype,
        "starting server"
    );

    // Build default sampler config from CLI flags / profile overrides.
    let default_sampler = oxillama_runtime::SamplerConfig {
        temperature: final_temp,
        top_p: final_top_p,
        top_k: final_top_k,
        seed: final_seed,
        ..Default::default()
    };

    // Load the inference engine
    let config = oxillama_runtime::EngineConfig {
        model_path: model,
        tokenizer_path: None,
        context_size: Some(final_ctx),
        num_threads: final_threads,
        sampler: default_sampler,
        kv_dtype: kv_dtype.into(),
        ..Default::default()
    };

    let pb_serve = ProgressBar::new_spinner();
    pb_serve.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb_serve.set_message(format!("Loading model: {model_id}"));
    pb_serve.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut engine = oxillama_runtime::InferenceEngine::new(config);
    engine.load_model()?;
    pb_serve.finish_and_clear();

    // Cache read-only fields before the engine moves into the worker.
    let cached_sampler = engine.config().sampler.clone();
    let hidden_size = engine.hidden_size().unwrap_or(0);
    let vocab_bytes = engine.vocab_bytes().map(std::sync::Arc::new);
    // Resolve the model's real chat template once, from the GGUF
    // metadata the engine already holds. Every chat-shaped route
    // renders through this instead of the fabricated, model-agnostic
    // `<|system|>…<|end|>` skeleton the server used to hardcode.
    let served_chat_template = engine.chat_template().unwrap_or_default();
    tracing::info!(
        chat_template = %served_chat_template,
        "resolved chat template for served model"
    );

    // Start the single inference worker that owns the engine.
    //
    // `prefix_registry` and `worker_alive` are shared `Arc`s between
    // the worker and `AppState` so the HTTP layer observes the
    // worker's real liveness (`/ready`) and both sides namespace the
    // prefix KV cache identically by LoRA selection.
    let (queue_tx, queue_rx) = tokio::sync::mpsc::channel::<oxillama_server::BatchRequest>(64);
    let prefix_registry = std::sync::Arc::new(oxillama_server::PrefixCacheRegistry::new(
        oxillama_runtime::PrefixCacheConfig::default(),
        oxillama_server::DEFAULT_MAX_NAMESPACES,
    ));
    let worker_alive = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let loras = std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::<
        String,
        std::sync::Arc<oxillama_runtime::LoadedLora>,
    >::new()));
    oxillama_server::spawn_inference_worker(
        engine,
        queue_rx,
        model_id.clone(),
        std::sync::Arc::clone(&prefix_registry),
        std::sync::Arc::clone(&loras),
        std::sync::Arc::clone(&worker_alive),
    );

    let state = oxillama_server::AppState::new(
        queue_tx,
        model_id,
        cached_sampler,
        vocab_bytes,
        hidden_size,
        served_chat_template,
        std::sync::Arc::clone(&prefix_registry),
        std::sync::Arc::clone(&worker_alive),
        batch_spool_dir.clone(),
    )?
    .with_allowed_model_dirs(allowed_model_dirs);

    // `admin_listen` mirrors the real listen address: the admin
    // routes are mounted on this same router rather than a separate
    // listener, so the loopback-only safety check
    // (`ensure_admin_security`) must see where we are actually
    // binding, not a fixed placeholder.
    let server_config = oxillama_server::ServerConfig {
        host: host.clone(),
        port,
        cors_enabled: !disable_cors,
        api_keys,
        max_concurrent,
        timeout_secs: request_timeout_secs,
        body_limit_bytes,
        admin_bearer_token: admin_token,
        admin_listen: format!("{host}:{port}"),
        batch_spool_dir: batch_spool_dir.map(|p| p.display().to_string()),
        ..Default::default()
    };

    let app = oxillama_server::build_app_with_config(std::sync::Arc::new(state), &server_config)?;
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on {addr}");
    // `into_make_service_with_connect_info` is required, not optional:
    // the admin auth guard's loopback detection reads
    // `ConnectInfo<SocketAddr>` and fails CLOSED (denies) when it is
    // absent, so plain `into_make_service()` would reject every
    // `/admin/*` request including legitimate loopback ones.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(oxillama_server::shutdown_signal())
    .await?;

    Ok(())
}
