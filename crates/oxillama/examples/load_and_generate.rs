//! Example: Load a GGUF model and generate text with a streaming callback.
//!
//! # Usage
//!
//! ```text
//! cargo run --example load_and_generate -- --model model.gguf --prompt "Once upon a time"
//! cargo run --example load_and_generate -- --help
//! ```
//!
//! # What this example demonstrates
//!
//! - Building an [`EngineConfig`] and loading the model via [`InferenceEngine::load_model`].
//! - Overriding the default sampler temperature and top-k through a custom
//!   [`SamplerConfig`] passed to [`InferenceEngine::generate_with_config`].
//! - Streaming tokens to stdout as they are produced (no buffering before print).
//! - Reporting per-run latency and token-generation rate from the engine's
//!   [`MetricsSnapshot`].
//! - Proper `anyhow::Result` propagation — no `unwrap()` anywhere.

use std::io::Write as _;
use std::time::Instant;

use clap::Parser;
use oxillama_runtime::{EngineConfig, InferenceEngine, SamplerConfig};

/// Load a GGUF model and stream generated tokens to stdout.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the GGUF model file.
    #[arg(long, short = 'm')]
    model: String,

    /// Path to a HuggingFace tokenizer.json (optional; auto-detected if absent).
    #[arg(long, short = 't')]
    tokenizer: Option<String>,

    /// Text prompt to start generation from.
    #[arg(long, short = 'p', default_value = "Once upon a time")]
    prompt: String,

    /// Maximum number of tokens to generate.
    #[arg(long, short = 'n', default_value_t = 128)]
    max_tokens: usize,

    /// Sampling temperature (higher = more creative; 0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-K cutoff (0 = disabled).
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Number of CPU threads for the forward pass.
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Context window override in tokens (0 = use model default).
    #[arg(long, default_value_t = 0)]
    context: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── 1. Build the engine configuration ────────────────────────────────────
    let context_size = if args.context == 0 {
        None
    } else {
        Some(args.context)
    };

    let config = EngineConfig {
        model_path: args.model.clone(),
        tokenizer_path: args.tokenizer.clone(),
        context_size,
        num_threads: args.threads,
        sampler: SamplerConfig::default(),
        ..EngineConfig::default()
    };

    // ── 2. Load model ─────────────────────────────────────────────────────────
    eprintln!("Loading model: {}", args.model);
    let mut engine = InferenceEngine::new(config);
    engine.load_model()?;

    if let Some(mc) = engine.model_config() {
        eprintln!(
            "  arch={} layers={} hidden={} vocab={}",
            mc.architecture, mc.num_layers, mc.hidden_size, mc.vocab_size
        );
    }

    // ── 3. Build per-request sampler config ───────────────────────────────────
    let sampler = SamplerConfig {
        temperature: args.temperature as f32,
        top_k: args.top_k,
        ..SamplerConfig::default()
    };

    // ── 4. Stream generation to stdout ───────────────────────────────────────
    eprintln!("\nPrompt: {}\n---", args.prompt);
    let stdout = std::io::stdout();
    let mut locked = stdout.lock();

    let wall_start = Instant::now();
    let mut token_count = 0usize;

    engine.generate_with_config(&args.prompt, args.max_tokens, sampler, |tok| {
        // tok is `&str`; callback returns `()`
        let _ = locked.write_all(tok.as_bytes());
        let _ = locked.flush();
        token_count += 1;
    })?;

    let elapsed = wall_start.elapsed();
    drop(locked); // release lock before eprintln

    // ── 5. Report throughput ──────────────────────────────────────────────────
    let snap = engine.metrics_snapshot();
    let tps = if elapsed.as_secs_f64() > 0.0 {
        token_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("\n---");
    eprintln!("Generated tokens : {token_count}");
    eprintln!("Wall time        : {:.2}s", elapsed.as_secs_f64());
    eprintln!("Throughput       : {tps:.1} tok/s");
    eprintln!("Requests done    : {}", snap.requests_completed);

    Ok(())
}
