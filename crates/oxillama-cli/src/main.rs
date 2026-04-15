//! OxiLLaMa CLI — Pure Rust LLM inference engine.

use anyhow::Result;
use clap::{Parser, Subcommand};

/// OxiLLaMa: Pure Rust LLM inference engine.
#[derive(Parser)]
#[command(name = "oxillama")]
#[command(
    version,
    about = "Pure Rust LLM inference engine — the sovereign alternative to llama.cpp"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a GGUF model.
    Run {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// Input prompt text.
        #[arg(short, long, default_value = "")]
        prompt: String,

        /// Path to tokenizer.json (auto-detected if not provided).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Maximum number of tokens to generate.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// llama.cpp alias: max tokens to predict (same as --max-tokens).
        #[arg(short = 'n', long = "n-predict", conflicts_with = "max_tokens")]
        n_predict: Option<usize>,

        /// Context size (max sequence length).
        #[arg(long, default_value_t = 4096)]
        ctx_size: usize,

        /// llama.cpp alias for --ctx-size.
        #[arg(short = 'c', long = "n-ctx", conflicts_with = "ctx_size")]
        n_ctx: Option<usize>,

        /// Number of threads.
        #[arg(short = 't', long, default_value_t = 4)]
        threads: usize,

        /// Temperature for sampling.
        #[arg(long, default_value_t = 0.7)]
        temp: f32,

        /// llama.cpp alias for --temp.
        #[arg(long = "temperature", conflicts_with = "temp")]
        temperature: Option<f32>,

        /// Top-P for nucleus sampling.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        /// Top-K for sampling.
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Seed for reproducible sampling (0 = random).
        #[arg(short = 's', long, default_value_t = 0u64)]
        seed: u64,

        /// Repetition penalty (1.0 = disabled).
        #[arg(long, default_value_t = 1.1f32)]
        repeat_penalty: f32,

        /// Min-P sampling threshold (0.0 = disabled).
        #[arg(long, default_value_t = 0.05f32)]
        min_p: f32,
    },

    /// Start the OpenAI-compatible API server.
    #[cfg(feature = "server")]
    Serve {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// Host address to bind to.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port number.
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Context size.
        #[arg(long, default_value_t = 4096)]
        ctx_size: usize,

        /// llama.cpp alias for --ctx-size.
        #[arg(short = 'c', long = "n-ctx", conflicts_with = "ctx_size")]
        n_ctx: Option<usize>,

        /// Number of threads.
        #[arg(short = 't', long, default_value_t = 4)]
        threads: usize,

        /// Temperature for default sampler.
        #[arg(long, default_value_t = 0.7f32)]
        temp: f32,

        /// Top-P for default sampler.
        #[arg(long, default_value_t = 0.9f32)]
        top_p: f32,

        /// Top-K for default sampler.
        #[arg(long, default_value_t = 40usize)]
        top_k: usize,
    },

    /// Print model information from a GGUF file.
    Info {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// Show all tensor names and shapes.
        #[arg(long)]
        tensors: bool,

        /// Show all metadata key-value pairs.
        #[arg(long)]
        metadata: bool,
    },

    /// Run inference benchmarks.
    #[cfg(feature = "bench")]
    Bench {
        /// Path to the GGUF model file.
        #[arg(short = 'm', long)]
        model: String,

        /// Number of warmup runs.
        #[arg(long, default_value_t = 3usize)]
        warmup: usize,

        /// Number of benchmark iterations.
        #[arg(long, default_value_t = 10usize)]
        iterations: usize,

        /// Number of tokens per run.
        #[arg(long = "n-predict", default_value_t = 128usize)]
        n_predict: usize,

        /// Number of threads.
        #[arg(short = 't', long, default_value_t = 4usize)]
        threads: usize,

        /// Context size.
        #[arg(long, default_value_t = 2048usize)]
        ctx_size: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            tokenizer,
            max_tokens,
            n_predict,
            ctx_size,
            n_ctx,
            threads,
            temp,
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            min_p,
        } => {
            let effective_max_tokens = n_predict.unwrap_or(max_tokens);
            let effective_temp = temperature.unwrap_or(temp);
            let effective_ctx = n_ctx.unwrap_or(ctx_size);
            let effective_seed = if seed == 0 { None } else { Some(seed) };

            tracing::info!(
                model = %model,
                ctx_size = effective_ctx,
                threads,
                temp = effective_temp,
                top_p,
                top_k,
                "starting inference"
            );

            let sampler = oxillama_runtime::SamplerConfig {
                temperature: effective_temp,
                top_p,
                top_k,
                min_p,
                repetition_penalty: repeat_penalty,
                seed: effective_seed,
                ..Default::default()
            };

            let config = oxillama_runtime::EngineConfig {
                model_path: model,
                tokenizer_path: tokenizer,
                context_size: Some(effective_ctx),
                num_threads: threads,
                sampler,
                ..Default::default()
            };

            let mut engine = oxillama_runtime::InferenceEngine::new(config);
            engine.load_model()?;
            engine.generate(&prompt, effective_max_tokens, |token| {
                print!("{token}");
            })?;
            println!();
        }

        #[cfg(feature = "server")]
        Commands::Serve {
            model,
            host,
            port,
            ctx_size,
            n_ctx,
            threads,
            temp,
            top_p,
            top_k,
        } => {
            let effective_ctx = n_ctx.unwrap_or(ctx_size);

            tracing::info!(
                model = %model,
                host = %host,
                port,
                ctx_size = effective_ctx,
                threads,
                "starting server"
            );

            // Extract model name from filename for the API model ID
            let model_id = std::path::Path::new(&model)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("oxillama-model")
                .to_string();

            // Build default sampler config from CLI flags
            let default_sampler = oxillama_runtime::SamplerConfig {
                temperature: temp,
                top_p,
                top_k,
                ..Default::default()
            };

            // Load the inference engine
            let config = oxillama_runtime::EngineConfig {
                model_path: model,
                tokenizer_path: None,
                context_size: Some(effective_ctx),
                num_threads: threads,
                sampler: default_sampler,
                ..Default::default()
            };

            let mut engine = oxillama_runtime::InferenceEngine::new(config);
            engine.load_model()?;

            // Cache read-only fields before the engine moves into the worker.
            let cached_sampler = engine.config().sampler.clone();
            let hidden_size = engine.hidden_size().unwrap_or(0);
            let vocab_bytes = engine.vocab_bytes().map(std::sync::Arc::new);

            // Start the single inference worker that owns the engine.
            let (queue_tx, queue_rx) =
                tokio::sync::mpsc::channel::<oxillama_server::BatchRequest>(64);
            oxillama_server::spawn_inference_worker(engine, queue_rx);

            let state = std::sync::Arc::new(oxillama_server::AppState::new(
                queue_tx,
                model_id,
                cached_sampler,
                vocab_bytes,
                hidden_size,
            ));
            let app = oxillama_server::build_app(state);
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            tracing::info!("listening on {addr}");
            axum::serve(listener, app).await?;
        }

        Commands::Info {
            model,
            tensors,
            metadata,
        } => {
            let path = std::path::Path::new(&model);
            if !path.exists() {
                anyhow::bail!("model file not found: {model}");
            }

            let gguf = oxillama_gguf::GgufModel::load(&model)?;

            // Print summary
            let mut summary = String::new();
            gguf.print_summary(&mut summary)?;
            print!("{summary}");

            // Print metadata if requested
            if metadata {
                println!("\nMetadata Key-Value Pairs");
                println!("========================");
                let mut keys: Vec<_> = gguf.file.metadata.keys().collect();
                keys.sort();
                for key in keys {
                    if let Some(value) = gguf.file.metadata.get(key) {
                        println!("  {key} = {value}");
                    }
                }
            }

            // Print tensor info if requested
            if tensors {
                println!("\nTensor Information");
                println!("==================");
                let mut tensor_list: Vec<_> = gguf.file.tensors.iter().collect();
                tensor_list.sort_by_key(|(name, _)| (*name).clone());
                for (name, info) in &tensor_list {
                    let dims: Vec<String> = info.dimensions.iter().map(|d| d.to_string()).collect();
                    println!(
                        "  {name}: [{dims}] {type_name} ({size:.2} MB)",
                        dims = dims.join(", "),
                        type_name = info.tensor_type.name(),
                        size = info.data_size() as f64 / 1_048_576.0,
                    );
                }
                println!("  Total: {} tensors", tensor_list.len());
            }
        }

        #[cfg(feature = "bench")]
        Commands::Bench {
            model,
            warmup,
            iterations,
            n_predict,
            threads,
            ctx_size,
        } => {
            tracing::info!(
                model = %model,
                warmup,
                iterations,
                n_predict,
                threads,
                ctx_size,
                "starting benchmark"
            );

            let config = oxillama_runtime::EngineConfig {
                model_path: model,
                tokenizer_path: None,
                context_size: Some(ctx_size),
                num_threads: threads,
                sampler: oxillama_runtime::SamplerConfig::default(),
                prefill_chunk_size: 512,
            };

            let mut engine = oxillama_runtime::InferenceEngine::new(config);
            engine.load_model()?;

            let prompt = "The quick brown fox";

            // Warmup runs — discard results
            for _ in 0..warmup {
                engine.generate(prompt, n_predict, |_| {})?;
            }

            // Benchmark runs
            let mut total_tokens = 0usize;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let result = engine.generate(prompt, n_predict, |_| {})?;
                total_tokens += result.split_whitespace().count();
            }
            let elapsed = start.elapsed().as_secs_f64();
            let tps = total_tokens as f64 / elapsed;

            println!("Benchmark results:");
            println!("  iterations:  {iterations}");
            println!("  total_time:  {elapsed:.3}s");
            println!("  ~tokens/s:   {tps:.1}");
        }
    }

    Ok(())
}
