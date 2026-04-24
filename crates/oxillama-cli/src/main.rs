//! OxiLLaMa CLI — Pure Rust LLM inference engine.

mod config;
mod exit_codes;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Read;
use std::path::PathBuf;

// ── Version banner ──────────────────────────────────────────────────────────

const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_verbose_version() {
    let git_sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
    let target = option_env!("CARGO_CFG_TARGET_ARCH").unwrap_or(std::env::consts::ARCH);
    let build_date = option_env!("VERGEN_BUILD_DATE").unwrap_or("2026-04-16");
    println!("oxillama {PKG_VERSION}");
    println!("  arch:       {target}");
    println!("  build-date: {build_date}");
    println!("  git-sha:    {git_sha}");
    println!("  license:    Apache-2.0");
}

/// OxiLLaMa: Pure Rust LLM inference engine.
#[derive(Parser)]
#[command(name = "oxillama")]
#[command(
    version,
    about = "Pure Rust LLM inference engine — the sovereign alternative to llama.cpp"
)]
struct Cli {
    /// Path to a TOML config file (overrides OXILLAMA_CONFIG env var).
    #[arg(long, global = true, env = "OXILLAMA_CONFIG", value_name = "PATH")]
    config: Option<PathBuf>,

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

        /// Input prompt text (reads stdin if neither --prompt nor --file is given).
        #[arg(short, long, default_value = "")]
        prompt: String,

        /// Read prompt from this file instead of --prompt or stdin.
        #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
        file: Option<PathBuf>,

        /// Read prompt from stdin explicitly (pipe mode); conflicts with --prompt and --file.
        #[arg(long, conflicts_with_all = ["prompt", "file"])]
        stdin: bool,

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

        /// Print verbose version banner and exit.
        #[arg(long)]
        verbose_version: bool,
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

    /// Interactive multi-turn chat REPL.
    Chat {
        /// Path to the GGUF model file.
        #[arg(short, long)]
        model: String,

        /// Path to tokenizer.json (auto-detected if not provided).
        #[arg(long)]
        tokenizer: Option<String>,

        /// Context size (max sequence length).
        #[arg(long, default_value_t = 4096)]
        ctx_size: usize,

        /// Number of threads.
        #[arg(short = 't', long, default_value_t = 4)]
        threads: usize,

        /// Temperature for sampling.
        #[arg(long, default_value_t = 0.7f32)]
        temp: f32,

        /// Top-P for nucleus sampling.
        #[arg(long, default_value_t = 0.9f32)]
        top_p: f32,

        /// Top-K for sampling.
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Seed for reproducible sampling (0 = random).
        #[arg(short = 's', long, default_value_t = 0u64)]
        seed: u64,
    },

    /// Generate shell completion scripts.
    Completions {
        /// Target shell.
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },

    /// Generate man pages for oxillama and write them to a directory.
    #[command(name = "generate-manpage")]
    GenManpage {
        /// Directory to write man pages into (default: current directory).
        #[arg(long, value_name = "DIR", default_value = ".")]
        output_dir: PathBuf,
    },

    /// Print verbose version information.
    Version,
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        let code = exit_codes::classify(&err);
        eprintln!("error: {err:#}");
        std::process::exit(code);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Load config (env > --config > default path > defaults).
    let cfg = config::load_config(cli.config.clone()).unwrap_or_else(|e| {
        tracing::warn!("config load failed, using defaults: {e:#}");
        config::OxillamaConfig::default()
    });

    // Initialise tracing; config log_level wins over RUST_LOG.
    let log_filter = cfg.log_level.clone().unwrap_or_default();
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        if log_filter.is_empty() {
            tracing_subscriber::EnvFilter::new("info")
        } else {
            tracing_subscriber::EnvFilter::new(&log_filter)
        }
    });
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    match cli.command {
        Commands::Version => {
            print_verbose_version();
        }

        Commands::GenManpage { output_dir } => {
            std::fs::create_dir_all(&output_dir).map_err(|e| {
                anyhow::anyhow!("creating output dir '{}': {e}", output_dir.display())
            })?;
            let cmd = Cli::command();
            let man = clap_mangen::Man::new(cmd);
            let mut buf = Vec::new();
            man.render(&mut buf)?;
            let out_path = output_dir.join("oxillama.1");
            std::fs::write(&out_path, &buf).map_err(|e| {
                anyhow::anyhow!("writing man page to '{}': {e}", out_path.display())
            })?;
            println!("Man page written to {}", out_path.display());
        }

        Commands::Run {
            model,
            prompt,
            file,
            stdin,
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
            verbose_version,
        } => {
            if verbose_version {
                print_verbose_version();
                return Ok(());
            }

            let effective_max_tokens = n_predict.unwrap_or(max_tokens);
            let effective_temp = temperature.unwrap_or(temp);
            let effective_ctx = n_ctx.unwrap_or(cfg.default_ctx_size.unwrap_or(ctx_size));
            let effective_threads = cfg.default_threads.unwrap_or(threads);
            let effective_seed = if seed == 0 { None } else { Some(seed) };

            // Resolve prompt: --file > --prompt > --stdin / empty prompt fallback.
            let effective_prompt = if let Some(ref path) = file {
                std::fs::read_to_string(path)
                    .map_err(|e| anyhow::anyhow!("reading prompt file '{}': {e}", path.display()))?
            } else if stdin || prompt.is_empty() {
                let mut buf = String::new();
                std::io::stdin()
                    .read_to_string(&mut buf)
                    .map_err(|e| anyhow::anyhow!("reading stdin: {e}"))?;
                buf
            } else {
                prompt.clone()
            };

            // Load per-model profile if one exists.
            let model_stem = std::path::Path::new(&model)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            let profile = config::load_model_profile(model_stem).unwrap_or_default();

            let final_temp = profile
                .as_ref()
                .and_then(|p| p.temp)
                .unwrap_or(effective_temp);
            let final_top_p = profile.as_ref().and_then(|p| p.top_p).unwrap_or(top_p);
            let final_top_k = profile
                .as_ref()
                .and_then(|p| p.top_k.map(|k| k as usize))
                .unwrap_or(top_k);
            let final_seed = effective_seed.or_else(|| profile.as_ref().and_then(|p| p.seed));
            let final_ctx = profile
                .as_ref()
                .and_then(|p| p.ctx_size)
                .unwrap_or(effective_ctx);
            let final_threads = profile
                .as_ref()
                .and_then(|p| p.threads)
                .unwrap_or(effective_threads);

            tracing::info!(
                model = %model,
                ctx_size = final_ctx,
                threads = final_threads,
                temp = final_temp,
                top_p = final_top_p,
                top_k = final_top_k,
                "starting inference"
            );

            let sampler = oxillama_runtime::SamplerConfig {
                temperature: final_temp,
                top_p: final_top_p,
                top_k: final_top_k,
                min_p,
                repetition_penalty: repeat_penalty,
                seed: final_seed,
                ..Default::default()
            };

            let config = oxillama_runtime::EngineConfig {
                model_path: model.clone(),
                tokenizer_path: tokenizer,
                context_size: Some(final_ctx),
                num_threads: final_threads,
                sampler,
                ..Default::default()
            };

            eprintln!("{}", "OxiLLaMa inference started".green().bold());

            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{spinner:.green} {msg}")
                    .unwrap_or_else(|_| ProgressStyle::default_spinner()),
            );
            pb.set_message(format!("Loading model: {model}"));
            pb.enable_steady_tick(std::time::Duration::from_millis(100));

            let mut engine = oxillama_runtime::InferenceEngine::new(config);
            engine.load_model()?;
            pb.finish_and_clear();

            engine.generate(&effective_prompt, effective_max_tokens, |token| {
                print!("{token}");
            })?;
            println!();
            eprintln!("{}", "✓ Done".green());
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

            eprintln!("{}: {}", "Model".cyan().bold(), model);

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

        Commands::Chat {
            model,
            tokenizer,
            ctx_size,
            threads,
            temp,
            top_p,
            top_k,
            seed,
        } => {
            let effective_seed = if seed == 0 { None } else { Some(seed) };

            let sampler = oxillama_runtime::SamplerConfig {
                temperature: temp,
                top_p,
                top_k,
                seed: effective_seed,
                ..Default::default()
            };

            let config = oxillama_runtime::EngineConfig {
                model_path: model,
                tokenizer_path: tokenizer,
                context_size: Some(ctx_size),
                num_threads: threads,
                sampler,
                ..Default::default()
            };

            let mut engine = oxillama_runtime::InferenceEngine::new(config);
            {
                let pb_chat = ProgressBar::new_spinner();
                pb_chat.set_style(
                    ProgressStyle::with_template("{spinner:.green} {msg}")
                        .unwrap_or_else(|_| ProgressStyle::default_spinner()),
                );
                pb_chat.set_message("Loading model...");
                pb_chat.enable_steady_tick(std::time::Duration::from_millis(100));
                engine.load_model()?;
                pb_chat.finish_and_clear();
            }

            // Set up history file directory.
            let history_path = {
                let state_dir = dirs::state_dir()
                    .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("state")))
                    .ok_or_else(|| anyhow::anyhow!("cannot determine state directory"))?;
                let dir = state_dir.join("oxillama");
                std::fs::create_dir_all(&dir)?;
                dir.join("history")
            };

            let mut rl = rustyline::DefaultEditor::new()?;
            let _ = rl.load_history(&history_path);

            let mut system_prompt: Option<String> = None;

            eprintln!(
                "{}",
                "OxiLLaMa Chat  (type /quit to exit, /reset, /system <text>)"
                    .green()
                    .bold()
            );

            loop {
                // rustyline is !Send; use block_in_place so we can call it from async
                // without moving it into a spawn_blocking closure.
                let readline = tokio::task::block_in_place(|| rl.readline("You> "));

                let line = match readline {
                    Ok(l) => l,
                    Err(rustyline::error::ReadlineError::Eof) => break,
                    Err(rustyline::error::ReadlineError::Interrupted) => break,
                    Err(e) => {
                        tracing::warn!("readline error: {e}");
                        break;
                    }
                };

                let input = line.trim().to_string();
                if input.is_empty() {
                    continue;
                }

                rl.add_history_entry(&input)?;

                if input == "/quit" || input == "/exit" {
                    break;
                } else if input == "/reset" {
                    engine.reset();
                    println!("[KV cache cleared]");
                    continue;
                } else if let Some(rest) = input.strip_prefix("/system ") {
                    system_prompt = Some(rest.to_string());
                    println!("[System prompt set]");
                    continue;
                }

                let full_prompt = if let Some(ref sys) = system_prompt {
                    format!("{sys}\n\nUser: {input}\nAssistant:")
                } else {
                    format!("User: {input}\nAssistant:")
                };

                print!("Assistant: ");
                use std::io::Write;
                std::io::stdout().flush()?;

                engine.generate(&full_prompt, 512, |token| {
                    print!("{token}");
                    let _ = std::io::stdout().flush();
                })?;
                println!();
            }

            let _ = rl.save_history(&history_path);
        }

        Commands::Completions { shell } => {
            clap_complete::generate(
                shell,
                &mut Cli::command(),
                "oxillama",
                &mut std::io::stdout(),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn manpage_args_parse_output_dir() {
        let cli = Cli::try_parse_from(["oxillama", "generate-manpage", "--output-dir", "/tmp"])
            .expect("parse generate-manpage");
        match cli.command {
            Commands::GenManpage { output_dir } => {
                assert_eq!(output_dir, std::path::PathBuf::from("/tmp"));
            }
            _ => panic!("expected GenManpage"),
        }
    }

    #[test]
    fn manpage_generate_to_temp() {
        let dir = std::env::temp_dir().join("oxillama_manpage_test");
        std::fs::create_dir_all(&dir).expect("create dir");
        assert!(dir.exists());
        // Validate that output_dir defaults to "."
        let cli = Cli::try_parse_from(["oxillama", "generate-manpage"]).expect("parse defaults");
        match cli.command {
            Commands::GenManpage { output_dir } => {
                assert_eq!(output_dir, std::path::PathBuf::from("."));
            }
            _ => panic!("expected GenManpage"),
        }
    }

    #[test]
    fn run_stdin_flag_parses() {
        let cli = Cli::try_parse_from(["oxillama", "run", "--model", "model.gguf", "--stdin"])
            .expect("parse run --stdin");
        match cli.command {
            Commands::Run { stdin, .. } => {
                assert!(stdin);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_stdin_conflicts_with_prompt() {
        let result = Cli::try_parse_from([
            "oxillama",
            "run",
            "--model",
            "model.gguf",
            "--prompt",
            "hello",
            "--stdin",
        ]);
        assert!(result.is_err(), "--stdin should conflict with --prompt");
    }

    #[test]
    fn run_file_and_stdin_conflict() {
        let result = Cli::try_parse_from([
            "oxillama",
            "run",
            "--model",
            "model.gguf",
            "--file",
            "/tmp/prompt.txt",
            "--stdin",
        ]);
        assert!(result.is_err(), "--stdin should conflict with --file");
    }
}
