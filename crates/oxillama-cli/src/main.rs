//! OxiLLaMa CLI — Pure Rust LLM inference engine.

#[cfg(feature = "bench")]
mod bench;
mod chat_template;
mod cli_args;
mod config;
mod convert;
mod dump_logits;
mod exit_codes;
mod hub;
mod manpage;
mod quantize;
mod run_cmd;
#[cfg(feature = "server")]
mod serve_cmd;
mod session;
mod tokenize;
#[cfg(feature = "tui")]
mod tui;
mod verify;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
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
    Run(run_cmd::RunArgs),

    /// Start the OpenAI-compatible API server.
    #[cfg(feature = "server")]
    Serve(serve_cmd::ServeArgs),

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

        /// Number of threads (0 = auto: one per logical CPU).
        #[arg(short = 't', long, default_value_t = 0usize)]
        threads: usize,

        /// Context size (max sequence length; 0 = auto: min(model's
        /// trained context, 4096)).
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

        /// Context size (max sequence length; 0 = auto: min(model's
        /// trained context, 4096)).
        #[arg(long, default_value_t = 4096)]
        ctx_size: usize,

        /// Number of threads (0 = auto: one per logical CPU).
        #[arg(short = 't', long, default_value_t = 0)]
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

        /// Launch the full-screen ratatui TUI interface instead of the REPL.
        // Only advertised in `--help` on builds that can actually honour it
        // (the `tui` feature); a `cargo install oxillama-cli` build that
        // disabled default features would otherwise ship a documented flag
        // the parser accepts and then unconditionally refuses.
        #[cfg(feature = "tui")]
        #[arg(long)]
        tui: bool,

        /// Override the model identifier reported to clients (defaults to the GGUF file stem).
        #[arg(long, value_name = "ID")]
        model_id: Option<String>,

        /// Named sampler profile to apply (`~/.config/oxillama/models/<name>.toml`).
        /// Falls back to a profile matching the model's file stem when omitted.
        #[arg(long, value_name = "NAME")]
        profile: Option<String>,

        /// System prompt for this session, sent once as the conversation's
        /// leading turn (overrides the profile's `system_prompt`, if any).
        #[arg(long, value_name = "TEXT")]
        system: Option<String>,
    },

    /// HuggingFace Hub operations (pull, list, rm).
    Hub {
        #[command(subcommand)]
        command: HubCommand,
    },

    /// Re-quantize a GGUF model to a different quantization format.
    Quantize(quantize::QuantizeArgs),

    /// Convert a safetensors file to GGUF format.
    Convert(convert::ConvertArgs),

    /// Verify GGUF file integrity (magic, version, parse, optional SHA-256).
    Verify(verify::VerifyArgs),

    /// Tokenize a text string using the model's embedded tokenizer.
    Tokenize(tokenize::TokenizeArgs),

    /// Decode token IDs back to text using the model's embedded tokenizer.
    Detokenize(tokenize::DetokenizeArgs),

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

/// Subcommands for `oxillama hub`.
#[derive(Debug, clap::Subcommand)]
enum HubCommand {
    /// Download a model from HuggingFace Hub.
    Pull {
        /// Repository ID (e.g. "cool-japan/bonsai-8b").
        repo: String,
        /// Specific GGUF file to download (auto-selected if omitted).
        #[arg(long)]
        file: Option<String>,
        /// Git revision / branch / commit SHA.
        #[arg(long, default_value = "main")]
        revision: String,
        /// Force re-download even if already cached.
        #[arg(long)]
        force: bool,
        /// Override cache directory.
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Verify SHA-256 after download (hex string).
        #[arg(long)]
        verify_sha256: Option<String>,
    },
    /// List cached models.
    List {
        /// Override cache directory.
        #[arg(long)]
        cache: Option<PathBuf>,
    },
    /// Remove a cached model.
    Rm {
        /// Repository ID (e.g. "cool-japan/bonsai-8b").
        repo: String,
        /// Override cache directory.
        #[arg(long)]
        cache: Option<PathBuf>,
    },
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
            let written = manpage::generate_all(Cli::command(), &output_dir).map_err(|e| {
                anyhow::anyhow!("generating man pages into '{}': {e}", output_dir.display())
            })?;
            for path in &written {
                println!("Man page written to {}", path.display());
            }
            println!(
                "{} man page(s) written to {}",
                written.len(),
                output_dir.display()
            );
        }

        Commands::Run(args) => {
            run_cmd::run_run(args, &cfg).await?;
        }

        #[cfg(feature = "server")]
        Commands::Serve(args) => {
            serve_cmd::run_serve(args).await?;
        }

        Commands::Info {
            model,
            tensors,
            metadata,
        } => {
            let path = std::path::Path::new(&model);
            if !path.exists() {
                return Err(exit_codes::CliError::ModelNotFound(model).into());
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
                offload_policy: oxillama_runtime::OffloadPolicy::None,
                ..Default::default()
            };

            let mut engine = oxillama_runtime::InferenceEngine::new(config);
            engine.load_model()?;

            let prompt = "The quick brown fox";
            let result = bench::run_benchmark(&mut engine, prompt, warmup, iterations, n_predict)?;

            println!("Benchmark results:");
            println!("  iterations:        {}", result.iterations);
            println!("  total_time:        {:.3}s", result.elapsed_secs);
            println!("  completion_tokens: {}", result.total_completion_tokens);
            println!("  tokens/s:          {:.1}", result.tokens_per_sec());
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
            #[cfg(feature = "tui")]
            tui,
            model_id: model_id_override,
            profile,
            system,
        } => {
            let effective_seed = if seed == 0 { None } else { Some(seed) };

            // Resolve a per-model sampler profile: --profile <name> wins,
            // otherwise fall back to a profile matching the model's file stem.
            let model_stem = std::path::Path::new(&model)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            let profile_name = profile.unwrap_or_else(|| model_stem.clone());
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
                .unwrap_or(ctx_size);
            let final_threads = model_profile
                .as_ref()
                .and_then(|p| p.threads)
                .unwrap_or(threads);

            // System prompt priority: --system flag > profile's system_prompt > none.
            let initial_system_prompt =
                system.or_else(|| model_profile.as_ref().and_then(|p| p.system_prompt.clone()));

            let sampler = oxillama_runtime::SamplerConfig {
                temperature: final_temp,
                top_p: final_top_p,
                top_k: final_top_k,
                seed: final_seed,
                ..Default::default()
            };

            // Derive model_id before moving `model` into the engine config,
            // unless the caller supplied an explicit override via --model-id.
            let chat_model_id_pre = model_id_override.unwrap_or_else(|| model_stem.clone());

            let model_path_buf = std::path::PathBuf::from(&model);

            let config = oxillama_runtime::EngineConfig {
                model_path: model,
                tokenizer_path: tokenizer,
                context_size: Some(final_ctx),
                num_threads: final_threads,
                sampler: sampler.clone(),
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

            // ── TUI branch ────────────────────────────────────────────────────
            #[cfg(feature = "tui")]
            if tui {
                use std::sync::{Arc, Mutex};
                let model_id = chat_model_id_pre;
                let engine_arc = Arc::new(Mutex::new(engine));
                return crate::tui::run_tui(
                    model_path_buf,
                    model_id,
                    engine_arc,
                    sampler,
                    512,
                    initial_system_prompt,
                );
            }

            // Resolve the chat template once: the model's own
            // `tokenizer.chat_template` (or a vocabulary fingerprint) is used
            // to render every turn, matching the TUI's strategy (see
            // `crate::chat_template`).
            let chat_template = chat_template::ChatTemplate::resolve_from_path(&model_path_buf);

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

            // Use the model_id derived before the move.
            let chat_model_id = chat_model_id_pre;

            // Active session snapshot (tracks conversation turns). The chat
            // frontend never resumes the KV cache across turns — instead the
            // engine is reset and the *entire* transcript is re-rendered
            // through the chat template every turn, the same strategy the
            // TUI already used (C2 fix: before this, the REPL instead relied
            // on the KV cache silently accumulating turn over turn and never
            // reset it, so `/load` could restore a transcript into
            // `chat_session` while the model's actual context stayed however
            // it was before the load — typically empty. The two chat
            // frontends now agree.  This trades the KV-cache-reuse
            // optimisation for correctness; recovering the throughput via
            // `InferenceEngine::prime_with_prefix`'s prefix-KV-cache fast
            // path is a natural follow-up that does not reintroduce this
            // bug.)
            let mut chat_session = session::SessionSnapshot::new(chat_model_id.clone());
            chat_session.sampler = session::SamplerConfig {
                temperature: final_temp,
                top_p: final_top_p,
                top_k: final_top_k as u32,
                repeat_penalty: 1.1,
                seed: final_seed,
            };
            let mut current_system_prompt = initial_system_prompt;
            chat_template::seed_system_message(&mut chat_session, &current_system_prompt);

            eprintln!(
                "{}",
                "OxiLLaMa Chat  (type /quit to exit, /reset, /system <text>, /save <path>, /load <path>)"
                    .green()
                    .bold()
            );

            let gen_config = oxillama_runtime::GenerationConfig {
                max_tokens: 512,
                sampler,
                stop: Vec::new(),
                render_special: false,
                // Only Llama3/Mistral templates emit a literal BOS marker
                // (`<|begin_of_text|>` / `<s>`) themselves; for those,
                // add_special = true would duplicate it. ChatML/Alpaca emit
                // no such marker, so add_special must stay true there or the
                // tokenizer's own `add_bos_token` policy is silently skipped
                // (see `ChatTemplate::emits_literal_bos`). Control-token text
                // the template emits (`<|im_start|>`, `<|eot_id|>`, …) must
                // still be recognised as single tokens either way, hence
                // `parse_special: true`.
                add_special: !chat_template.emits_literal_bos(),
                parse_special: true,
                // The REPL cancels by breaking out of the readline loop, not
                // mid-decode: Ctrl-C ends the process.
                cancel_flag: None,
            };

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
                    chat_session.messages.clear();
                    // A system prompt is a session setting, not a
                    // conversation turn — it survives /reset, matching the
                    // pre-fix REPL's separate `system_prompt` variable that
                    // was never cleared by /reset either.
                    chat_template::seed_system_message(&mut chat_session, &current_system_prompt);
                    println!("[KV cache cleared, conversation reset]");
                    continue;
                } else if let Some(rest) = input.strip_prefix("/system ") {
                    current_system_prompt = Some(rest.to_string());
                    chat_template::seed_system_message(&mut chat_session, &current_system_prompt);
                    println!("[System prompt set]");
                    continue;
                } else if let Some(rest) = input.strip_prefix("/save ") {
                    let save_path = std::path::Path::new(rest.trim());
                    match session::save(&chat_session, save_path) {
                        Ok(()) => println!("[Session saved to {}]", save_path.display()),
                        Err(e) => eprintln!("[save error: {e}]"),
                    }
                    continue;
                } else if let Some(rest) = input.strip_prefix("/load ") {
                    let load_path = std::path::Path::new(rest.trim());
                    match session::load_for_model(load_path, &chat_model_id) {
                        Ok(loaded) => {
                            // The engine is reset unconditionally at the top
                            // of the next turn (below), so there is no stale
                            // KV state left over from before the load — the
                            // restored transcript and the model's actual
                            // context always agree.
                            chat_session = loaded;
                            println!(
                                "[Session loaded from {} ({} messages)]",
                                load_path.display(),
                                chat_session.messages.len()
                            );
                        }
                        Err(e) => eprintln!("[load error: {e}]"),
                    }
                    continue;
                }

                // Record user turn.
                chat_session.messages.push(session::ChatMessage {
                    role: "user".into(),
                    content: input.clone(),
                });

                print!("Assistant: ");
                use std::io::Write;
                std::io::stdout().flush()?;

                // Reset the KV cache and re-render the whole transcript
                // through the chat template — see the comment above
                // `chat_session`'s declaration for why.
                engine.reset();
                let full_prompt = chat_template::render_session(&chat_session, chat_template);

                let mut assistant_reply = String::new();
                engine.generate_detailed(&full_prompt, &gen_config, |token| {
                    assistant_reply.push_str(token);
                    print!("{token}");
                    let _ = std::io::stdout().flush();
                })?;
                println!();

                // Record assistant turn.
                chat_session.messages.push(session::ChatMessage {
                    role: "assistant".into(),
                    content: assistant_reply,
                });
            }

            let _ = rl.save_history(&history_path);
        }

        Commands::Hub { command } => match command {
            HubCommand::List { cache } => {
                let cache_dir = cache.unwrap_or_else(hub::default_cache_dir);
                hub::print_list(&cache_dir);
            }
            HubCommand::Rm { repo, cache } => {
                let cache_dir = cache.unwrap_or_else(hub::default_cache_dir);
                hub::remove_cached(&cache_dir, &repo)
                    .map_err(|e| anyhow::anyhow!("hub rm failed: {e}"))?;
            }
            HubCommand::Pull {
                repo,
                file,
                revision,
                force,
                cache,
                verify_sha256,
            } => {
                let opts = hub::PullOptions {
                    repo,
                    file,
                    revision,
                    force,
                    cache,
                    verify_sha256,
                };
                hub::pull(opts).map_err(|e| anyhow::anyhow!("hub pull failed: {e}"))?;
            }
        },

        Commands::Quantize(args) => {
            quantize::run_quantize(&args)?;
        }

        Commands::Convert(args) => {
            convert::run_convert(&args)?;
        }

        Commands::Verify(args) => {
            verify::run_verify(&args)?;
        }

        Commands::Tokenize(args) => {
            tokenize::run_tokenize(&args)?;
        }

        Commands::Detokenize(args) => {
            tokenize::run_detokenize(&args)?;
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
    use crate::cli_args::AddBosArg;
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
            Commands::Run(args) => {
                assert!(args.stdin);
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

    #[test]
    fn run_dump_logits_flags_parse_together() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "run",
            "--model",
            "model.gguf",
            "--dump-logits",
            "/dev/null/dump",
            "--prompt-tokens",
            "791,6864,315",
            "--force-tokens",
            "12366,627",
        ])
        .expect("parse run --dump-logits");
        match cli.command {
            Commands::Run(args) => {
                assert_eq!(args.dump_logits, Some(PathBuf::from("/dev/null/dump")));
                assert_eq!(args.prompt_tokens.as_deref(), Some("791,6864,315"));
                assert_eq!(args.force_tokens.as_deref(), Some("12366,627"));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_token_level_debug_flags_require_dump_logits() {
        // Both only make sense while dumping; accepting them alone would
        // silently ignore them.
        assert!(
            Cli::try_parse_from([
                "oxillama",
                "run",
                "--model",
                "model.gguf",
                "--prompt-tokens",
                "1,2,3",
            ])
            .is_err(),
            "--prompt-tokens should require --dump-logits"
        );
        assert!(
            Cli::try_parse_from([
                "oxillama",
                "run",
                "--model",
                "model.gguf",
                "--force-tokens",
                "1,2,3",
            ])
            .is_err(),
            "--force-tokens should require --dump-logits"
        );
    }

    #[test]
    fn run_prompt_tokens_conflicts_with_text_prompt_sources() {
        for extra in [
            vec!["--prompt", "hello"],
            vec!["--file", "prompt.txt"],
            vec!["--stdin"],
        ] {
            let mut args = vec![
                "oxillama",
                "run",
                "--model",
                "model.gguf",
                "--dump-logits",
                "dump",
                "--prompt-tokens",
                "1,2",
            ];
            args.extend(extra.iter().copied());
            assert!(
                Cli::try_parse_from(&args).is_err(),
                "--prompt-tokens should conflict with {extra:?}"
            );
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn serve_model_id_override_parses() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "serve",
            "--model",
            "m.gguf",
            "--model-id",
            "custom-id",
        ])
        .expect("parse serve --model-id");
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.model_id, Some("custom-id".to_string()));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn serve_model_id_defaults_none() {
        let cli = Cli::try_parse_from(["oxillama", "serve", "--model", "m.gguf"])
            .expect("parse serve without --model-id");
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.model_id, None);
            }
            _ => panic!("expected Serve"),
        }
    }

    #[test]
    fn chat_model_id_override_parses() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "chat",
            "--model",
            "m.gguf",
            "--model-id",
            "custom-id",
        ])
        .expect("parse chat --model-id");
        match cli.command {
            Commands::Chat { model_id, .. } => {
                assert_eq!(model_id, Some("custom-id".to_string()));
            }
            _ => panic!("expected Chat"),
        }
    }

    #[test]
    fn chat_model_id_defaults_none() {
        let cli = Cli::try_parse_from(["oxillama", "chat", "--model", "m.gguf"])
            .expect("parse chat without --model-id");
        match cli.command {
            Commands::Chat { model_id, .. } => {
                assert_eq!(model_id, None);
            }
            _ => panic!("expected Chat"),
        }
    }

    // ── C7: --profile / --system argument parsing ──────────────────────────

    #[test]
    fn run_profile_flag_parses() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "run",
            "--model",
            "m.gguf",
            "--profile",
            "qwen3-7b",
        ])
        .expect("parse run --profile");
        match cli.command {
            Commands::Run(args) => {
                assert_eq!(args.profile, Some("qwen3-7b".to_string()));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_profile_defaults_none() {
        let cli = Cli::try_parse_from(["oxillama", "run", "--model", "m.gguf"]).expect("parse run");
        match cli.command {
            Commands::Run(args) => assert_eq!(args.profile, None),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_add_bos_defaults_to_auto() {
        let cli = Cli::try_parse_from(["oxillama", "run", "--model", "m.gguf"]).expect("parse run");
        match cli.command {
            Commands::Run(args) => assert_eq!(args.add_bos, AddBosArg::Auto),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_add_bos_flag_parses_each_value() {
        for (spelling, expected) in [
            ("auto", AddBosArg::Auto),
            ("always", AddBosArg::Always),
            ("never", AddBosArg::Never),
        ] {
            let cli = Cli::try_parse_from([
                "oxillama",
                "run",
                "--model",
                "m.gguf",
                "--add-bos",
                spelling,
            ])
            .unwrap_or_else(|e| panic!("parse run --add-bos {spelling}: {e}"));
            match cli.command {
                Commands::Run(args) => assert_eq!(args.add_bos, expected),
                _ => panic!("expected Run"),
            }
        }
    }

    #[test]
    fn chat_profile_and_system_flags_parse() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "chat",
            "--model",
            "m.gguf",
            "--profile",
            "qwen3-7b",
            "--system",
            "Be terse.",
        ])
        .expect("parse chat --profile --system");
        match cli.command {
            Commands::Chat {
                profile, system, ..
            } => {
                assert_eq!(profile, Some("qwen3-7b".to_string()));
                assert_eq!(system, Some("Be terse.".to_string()));
            }
            _ => panic!("expected Chat"),
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn serve_profile_and_auth_flags_parse() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "serve",
            "--model",
            "m.gguf",
            "--profile",
            "qwen3-7b",
            "--api-key",
            "key-a",
            "--api-key",
            "key-b",
            "--admin-token",
            "secret",
        ])
        .expect("parse serve --profile --api-key --admin-token");
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.profile, Some("qwen3-7b".to_string()));
                assert_eq!(
                    args.api_keys,
                    vec!["key-a".to_string(), "key-b".to_string()]
                );
                assert_eq!(args.admin_token, Some("secret".to_string()));
            }
            _ => panic!("expected Serve"),
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn serve_defaults_have_no_auth() {
        let cli = Cli::try_parse_from(["oxillama", "serve", "--model", "m.gguf"])
            .expect("parse serve with defaults");
        match cli.command {
            Commands::Serve(args) => {
                assert!(args.api_keys.is_empty());
                assert_eq!(args.admin_token, None);
                assert!(!args.disable_cors);
                assert_eq!(args.max_concurrent, 64);
            }
            _ => panic!("expected Serve"),
        }
    }

    // ── C11: argument validation for the remaining subcommands ─────────────

    #[test]
    fn convert_requires_input_and_output_positionals() {
        assert!(
            Cli::try_parse_from(["oxillama", "convert"]).is_err(),
            "convert with no positional args should fail to parse"
        );
        let cli = Cli::try_parse_from(["oxillama", "convert", "in.safetensors", "out.gguf"])
            .expect("parse convert with both positionals");
        match cli.command {
            Commands::Convert(args) => {
                assert_eq!(args.input, std::path::PathBuf::from("in.safetensors"));
                assert_eq!(args.output, std::path::PathBuf::from("out.gguf"));
            }
            _ => panic!("expected Convert"),
        }
    }

    #[test]
    fn verify_optional_sha256_flag_parses() {
        let cli = Cli::try_parse_from(["oxillama", "verify", "model.gguf"])
            .expect("parse verify without --sha256");
        match cli.command {
            Commands::Verify(args) => assert_eq!(args.sha256, None),
            _ => panic!("expected Verify"),
        }

        let cli = Cli::try_parse_from(["oxillama", "verify", "model.gguf", "--sha256", "deadbeef"])
            .expect("parse verify with --sha256");
        match cli.command {
            Commands::Verify(args) => assert_eq!(args.sha256, Some("deadbeef".to_string())),
            _ => panic!("expected Verify"),
        }
    }

    #[test]
    fn quantize_target_enum_rejects_unknown_value() {
        let result = Cli::try_parse_from([
            "oxillama",
            "quantize",
            "in.gguf",
            "out.gguf",
            "--target",
            "NOT_A_REAL_FORMAT",
        ]);
        assert!(
            result.is_err(),
            "an unrecognised --target value must fail to parse"
        );
    }

    #[test]
    fn quantize_target_accepts_all_advertised_values() {
        // Every mixture `oxillama-quant` has an encoder for is a valid
        // `--target`, spelled the way llama.cpp's `quantize` tool spells it so
        // a command line moves between the two unchanged.
        for target in [
            "Q4_0", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S",
            "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
        ] {
            let result = Cli::try_parse_from([
                "oxillama", "quantize", "in.gguf", "out.gguf", "--target", target,
            ]);
            assert!(result.is_ok(), "--target {target} should parse");
        }
    }

    #[test]
    fn tokenize_format_flag_parses() {
        let cli = Cli::try_parse_from([
            "oxillama", "tokenize", "--model", "m.gguf", "--format", "json", "hello",
        ])
        .expect("parse tokenize --format json");
        match cli.command {
            Commands::Tokenize(args) => {
                assert_eq!(args.text, "hello");
                assert!(matches!(args.format, tokenize::TokenizeFormat::Json));
            }
            _ => panic!("expected Tokenize"),
        }
    }

    #[test]
    fn detokenize_ids_parse_as_comma_free_repeats() {
        let cli = Cli::try_parse_from([
            "oxillama",
            "detokenize",
            "--model",
            "m.gguf",
            "--ids",
            "1",
            "--ids",
            "2",
        ])
        .expect("parse detokenize --ids");
        match cli.command {
            Commands::Detokenize(args) => assert_eq!(args.ids, vec![1, 2]),
            _ => panic!("expected Detokenize"),
        }
    }

    #[test]
    fn hub_pull_requires_repo_positional() {
        assert!(
            Cli::try_parse_from(["oxillama", "hub", "pull"]).is_err(),
            "hub pull with no repo should fail to parse"
        );
        let cli = Cli::try_parse_from(["oxillama", "hub", "pull", "cool-japan/bonsai-8b"])
            .expect("parse hub pull with repo");
        match cli.command {
            Commands::Hub {
                command:
                    HubCommand::Pull {
                        repo,
                        revision,
                        force,
                        ..
                    },
            } => {
                assert_eq!(repo, "cool-japan/bonsai-8b");
                assert_eq!(revision, "main", "revision should default to 'main'");
                assert!(!force);
            }
            _ => panic!("expected Hub Pull"),
        }
    }

    #[test]
    fn hub_rm_requires_repo_positional() {
        assert!(Cli::try_parse_from(["oxillama", "hub", "rm"]).is_err());
        let cli = Cli::try_parse_from(["oxillama", "hub", "rm", "cool-japan/bonsai-8b"])
            .expect("parse hub rm with repo");
        match cli.command {
            Commands::Hub {
                command: HubCommand::Rm { repo, .. },
            } => assert_eq!(repo, "cool-japan/bonsai-8b"),
            _ => panic!("expected Hub Rm"),
        }
    }

    #[test]
    fn hub_list_has_no_required_args() {
        assert!(Cli::try_parse_from(["oxillama", "hub", "list"]).is_ok());
    }

    #[test]
    fn ctx_size_help_documents_default_behaviour() {
        // C12: --ctx-size's help text should describe what happens when the
        // effective context is left at its default, not just show a bare
        // number with no explanation.
        let mut cmd = Cli::command();
        let run_cmd = cmd
            .find_subcommand_mut("run")
            .expect("run subcommand should exist");
        let ctx_arg = run_cmd
            .get_arguments()
            .find(|a| a.get_id().as_str() == "ctx_size")
            .expect("--ctx-size argument should exist on run");
        let help = ctx_arg
            .get_help()
            .map(|h| h.to_string())
            .unwrap_or_default();
        assert!(
            help.contains("4096"),
            "--ctx-size help should document the min(trained, 4096) default, got: {help:?}"
        );
    }
}
