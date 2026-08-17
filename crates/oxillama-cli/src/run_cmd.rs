//! `oxillama run --model <model.gguf>` — one-shot completion.
//!
//! Resolves the prompt (`--file` > `--prompt` > stdin), layers the named
//! sampler profile over the command-line flags and the loaded config, loads
//! the model and streams the generated text to stdout. `--dump-logits`
//! diverts into the debug path in [`crate::dump_logits`] instead of
//! streaming text.
//!
//! Precedence for the sampler/context knobs, weakest first: built-in clap
//! default → `OxillamaConfig` (`default_ctx_size`, `default_threads`) →
//! command-line flag → model profile.

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli_args::{tokenize_with_add_bos, AddBosArg, KvDtypeArg};
use crate::config::{self, OxillamaConfig};
use crate::dump_logits;

/// Arguments for the `run` subcommand.
#[derive(clap::Args)]
pub struct RunArgs {
    /// Path to the GGUF model file.
    #[arg(short, long)]
    pub model: String,

    /// Input prompt text (reads stdin if neither --prompt nor --file is given).
    #[arg(short, long, default_value = "")]
    pub prompt: String,

    /// Read prompt from this file instead of --prompt or stdin.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    pub file: Option<PathBuf>,

    /// Read prompt from stdin explicitly (pipe mode); conflicts with --prompt and --file.
    #[arg(long, conflicts_with_all = ["prompt", "file"])]
    pub stdin: bool,

    /// Path to tokenizer.json (auto-detected if not provided).
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,

    /// llama.cpp alias: max tokens to predict (same as --max-tokens).
    #[arg(short = 'n', long = "n-predict", conflicts_with = "max_tokens")]
    pub n_predict: Option<usize>,

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

    /// Temperature for sampling.
    #[arg(long, default_value_t = 0.7)]
    pub temp: f32,

    /// llama.cpp alias for --temp.
    #[arg(long = "temperature", conflicts_with = "temp")]
    pub temperature: Option<f32>,

    /// Top-P for nucleus sampling.
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    /// Top-K for sampling.
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Seed for reproducible sampling (0 = random).
    #[arg(short = 's', long, default_value_t = 0u64)]
    pub seed: u64,

    /// Repetition penalty (1.0 = disabled).
    #[arg(long, default_value_t = 1.1f32)]
    pub repeat_penalty: f32,

    /// Min-P sampling threshold (0.0 = disabled).
    #[arg(long, default_value_t = 0.05f32)]
    pub min_p: f32,

    /// Print verbose version banner and exit.
    #[arg(long)]
    pub verbose_version: bool,

    /// Named sampler profile to apply (`~/.config/oxillama/models/<name>.toml`).
    /// Falls back to a profile matching the model's file stem when omitted.
    #[arg(long, value_name = "NAME")]
    pub profile: Option<String>,

    /// KV cache storage type: `f32` (lossless, default) or `f16` (half the
    /// KV memory). Attention still accumulates in f32 either way.
    #[arg(long = "kv-dtype", value_enum, default_value_t = KvDtypeArg::F32)]
    pub kv_dtype: KvDtypeArg,

    /// Override whether BOS is prepended to the prompt: `auto` (default,
    /// OxiLLaMa's own policy), `always`, or `never`. Use `never` to
    /// reproduce llama.cpp's output on a LLaMA-3 GGUF that lacks
    /// `tokenizer.ggml.pre`, where llama.cpp defaults to `add_bos =
    /// false` but OxiLLaMa infers the LLaMA-3 pre-tokenizer from the
    /// vocabulary and defaults to `add_bos = true`.
    #[arg(long = "add-bos", value_enum, default_value_t = AddBosArg::Auto)]
    pub add_bos: AddBosArg,

    /// (debug) Write the full final-position logit vector of every
    /// generation step into DIR.
    ///
    /// DIR receives `step<k>.logits.f32.bin` — headerless, `vocab_size`
    /// little-endian `f32` values, index == token id, raw pre-softmax
    /// logits — plus a `manifest.json` recording the prompt token ids,
    /// whether BOS was applied, the emitted token ids and the vocab size.
    /// This is a verification hook for comparing OxiLLaMa against another
    /// implementation; it is not part of normal inference. Generation runs
    /// the full `--max-tokens` steps and does **not** stop at an
    /// end-of-generation token, so that step indices stay comparable with
    /// a reference dump; the first EOG is recorded as `eog_at_step`.
    #[arg(long, value_name = "DIR")]
    pub dump_logits: Option<PathBuf>,

    /// (debug) Use these comma-separated token IDs as the prompt, bypassing
    /// the tokenizer entirely (no BOS is added). Requires --dump-logits.
    ///
    /// Cross-implementation logit comparisons must feed both sides the
    /// identical token ids, or a tokenizer difference is indistinguishable
    /// from a kernel bug.
    #[arg(long, value_name = "IDS", requires = "dump_logits",
          conflicts_with_all = ["prompt", "file", "stdin"])]
    pub prompt_tokens: Option<String>,

    /// (debug) Teacher forcing: override the token fed into each step with
    /// these comma-separated token IDs instead of the sampled one.
    /// Requires --dump-logits.
    ///
    /// Without it, one divergent token makes every later step's logits
    /// incomparable with a reference run, because the two models are no
    /// longer conditioned on the same prefix.
    #[arg(long, value_name = "IDS", requires = "dump_logits")]
    pub force_tokens: Option<String>,
}

/// Execute the `run` subcommand.
///
/// `cfg` supplies the config-file fallbacks (`default_ctx_size`,
/// `default_threads`) that sit between the clap defaults and the model
/// profile; it is not consulted for anything else.
///
/// # Errors
///
/// Propagates prompt-resolution, model-loading, tokenization and generation
/// failures. The `std::io::Error` behind an unreadable prompt file is kept as
/// the error's `source()` so `exit_codes::classify` can see it.
pub async fn run_run(args: RunArgs, cfg: &OxillamaConfig) -> Result<()> {
    let RunArgs {
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
        profile,
        kv_dtype,
        add_bos,
        dump_logits: dump_logits_dir,
        prompt_tokens,
        force_tokens,
    } = args;

    if verbose_version {
        crate::print_verbose_version();
        return Ok(());
    }

    // Parsed up front so a typo fails before a multi-GB model load.
    let explicit_prompt_tokens = prompt_tokens
        .as_deref()
        .map(dump_logits::parse_token_ids)
        .transpose()
        .context("parsing --prompt-tokens")?;
    let forced_tokens = force_tokens
        .as_deref()
        .map(dump_logits::parse_token_ids)
        .transpose()
        .context("parsing --force-tokens")?
        .unwrap_or_default();
    if let Some(ref ids) = explicit_prompt_tokens {
        if ids.is_empty() {
            anyhow::bail!("--prompt-tokens is empty; give at least one token id");
        }
    }

    let effective_max_tokens = n_predict.unwrap_or(max_tokens);
    let effective_temp = temperature.unwrap_or(temp);
    let effective_ctx = n_ctx.unwrap_or(cfg.default_ctx_size.unwrap_or(ctx_size));
    let effective_threads = cfg.default_threads.unwrap_or(threads);
    let effective_seed = if seed == 0 { None } else { Some(seed) };

    // Resolve prompt: --file > --prompt > --stdin / empty prompt fallback.
    //
    // `.with_context` (rather than `anyhow!("...: {e}")`) keeps the
    // original `std::io::Error` as this error's `source()` instead of
    // only interpolating its message as text. `exit_codes::classify`
    // downcasts on the typed source, so losing it here would make an
    // unreadable prompt file misclassify as a model-loading failure.
    let effective_prompt = if explicit_prompt_tokens.is_some() {
        // The prompt arrived as token ids; there is no text to resolve,
        // and falling through would block on stdin.
        String::new()
    } else if let Some(ref path) = file {
        std::fs::read_to_string(path)
            .with_context(|| format!("reading prompt file '{}'", path.display()))?
    } else if stdin || prompt.is_empty() {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading stdin")?;
        buf
    } else {
        prompt.clone()
    };

    // Resolve a per-model sampler profile: --profile <name> wins,
    // otherwise fall back to a profile matching the model's file stem.
    let model_stem = std::path::Path::new(&model)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let profile_name = profile.unwrap_or_else(|| model_stem.clone());
    let model_profile = config::load_model_profile(&profile_name).unwrap_or_default();

    let final_temp = model_profile
        .as_ref()
        .and_then(|p| p.temp)
        .unwrap_or(effective_temp);
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
        .unwrap_or(effective_threads);

    // System prompt (from the profile): `run` is a raw-completion
    // entry point rather than a turn-based chat REPL, so there is no
    // per-turn chat template to route it through here (see `chat`
    // for that) — it is simply prepended once, ahead of the prompt.
    let effective_prompt = match model_profile.as_ref().and_then(|p| p.system_prompt.clone()) {
        Some(sys) => format!("{sys}\n\n{effective_prompt}"),
        None => effective_prompt,
    };

    tracing::info!(
        model = %model,
        ctx_size = final_ctx,
        threads = final_threads,
        temp = final_temp,
        top_p = final_top_p,
        top_k = final_top_k,
        kv_dtype = ?kv_dtype,
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
        kv_dtype: kv_dtype.into(),
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

    if let Some(out_dir) = dump_logits_dir {
        // Debug path: dump raw per-step logits instead of streaming
        // text. `--prompt-tokens` bypasses the tokenizer so that a
        // cross-implementation comparison is not contaminated by a
        // tokenization difference.
        let prompt_token_ids = match explicit_prompt_tokens {
            Some(ids) => ids,
            None => {
                let tok = engine
                    .tokenizer()
                    .context("--dump-logits requires a loaded tokenizer")?;
                tokenize_with_add_bos(tok, &effective_prompt, add_bos)?
            }
        };
        let options = dump_logits::DumpOptions {
            out_dir: out_dir.clone(),
            max_steps: effective_max_tokens,
            sampler: engine.config().sampler.clone(),
            forced_tokens,
            prompt: match prompt_tokens {
                Some(_) => None,
                None => Some(effective_prompt.clone()),
            },
            model_path: model.clone(),
        };
        let manifest = dump_logits::run(&mut engine, &prompt_token_ids, &options)?;
        println!("{}", manifest.continuation);
        eprintln!(
            "{} {} step(s) + manifest.json written to {}",
            "✓".green(),
            manifest.steps.len(),
            out_dir.display()
        );
        eprintln!(
            "  prompt tokens: {} (add_bos_applied={}), emitted: {:?}",
            manifest.n_prompt_tokens, manifest.add_bos_applied, manifest.emitted_token_ids
        );
        return Ok(());
    }

    let tok = engine
        .tokenizer()
        .context("run requires a loaded tokenizer")?;
    let prompt_token_ids = tokenize_with_add_bos(tok, &effective_prompt, add_bos)?;
    let run_gen_config = oxillama_runtime::GenerationConfig {
        max_tokens: effective_max_tokens,
        sampler: engine.config().sampler.clone(),
        ..Default::default()
    };
    engine.generate_detailed_with_tokens(prompt_token_ids, &run_gen_config, |token| {
        print!("{token}");
    })?;
    println!();
    eprintln!("{}", "✓ Done".green());

    Ok(())
}
