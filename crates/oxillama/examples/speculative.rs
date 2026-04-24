//! Example: Speculative decoding with a draft + target model pair.
//!
//! # Usage
//!
//! ```text
//! cargo run --example speculative -- \
//!     --target llama-3-8b.gguf \
//!     --draft llama-3-1b.gguf \
//!     --prompt "The meaning of life is"
//! cargo run --example speculative -- --help
//! ```
//!
//! # What this example demonstrates
//!
//! - Building a [`SpeculativeConfig`] with a target model (large, accurate) and
//!   a draft model (small, fast).
//! - Constructing a [`SpeculativeEngine`] which loads both models.
//! - Running [`SpeculativeEngine::generate`] and measuring:
//!   - total wall time,
//!   - total accepted tokens,
//!   - estimated acceptance rate based on per-round statistics.
//! - The token-accept/reject loop is invisible to the caller — only accepted
//!   tokens are delivered to the callback.
//!
//! # Algorithm sketch
//!
//! Each speculation round:
//! 1. **Draft phase** — draft model proposes `num_speculative` candidate tokens.
//! 2. **Verify phase** — target model scores each candidate.
//! 3. **Accept/reject** — tokens accepted when `p_target ≥ p_draft`; otherwise
//!    a corrected "bonus" token is sampled from the residual distribution.
//! 4. Accepted tokens are emitted to the callback; the draft KV cache is
//!    re-synchronised.

use std::io::Write as _;
use std::time::Instant;

use clap::Parser;
use oxillama_runtime::{EngineConfig, SamplerConfig, SpeculativeConfig, SpeculativeEngine};

/// Run speculative decoding with a draft and target model pair.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the large (target / verification) GGUF model.
    #[arg(long, short = 't')]
    target: String,

    /// Path to the small (draft / speculation) GGUF model.
    #[arg(long, short = 'd')]
    draft: String,

    /// Text prompt.
    #[arg(long, short = 'p', default_value = "The meaning of life is")]
    prompt: String,

    /// Maximum number of accepted tokens to generate.
    #[arg(long, short = 'n', default_value_t = 200)]
    max_tokens: usize,

    /// Number of candidate tokens the draft model proposes per round (k).
    ///
    /// Larger values give more potential speedup but risk more resampling when
    /// the draft and target distributions diverge.  A value of 4–8 is typical.
    #[arg(long, short = 'k', default_value_t = 5)]
    num_speculative: usize,

    /// Random seed for the accept/reject sampler (0 = use internal default).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// CPU threads for the target model.
    #[arg(long, default_value_t = 4)]
    target_threads: usize,

    /// CPU threads for the draft model.
    #[arg(long, default_value_t = 2)]
    draft_threads: usize,

    /// Sampling temperature applied to the draft model.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Top-K cutoff applied to the draft sampler (0 = disabled).
    #[arg(long, default_value_t = 40)]
    top_k: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── 1. Build target + draft engine configurations ─────────────────────────
    let draft_sampler = SamplerConfig {
        temperature: args.temperature as f32,
        top_k: args.top_k,
        ..SamplerConfig::default()
    };

    let target_config = EngineConfig {
        model_path: args.target.clone(),
        num_threads: args.target_threads,
        ..EngineConfig::default()
    };

    let draft_config = EngineConfig {
        model_path: args.draft.clone(),
        num_threads: args.draft_threads,
        sampler: draft_sampler,
        ..EngineConfig::default()
    };

    // ── 2. Build the speculative config ───────────────────────────────────────
    let mut spec_config = SpeculativeConfig::new(target_config, draft_config);
    spec_config.num_speculative = args.num_speculative;
    if args.seed != 0 {
        spec_config.seed = Some(args.seed);
    }

    eprintln!("Loading target model : {}", args.target);
    eprintln!("Loading draft model  : {}", args.draft);
    eprintln!(
        "Speculation window   : {} tokens/round",
        args.num_speculative
    );

    // ── 3. Construct the engine (loads both models) ───────────────────────────
    let mut engine = SpeculativeEngine::new(spec_config)?;
    eprintln!("Both models loaded.\n");

    // ── 4. Run speculative generation ─────────────────────────────────────────
    eprintln!("Prompt: {}", args.prompt);
    eprintln!("---");

    let stdout = std::io::stdout();
    let mut locked = stdout.lock();
    let mut accepted_tokens = 0usize;

    let wall_start = Instant::now();

    engine.generate(&args.prompt, args.max_tokens, |tok| {
        let _ = locked.write_all(tok.as_bytes());
        let _ = locked.flush();
        accepted_tokens += 1;
    })?;

    let elapsed = wall_start.elapsed();
    drop(locked); // release stdout lock

    // ── 5. Report statistics ──────────────────────────────────────────────────
    //
    // Acceptance rate is estimated from the number of accepted tokens and the
    // maximum that could have been accepted if every draft token was correct.
    // The true acceptance rate requires per-round counters which SpeculativeEngine
    // does not yet expose; we report the observable tokens/second here.
    let tps = if elapsed.as_secs_f64() > 0.0 {
        accepted_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("\n---");
    eprintln!("Accepted tokens : {accepted_tokens}");
    eprintln!("Wall time       : {:.2}s", elapsed.as_secs_f64());
    eprintln!("Throughput      : {tps:.1} accepted tok/s");
    eprintln!("Draft window    : {} tokens/round", args.num_speculative);
    eprintln!(
        "Note: acceptance rate requires per-round counters \
         (not yet exposed by SpeculativeEngine)."
    );

    Ok(())
}
