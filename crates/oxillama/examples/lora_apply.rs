//! Example: Hot-swap LoRA adapters on a loaded model.
//!
//! # Usage
//!
//! ```text
//! cargo run --example lora_apply -- \
//!     --model base.gguf \
//!     --adapter-a style_a.gguf \
//!     --adapter-b style_b.gguf \
//!     --prompt "Translate: hello world"
//! cargo run --example lora_apply -- --help
//! ```
//!
//! # What this example demonstrates
//!
//! - Loading the base model with [`InferenceEngine::new`] + [`load_model`].
//! - Loading a LoRA adapter from a GGUF file via [`LoadedLora::load`].
//! - Pushing an adapter onto the engine's LoRA stack with [`push_lora`] and
//!   materialising the merged weights via [`apply_lora_stack`].
//! - Generating text with the adapter active.
//! - Hot-swapping to a second adapter by calling [`pop_lora`], pushing the new
//!   adapter, and calling [`apply_lora_stack`] again — without reloading the
//!   base model.
//! - Clearing all adapters and generating with the bare base model.

use std::io::Write as _;
use std::sync::Arc;

use clap::Parser;
use oxillama_arch::lora::LoadedLora;
use oxillama_runtime::{EngineConfig, InferenceEngine, SamplerConfig};

/// Load a base model and demonstrate LoRA adapter hot-swap.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the base GGUF model file.
    #[arg(long, short = 'm')]
    model: String,

    /// Path to the first LoRA adapter GGUF file.
    #[arg(long)]
    adapter_a: Option<String>,

    /// Path to an optional second LoRA adapter for hot-swap demonstration.
    #[arg(long)]
    adapter_b: Option<String>,

    /// Text prompt.
    #[arg(long, short = 'p', default_value = "Translate: hello world")]
    prompt: String,

    /// Maximum tokens per generation pass.
    #[arg(long, short = 'n', default_value_t = 64)]
    max_tokens: usize,

    /// LoRA scale multiplier applied during the stack apply step.
    #[arg(long, default_value_t = 1.0)]
    lora_scale: f32,

    /// Number of CPU threads for the forward pass.
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

fn generate_and_print(
    engine: &mut InferenceEngine,
    prompt: &str,
    max_tokens: usize,
    label: &str,
) -> anyhow::Result<()> {
    eprintln!("\n[{label}] Prompt: {prompt}");
    let stdout = std::io::stdout();
    let mut locked = stdout.lock();
    engine.generate_with_config(prompt, max_tokens, SamplerConfig::default(), |tok| {
        let _ = locked.write_all(tok.as_bytes());
        let _ = locked.flush();
    })?;
    drop(locked);
    eprintln!(); // newline after generation
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // ── 1. Load the base model ────────────────────────────────────────────────
    eprintln!("Loading base model: {}", args.model);
    let config = EngineConfig {
        model_path: args.model.clone(),
        num_threads: args.threads,
        ..EngineConfig::default()
    };
    let mut engine = InferenceEngine::new(config);
    engine.load_model()?;
    eprintln!("  Base model loaded. is_loaded={}", engine.is_loaded());

    // ── 2. Generate once with bare base model ─────────────────────────────────
    generate_and_print(&mut engine, &args.prompt, args.max_tokens, "base (no LoRA)")?;

    // ── 3. Push adapter A if provided ─────────────────────────────────────────
    if let Some(ref path_a) = args.adapter_a {
        eprintln!("\nLoading LoRA adapter A: {path_a}");
        let lora_a = LoadedLora::load(path_a)?;
        let lora_a_arc = Arc::new(lora_a);
        eprintln!(
            "  Adapter A: rank={} alpha={}",
            lora_a_arc.rank, lora_a_arc.alpha
        );

        // Push onto the stack with the caller-supplied scale.
        engine.push_lora(Arc::clone(&lora_a_arc), args.lora_scale);
        eprintln!("  Stack depth after push: {}", engine.lora_stack().len());

        // Materialise the merged weights from the stack.
        engine.apply_lora_stack()?;
        eprintln!("  Adapter A applied to model weights.");

        generate_and_print(&mut engine, &args.prompt, args.max_tokens, "adapter A")?;

        // ── 4. Hot-swap to adapter B if provided ──────────────────────────────
        if let Some(ref path_b) = args.adapter_b {
            eprintln!("\nHot-swapping to LoRA adapter B: {path_b}");

            // Remove A from the stack.
            let _popped = engine.pop_lora();
            eprintln!(
                "  Adapter A popped. Stack depth: {}",
                engine.lora_stack().len()
            );

            let lora_b = LoadedLora::load(path_b)?;
            let lora_b_arc = Arc::new(lora_b);
            eprintln!(
                "  Adapter B: rank={} alpha={}",
                lora_b_arc.rank, lora_b_arc.alpha
            );

            engine.push_lora(lora_b_arc, args.lora_scale);
            engine.apply_lora_stack()?;
            eprintln!("  Adapter B applied to model weights.");

            generate_and_print(&mut engine, &args.prompt, args.max_tokens, "adapter B")?;
        }

        // ── 5. Clear all adapters and run the bare base model again ───────────
        engine.clear_loras();
        eprintln!(
            "\nAll adapters cleared. Stack depth: {}",
            engine.lora_stack().len()
        );

        // Reset the engine KV state for a clean comparison.
        engine.reset();

        generate_and_print(
            &mut engine,
            &args.prompt,
            args.max_tokens,
            "base (adapters cleared)",
        )?;
    } else {
        eprintln!("\n(No adapter paths provided — skipping LoRA demo.)");
        eprintln!("Use --adapter-a <path.gguf> to enable the hot-swap demonstration.");
    }

    eprintln!("\nDone.");
    Ok(())
}
