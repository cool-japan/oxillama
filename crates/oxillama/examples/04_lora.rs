//! Example: Load and apply a LoRA adapter.
//!
//! Usage:
//! ```text
//! cargo run --example 04_lora -- base.gguf adapter.gguf "Hello"
//! ```
//!
//! If no paths are provided the example exits cleanly.

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let base_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: 04_lora <base.gguf> <adapter.gguf> [prompt]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let adapter_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: 04_lora <base.gguf> <adapter.gguf> [prompt]");
            eprintln!("(No adapter path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    // 1. Load the base model.
    let config = oxillama_runtime::EngineConfig {
        model_path: base_path,
        ..Default::default()
    };
    let mut engine = oxillama_runtime::InferenceEngine::new(config);
    engine.load_model()?;
    println!("Base model loaded.");

    // 2. Apply the LoRA adapter on top.
    oxillama_runtime::apply_lora(&mut engine, &adapter_path)?;
    println!("LoRA adapter applied: {adapter_path}");

    // 3. Generate with the patched model.
    println!("Generating…");
    engine.generate(&prompt, 128, |tok| print!("{tok}"))?;
    println!();

    Ok(())
}
