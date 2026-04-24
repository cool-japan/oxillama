//! Example: Run a single inference pass.
//!
//! Usage:
//! ```text
//! cargo run --example 02_inference -- model.gguf "Hello, world"
//! ```
//!
//! If no model path is provided the example prints usage and exits cleanly.

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let model_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: 02_inference <model.gguf> [prompt]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    println!("Model : {model_path}");
    println!("Prompt: {prompt}");
    println!();

    let config = oxillama_runtime::EngineConfig {
        model_path: model_path.clone(),
        ..Default::default()
    };

    let mut engine = oxillama_runtime::InferenceEngine::new(config);

    println!("Loading model…");
    engine.load_model()?;

    println!("Generating (max 128 tokens)…");
    engine.generate(&prompt, 128, |tok| print!("{tok}"))?;
    println!();

    Ok(())
}
