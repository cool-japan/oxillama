//! Example: Streaming token-by-token output.
//!
//! Usage:
//! ```text
//! cargo run --example 03_streaming -- model.gguf "Tell me a story"
//! ```
//!
//! Tokens are printed as they are produced rather than buffering the full
//! response. If no model path is provided the example exits cleanly.

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let model_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: 03_streaming <model.gguf> [prompt]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let prompt = args
        .next()
        .unwrap_or_else(|| "Tell me a story.".to_string());

    let config = oxillama_runtime::EngineConfig {
        model_path: model_path.clone(),
        ..Default::default()
    };

    let mut engine = oxillama_runtime::InferenceEngine::new(config);
    engine.load_model()?;

    println!("Streaming response for: {prompt}");
    println!("---");

    // The callback is invoked for each decoded token fragment.
    // Use `std::io::Write::flush` to ensure each piece is visible immediately.
    use std::io::Write as _;
    engine.generate(&prompt, 256, |tok| {
        print!("{tok}");
        let _ = std::io::stdout().flush();
    })?;
    println!("\n---");
    println!("Done.");

    Ok(())
}
