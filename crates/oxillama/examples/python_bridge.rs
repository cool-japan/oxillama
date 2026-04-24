//! Example: Python bridge — Rust ↔ Python API parity.
//!
//! This example shows the Rust API surface that `oxillama-py` wraps.
//! Each annotated line maps to the corresponding `oxillama_py` Python call.
//!
//! Python equivalent:
//! ```python
//! import oxillama_py as ox
//!
//! engine = ox.InferenceEngine("model.gguf")
//! engine.load_model()
//! result = engine.generate("Hello", max_tokens=128)
//! print(result)
//! ```
//!
//! Usage:
//! ```text
//! cargo run --example python_bridge -- model.gguf "Hello"
//! ```
//!
//! If no model path is provided the example exits cleanly.

fn rust_inference_example(model_path: &str, prompt: &str) -> anyhow::Result<()> {
    use oxillama_runtime::{EngineConfig, InferenceEngine};

    // Python: engine = ox.InferenceEngine("model.gguf")
    let config = EngineConfig {
        model_path: model_path.to_string(),
        ..Default::default()
    };
    let mut engine = InferenceEngine::new(config);

    // Python: engine.load_model()  [implicit in __init__ on the Python side]
    eprintln!("Loading model: {model_path}");
    engine.load_model()?;
    eprintln!("Model loaded.");

    // Python: result = engine.generate(prompt, max_tokens=128)
    //
    // The Rust API streams tokens via a callback; `oxillama-py` collects them
    // into a single `str` before returning to the caller.
    let mut collected = String::new();
    engine.generate(prompt, 128, |tok| collected.push_str(tok))?;

    // Python: print(result)
    eprintln!("--- output ---");
    println!("{collected}");
    eprintln!("--- end ---");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let model_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: python_bridge <model.gguf> [prompt]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    rust_inference_example(&model_path, &prompt)
}
