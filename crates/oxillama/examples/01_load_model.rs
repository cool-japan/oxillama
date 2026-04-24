//! Example: Load a GGUF model and inspect its metadata.
//!
//! Usage:
//! ```text
//! cargo run --example 01_load_model -- path/to/model.gguf
//! ```
//!
//! If no path is provided, the example prints usage and exits cleanly.

fn main() -> anyhow::Result<()> {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("Usage: 01_load_model <model.gguf>");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let model = oxillama_gguf::GgufModel::load(&path)?;

    let arch = model.architecture().unwrap_or("unknown");
    println!("Architecture: {arch}");
    println!("Tensors: {}", model.file.header.tensor_count);

    println!("\nFirst 10 metadata entries:");
    for (key, value) in model.file.metadata.iter().take(10) {
        println!("  {key}: {value}");
    }

    Ok(())
}
