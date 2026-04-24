//! Example: GPU-accelerated inference with the `gpu` feature.
//!
//! When built with `--features gpu`, Q4_0 and Q8_0 matrix-vector multiplications
//! are dispatched through the `wgpu` compute pipeline, reducing memory bandwidth
//! versus the CPU path.
//!
//! Usage:
//! ```text
//! cargo run --example gpu_enabled --features gpu -- model.gguf "Hello"
//! ```
//!
//! Without the `gpu` feature the binary compiles and runs but falls back to the
//! CPU path automatically.

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let model_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: gpu_enabled [--features gpu] <model.gguf> [prompt]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let prompt = args
        .next()
        .unwrap_or_else(|| "Hello from the GPU path!".to_string());

    // Print GPU support status at runtime.
    #[cfg(feature = "gpu")]
    {
        println!("GPU feature: enabled");
        // Enumerate available GPU adapters via oxillama-gpu.
        let adapters = oxillama::gpu::GpuDispatcher::enumerate_devices();
        if adapters.is_empty() {
            println!("  No GPU adapters found — execution will use CPU fallback.");
        } else {
            println!("  Available adapters:");
            for (i, a) in adapters.iter().enumerate() {
                println!("    [{i}] {} ({}, {})", a.name, a.backend, a.device_type);
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature: disabled (compile with --features gpu to enable)");
    }

    println!("Model : {model_path}");
    println!("Prompt: {prompt}");
    println!();

    // Build engine config — the runtime picks up the GPU dispatcher automatically
    // when the `gpu` feature is enabled and a compatible adapter is present.
    let config = oxillama_runtime::EngineConfig {
        model_path,
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
