//! Example: Speculative decoding with a draft model.
//!
//! Usage:
//! ```text
//! cargo run --example 05_speculative -- target.gguf draft.gguf "Hello"
//! ```
//!
//! Speculative decoding uses a small draft model to propose candidate tokens
//! that are verified in bulk by the larger target model, achieving higher
//! throughput while maintaining identical output distribution.
//!
//! If no paths are provided the example exits cleanly.

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let target_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: 05_speculative <target.gguf> <draft.gguf> [prompt]");
            eprintln!("(No model path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let draft_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: 05_speculative <target.gguf> <draft.gguf> [prompt]");
            eprintln!("(No draft path provided — exiting cleanly)");
            return Ok(());
        }
    };

    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    let target_config = oxillama_runtime::EngineConfig {
        model_path: target_path,
        ..Default::default()
    };
    let draft_config = oxillama_runtime::EngineConfig {
        model_path: draft_path,
        ..Default::default()
    };

    let spec_config = oxillama_runtime::SpeculativeConfig::new(target_config, draft_config);

    let mut spec_engine = oxillama_runtime::SpeculativeEngine::new(spec_config)?;

    println!("Running speculative decoding…");
    use std::io::Write as _;
    spec_engine.generate(&prompt, 128, |tok| {
        print!("{tok}");
        let _ = std::io::stdout().flush();
    })?;
    println!();

    Ok(())
}
