//! Example: Engine metrics collection.
//!
//! Usage:
//! ```text
//! cargo run --example 06_metrics -- model.gguf "Hello, world"
//! ```
//!
//! Shows how to read [`EngineMetrics`] / [`MetricsSnapshot`] from an
//! [`InferenceEngine`] to observe throughput and KV-cache statistics.
//!
//! If no model path is provided the example prints a zero-state snapshot
//! to demonstrate the API surface.

use oxillama_runtime::{EngineConfig, EngineMetrics, InferenceEngine};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);

    let model_path = match args.next() {
        Some(p) => p,
        None => {
            // Demonstrate the API without a real model by printing an empty snapshot.
            println!("(No model.gguf provided — showing zero-state metrics snapshot)\n");
            let metrics = EngineMetrics::new();
            let snap = metrics.snapshot();
            print_snapshot(&snap);
            return Ok(());
        }
    };

    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    let config = EngineConfig {
        model_path,
        ..Default::default()
    };

    let mut engine = InferenceEngine::new(config);
    engine.load_model()?;

    // Snapshot before generation (all counters should be zero).
    let before = engine.metrics_snapshot();
    println!("=== Before generation ===");
    print_snapshot(&before);

    engine.generate(&prompt, 64, |tok| print!("{tok}"))?;
    println!();

    // Snapshot after generation — counters now reflect the run.
    let after = engine.metrics_snapshot();
    println!("\n=== After generation ===");
    print_snapshot(&after);

    Ok(())
}

fn print_snapshot(snap: &oxillama_runtime::MetricsSnapshot) {
    println!("  tokens_generated    : {}", snap.tokens_generated);
    println!("  tokens_prefilled    : {}", snap.tokens_prefilled);
    println!("  decode_tokens/sec   : {:.2}", snap.decode_tokens_per_sec);
    println!("  prefill_tokens/sec  : {:.2}", snap.prefill_tokens_per_sec);
    println!(
        "  kv_cache_hit_rate   : {:.2}%",
        snap.kv_cache_hit_rate * 100.0
    );
    println!("  requests_started    : {}", snap.requests_started);
    println!("  requests_completed  : {}", snap.requests_completed);
}
