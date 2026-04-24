//! Memory profiler benchmark harness.
//!
//! This is a plain binary-mode bench (not a Criterion bench) that demonstrates
//! the `AsyncMemoryProfiler` API.  It starts the profiler, performs a series of
//! allocations that mimic KV-cache and weight-load events, waits for several
//! sample intervals, and then writes the resulting profile report to
//! `target/bench-memory-<timestamp>.json`.
//!
//! Run with:
//! ```
//! cargo bench -p oxillama-bench --bench memory
//! ```

#[tokio::main]
async fn main() {
    use oxillama_bench::memory_profiler::{AsyncMemoryProfiler, MemEventKind};
    use std::time::SystemTime;

    println!("Starting AsyncMemoryProfiler demo bench...");

    let profiler = AsyncMemoryProfiler::start(20); // sample every 20ms

    // Simulate weight loading
    profiler.record_event(MemEventKind::WeightsLoaded, 4_000_000_000);

    // Simulate KV-cache alloc
    let _kv_cache = vec![0u8; 16 * 1024 * 1024]; // 16 MiB
    profiler.record_event(MemEventKind::KvCacheAlloc, 16 * 1024 * 1024);

    // Simulate state alloc
    let _scratch = vec![0f32; 4096 * 4096];
    profiler.record_event(MemEventKind::StateAlloc, 4096 * 4096 * 4);

    // Wait for a few sample intervals
    tokio::time::sleep(tokio::time::Duration::from_millis(120)).await;

    // Simulate KV-cache free
    drop(_kv_cache);
    profiler.record_event(MemEventKind::KvCacheFree, 16 * 1024 * 1024);

    tokio::time::sleep(tokio::time::Duration::from_millis(40)).await;

    let report = profiler.stop();
    report.print_summary();

    // Write to target/bench-memory-<timestamp>.json
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let out_path = std::path::PathBuf::from(format!("target/bench-memory-{ts}.json"));
    if let Err(e) = report.write_json(&out_path) {
        eprintln!("Warning: could not write JSON report: {e}");
    } else {
        println!("Report written to {}", out_path.display());
    }
}
