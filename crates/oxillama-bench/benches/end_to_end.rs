//! End-to-end tokens/sec Criterion benchmarks.
//!
//! Uses a synthetic stub engine to simulate inference without requiring a real
//! GGUF model file.  Each bench group exercises a different architecture and
//! quant combination so that the harness, throughput reporting, and CI gate
//! all work without model artefacts.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxillama_bench::{run_e2e_bench, ArchBenchConfig, E2eBenchConfig, InferenceBenchmark};

// ── Stub engine ─────────────────────────────────────────────────────────

/// Synthetic engine: sleeps proportional to `ns_per_token` to simulate real
/// token generation cost while keeping wall-time reasonable under `--test`.
struct StubEngine {
    /// Simulated cost per token in nanoseconds.
    ns_per_token: u64,
    /// Simulated time-to-first-token overhead in nanoseconds.
    ns_ttft: u64,
}

impl StubEngine {
    fn new(ns_per_token: u64, ns_ttft: u64) -> Self {
        Self {
            ns_per_token,
            ns_ttft,
        }
    }

    /// Busy-loop for `nanos` nanoseconds.  Using a busy-loop (rather than
    /// `std::thread::sleep`) keeps the simulated latency deterministic and
    /// avoids OS timer granularity issues in CI.
    fn spin_ns(nanos: u64) {
        if nanos == 0 {
            return;
        }
        let start = std::time::Instant::now();
        let target = std::time::Duration::from_nanos(nanos);
        while start.elapsed() < target {
            std::hint::spin_loop();
        }
    }
}

impl InferenceBenchmark for StubEngine {
    /// Returns per-token latency list in ms: element[0] is TTFT, rest are
    /// decode latencies.
    fn run_inference(&mut self, _prompt: &str, max_tokens: usize) -> Vec<f64> {
        let mut times = Vec::with_capacity(max_tokens + 1);

        // Simulate TTFT (prefill).
        Self::spin_ns(self.ns_ttft);
        times.push(self.ns_ttft as f64 / 1_000_000.0);

        // Simulate decode token-by-token.
        for _ in 0..max_tokens {
            Self::spin_ns(self.ns_per_token);
            times.push(self.ns_per_token as f64 / 1_000_000.0);
        }

        times
    }
}

// ── Bench helpers ────────────────────────────────────────────────────────

/// Common E2E bench config: 128 tokens, 512 context.  Warm-up and measure
/// counts are kept very small so `--test` finishes quickly in CI.
fn e2e_config(max_tokens: usize) -> E2eBenchConfig {
    E2eBenchConfig {
        warmup_iters: 1,
        measure_iters: 2,
        max_tokens,
        prompt: "The quick brown fox jumps over the lazy dog.".to_string(),
        track_memory: false,
    }
}

/// Drive the Criterion harness for one architecture / quant combination.
///
/// `bench_label`  — shown in Criterion output (e.g. `"e2e_llama3_q4km"`).
/// `arch_cfg`     — pre-built architecture config (for metadata / documentation).
/// `ns_per_token` — simulated per-token decode cost in nanoseconds.
/// `ns_ttft`      — simulated time-to-first-token in nanoseconds.
fn run_bench_group(
    c: &mut Criterion,
    bench_label: &str,
    arch_cfg: &ArchBenchConfig,
    ns_per_token: u64,
    ns_ttft: u64,
) {
    let n_tokens: usize = arch_cfg.decode_tokens;
    let config = e2e_config(n_tokens);

    let mut group = c.benchmark_group(bench_label);
    group.throughput(Throughput::Elements(n_tokens as u64));

    let label = format!("{}_{}tok", arch_cfg.arch_name, n_tokens);

    group.bench_function(&label, |b| {
        let mut engine = StubEngine::new(ns_per_token, ns_ttft);
        b.iter(|| {
            // run_e2e_bench drives warm-up + measure internally.
            // We wrap it with black_box so the compiler cannot elide it.
            std::hint::black_box(run_e2e_bench(&mut engine, &config))
        });
    });

    group.finish();
}

// ── Individual bench functions ───────────────────────────────────────────

/// LLaMA-3 8B · Q4_K_M · 128 tokens · 512 context.
pub fn bench_e2e_llama3_q4km(c: &mut Criterion) {
    let arch = ArchBenchConfig::llama3();
    // Simulate ~33 tok/s (30 ms/token) with 50 ms TTFT.
    run_bench_group(c, "e2e_llama3_q4km", &arch, 30_000_000, 50_000_000);
}

/// Qwen3 7B · Q4_K_M · 128 tokens · 512 context.
pub fn bench_e2e_qwen3_q4km(c: &mut Criterion) {
    let arch = ArchBenchConfig::qwen3();
    // Simulate ~28 tok/s (36 ms/token) with 55 ms TTFT.
    run_bench_group(c, "e2e_qwen3_q4km", &arch, 36_000_000, 55_000_000);
}

/// Mistral 7B · Q4_K_M · 128 tokens · 512 context.
pub fn bench_e2e_mistral_q4km(c: &mut Criterion) {
    let arch = ArchBenchConfig::mistral();
    // Simulate ~35 tok/s (28 ms/token) with 45 ms TTFT.
    run_bench_group(c, "e2e_mistral_q4km", &arch, 28_000_000, 45_000_000);
}

criterion_group!(
    benches,
    bench_e2e_llama3_q4km,
    bench_e2e_qwen3_q4km,
    bench_e2e_mistral_q4km,
);
criterion_main!(benches);
