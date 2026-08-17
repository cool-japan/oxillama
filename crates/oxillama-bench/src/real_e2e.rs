//! Real end-to-end benchmark against `oxillama-runtime`.
//!
//! Every other harness in this crate (`e2e.rs`'s `InferenceBenchmark` trait,
//! `prefill_decode.rs`'s `PrefillDecodeBench` trait, and the `StubEngine`
//! types driving `benches/end_to_end.rs` / `benches/kv_cache.rs`) is a
//! synthetic timer: it busy-loops for a configured number of nanoseconds and
//! calls that "inference". None of them load a GGUF file or run a forward
//! pass. `oxillama-bench` had no dependency on `oxillama-runtime` at all
//! until this module — there was no way to get a real tokens/sec number out
//! of this crate.
//!
//! This module drives the actual [`InferenceEngine`] against a real GGUF
//! file on disk and measures:
//!
//! 1. **Model load wall-time** — [`InferenceEngine::load_model`].
//! 2. **Prefill throughput** — a single batched [`InferenceEngine::forward_prefill`]
//!    call over the whole prompt.
//! 3. **Decode throughput** — exactly [`DECODE_TOKENS`] greedy (argmax) steps
//!    via [`InferenceEngine::forward_decode`], deliberately ignoring EOS (see
//!    [`run_real_e2e_bench_for`] for why).
//! 4. **Peak RSS** — sampled with [`crate::memory::MemoryProfiler`] at every
//!    phase boundary and after every decode step.
//!
//! The model path is never hardcoded: it comes from the [`MODEL_ENV_VAR`]
//! environment variable, and [`run_real_e2e_bench`] returns `Ok(None)` —
//! never an error — when it is unset, so this crate, its tests, and CI all
//! keep working without a multi-gigabyte model file on disk.

use std::ffi::OsString;
use std::time::{Duration, Instant};

use oxillama_runtime::{EngineConfig, InferenceEngine, RuntimeResult};

use crate::memory::{MemoryProfiler, MemoryReport};

/// Environment variable naming the GGUF model to benchmark end-to-end.
///
/// Unset by default so `cargo bench -p oxillama-bench --bench real_e2e` (and
/// CI, which has no multi-gigabyte model file on disk) skips gracefully
/// instead of failing. Point it at a real GGUF file to get real numbers:
///
/// ```text
/// OXILLAMA_BENCH_MODEL=/path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
///     cargo bench -p oxillama-bench --bench real_e2e
/// ```
pub const MODEL_ENV_VAR: &str = "OXILLAMA_BENCH_MODEL";

/// Fixed decode length for the greedy decode phase.
///
/// This is a mission-fixed constant, not a config field: the point of this
/// benchmark is a reproducible number to track release over release, so the
/// decode phase always runs exactly this many steps regardless of prompt or
/// model.
pub const DECODE_TOKENS: usize = 64;

/// Default benchmark prompt.
///
/// Long enough to exercise a real batched prefill call (tens of tokens)
/// without depending on any specific tokenizer's BPE merges to land on an
/// exact token count.
pub const DEFAULT_PROMPT: &str =
    "You are a helpful assistant. Explain, in a few sentences, why the sky \
     appears blue during the day and orange at sunset.";

/// Configuration for a real end-to-end benchmark run.
#[derive(Debug, Clone)]
pub struct RealE2eConfig {
    /// Prompt used for the prefill phase.
    pub prompt: String,
    /// Optional context-size override forwarded to [`EngineConfig`].
    /// `None` uses the engine's own default (`min(n_ctx_train, 4096)`).
    pub context_size: Option<usize>,
}

impl Default for RealE2eConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_PROMPT.to_string(),
            context_size: None,
        }
    }
}

/// Outcome of a real end-to-end benchmark run.
#[derive(Debug, Clone)]
pub struct RealE2eReport {
    /// Path to the GGUF model that was benchmarked.
    pub model_path: String,
    /// Size of the GGUF file on disk, in bytes (`0` if `stat` failed).
    pub model_file_bytes: u64,
    /// Wall-clock time to [`InferenceEngine::load_model`], in milliseconds.
    ///
    /// Model loading is mmap-backed (this crate depends on
    /// `oxillama-gguf/mmap` via `oxillama-runtime/mmap`): this measures the
    /// time to map the file and build the architecture's forward pass, not a
    /// full read of the weights into RAM. Pages are faulted in lazily as
    /// prefill/decode touch them, which is why `memory.peak_bytes` keeps
    /// growing past this point rather than plateauing here.
    pub load_wall_ms: f64,
    /// Effective context size the engine allocated the KV cache for.
    pub effective_context_size: usize,
    /// Number of tokens the prompt encoded to (with the model's own
    /// BOS/special-token policy applied, matching what a real `generate()`
    /// call would prefill).
    pub prompt_tokens: usize,
    /// Wall-clock time for the single batched prefill forward call, in ms.
    pub prefill_ms: f64,
    /// `prompt_tokens / prefill_ms * 1000`. `0.0` if `prefill_ms` is `0.0`.
    pub prefill_tokens_per_sec: f64,
    /// Number of tokens greedily decoded. Always [`DECODE_TOKENS`] — EOS is
    /// deliberately ignored so this always reflects the same fixed amount of
    /// work regardless of prompt or model.
    pub decode_tokens: usize,
    /// Sum of the `decode_tokens` per-token forward-call latencies, in ms.
    /// Excludes prefill and the trivial per-step argmax.
    pub decode_ms: f64,
    /// `decode_tokens / decode_ms * 1000`. `0.0` if `decode_ms` is `0.0`.
    pub decode_tokens_per_sec: f64,
    /// RSS samples taken at every phase boundary (construction, post-load,
    /// post-tokenize, post-prefill, after every decode step). `peak_bytes`
    /// is the number to watch; the rest is context for interpreting it.
    pub memory: MemoryReport,
}

impl RealE2eReport {
    /// Peak RSS observed across the whole run (construction through the end
    /// of decode), in bytes.
    pub fn peak_rss_bytes(&self) -> usize {
        self.memory.peak_bytes
    }

    /// Render a compact, human-readable multi-line summary.
    pub fn display(&self) -> String {
        format!(
            "real_e2e: {}\n\
             \x20 file size     = {}\n\
             \x20 load wall     = {:.1} ms\n\
             \x20 context size  = {}\n\
             \x20 prefill       = {} tokens in {:.1} ms  ({:.2} tok/s)\n\
             \x20 decode        = {} tokens in {:.1} ms  ({:.2} tok/s)\n\
             \x20 {}",
            self.model_path,
            format_size(self.model_file_bytes),
            self.load_wall_ms,
            self.effective_context_size,
            self.prompt_tokens,
            self.prefill_ms,
            self.prefill_tokens_per_sec,
            self.decode_tokens,
            self.decode_ms,
            self.decode_tokens_per_sec,
            self.memory.display(),
        )
    }
}

/// Run the real end-to-end benchmark against the model named by the
/// [`MODEL_ENV_VAR`] environment variable.
///
/// Returns `Ok(None)` — never an error — when the variable is unset. This is
/// the "skip gracefully" contract: this crate, its tests, and CI must all
/// build and run without a multi-gigabyte model file on disk.
pub fn run_real_e2e_bench(config: &RealE2eConfig) -> RuntimeResult<Option<RealE2eReport>> {
    run_real_e2e_bench_from_env(config, std::env::var_os(MODEL_ENV_VAR))
}

/// [`run_real_e2e_bench`] with the environment lookup injected, so the
/// "unset → skip" branch is unit-testable without mutating real process
/// environment state (`std::env::set_var`/`remove_var` are process-global
/// and racy under any test runner that shares a process across tests).
fn run_real_e2e_bench_from_env(
    config: &RealE2eConfig,
    model_path_var: Option<OsString>,
) -> RuntimeResult<Option<RealE2eReport>> {
    let Some(model_path) = model_path_var else {
        eprintln!(
            "[oxillama-bench] {MODEL_ENV_VAR} is not set — skipping the real end-to-end \
             benchmark. Set it to a GGUF model path to measure load time, prefill/decode \
             tok/s, and peak RSS against a real model, e.g.:\n  \
             OXILLAMA_BENCH_MODEL=/path/to/model.gguf cargo bench -p oxillama-bench --bench real_e2e"
        );
        return Ok(None);
    };
    let model_path = model_path.to_string_lossy().into_owned();
    run_real_e2e_bench_for(&model_path, config).map(Some)
}

/// Run the benchmark against an explicit model path, bypassing the
/// environment variable. Used by [`run_real_e2e_bench`] and directly by
/// callers (including tests) that already have a model path in hand.
///
/// # Why decode ignores EOS
///
/// [`InferenceEngine::generate_detailed`] is the right choice for real
/// serving: it stops as soon as the model produces an end-of-generation
/// token. It is the wrong choice for a benchmark number that has to be
/// comparable release over release — a prompt that happens to make the model
/// stop after 9 tokens this week and 40 tokens next week (because a
/// quantization change shifted a logit) would silently change "decode
/// tok/s" into "how chatty did the model feel today", which is not what
/// Mission A asks for. So this function drives [`InferenceEngine::forward_prefill`]
/// and [`InferenceEngine::forward_decode`] directly with a plain greedy
/// (argmax) selector, running the full [`DECODE_TOKENS`] steps every time.
///
/// # Errors
///
/// Propagates [`oxillama_runtime::RuntimeError`] from model loading,
/// tokenization, or the forward pass (e.g. `ModelLoadError` if `model_path`
/// does not exist, or an unsupported-architecture error if this crate was
/// not built with the matching `oxillama-runtime` feature).
pub fn run_real_e2e_bench_for(
    model_path: &str,
    config: &RealE2eConfig,
) -> RuntimeResult<RealE2eReport> {
    let model_file_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let mut profiler = MemoryProfiler::new();
    profiler.record(); // baseline, before the engine exists

    let engine_config = EngineConfig {
        model_path: model_path.to_string(),
        context_size: config.context_size,
        ..EngineConfig::default()
    };
    let mut engine = InferenceEngine::new(engine_config);

    let load_start = Instant::now();
    engine.load_model()?;
    let load_wall_ms = duration_ms(load_start.elapsed());
    profiler.record();

    let effective_context_size = engine
        .model_config()
        .map(|c| c.max_context_length)
        .unwrap_or(0);

    let prompt_tokens = engine.tokenize(&config.prompt)?;
    profiler.record();

    // `reset()` is a no-op on a freshly loaded engine (load_model() always
    // starts with an empty KV cache) but keeps the precondition documented
    // on `forward_prefill` ("the KV cache is not reset by this call") true
    // by construction rather than by accident, and protects any future
    // caller of this function that reuses one `InferenceEngine` for more
    // than one report.
    engine.reset();
    let prefill_start = Instant::now();
    let mut logits = engine.forward_prefill(&prompt_tokens, 0)?;
    let prefill_ms = duration_ms(prefill_start.elapsed());
    profiler.record();

    let prefill_tokens_per_sec = if prefill_ms > 0.0 {
        prompt_tokens.len() as f64 / prefill_ms * 1000.0
    } else {
        0.0
    };

    let mut decode_elapsed = Duration::ZERO;
    let decode_start_pos = prompt_tokens.len();
    for pos in decode_start_pos..(decode_start_pos + DECODE_TOKENS) {
        let next_token = argmax(&logits);
        let step_start = Instant::now();
        logits = engine.forward_decode(next_token, pos)?;
        decode_elapsed += step_start.elapsed();
        profiler.record();
    }
    let decode_ms = duration_ms(decode_elapsed);
    let decode_tokens_per_sec = if decode_ms > 0.0 {
        DECODE_TOKENS as f64 / decode_ms * 1000.0
    } else {
        0.0
    };

    Ok(RealE2eReport {
        model_path: model_path.to_string(),
        model_file_bytes,
        load_wall_ms,
        effective_context_size,
        prompt_tokens: prompt_tokens.len(),
        prefill_ms,
        prefill_tokens_per_sec,
        decode_tokens: DECODE_TOKENS,
        decode_ms,
        decode_tokens_per_sec,
        memory: profiler.report(),
    })
}

/// Greedy (argmax) token selection over raw logits.
///
/// Returns `0` for an empty slice rather than unwrapping — this should never
/// happen once a model is loaded (`forward_prefill`/`forward_decode` always
/// return one logit per vocabulary entry), but the no-`unwrap()` policy
/// applies to benchmark code too. `NaN` entries are never selected (`NaN >
/// best_val` is always `false`), so a corrupted logit degrades to "skipped"
/// rather than panicking or propagating.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0usize;
    while value >= 1024.0 && unit_idx + 1 < UNITS.len() {
        value /= 1024.0;
        unit_idx += 1;
    }
    format!("{value:.2} {}", UNITS[unit_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest_logit() {
        assert_eq!(argmax(&[0.1, 5.0, -3.0, 4.9]), 1);
    }

    #[test]
    fn argmax_first_on_tie() {
        assert_eq!(argmax(&[2.0, 2.0, 1.0]), 0);
    }

    #[test]
    fn argmax_skips_nan_rather_than_panicking() {
        assert_eq!(argmax(&[f32::NAN, 1.0, f32::NAN]), 1);
    }

    #[test]
    fn argmax_empty_returns_zero_not_a_panic() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn format_size_scales_units() {
        assert_eq!(format_size(512), "512.00 B");
        assert_eq!(format_size(1024), "1.00 KiB");
        assert!(format_size(5_000_000_000).contains("GiB"));
    }

    #[test]
    fn skips_gracefully_when_env_var_unset() {
        let config = RealE2eConfig::default();
        let result = run_real_e2e_bench_from_env(&config, None)
            .expect("must return Ok(None), never an Err, when the env var is unset");
        assert!(
            result.is_none(),
            "must skip (Ok(None)) rather than fabricate a report when unset"
        );
    }

    #[test]
    fn runs_against_a_real_synthetic_gguf_file_on_disk() {
        // Not the huge real-model path — a tiny, fast, fully in-memory-built
        // GGUF written to a temp file so this test exercises the *real*
        // `InferenceEngine::load_model()` file path (mmap, tokenizer
        // resolution, forward pass construction) without needing a
        // multi-gigabyte model on disk.
        //
        // `build_minimal_llama_gguf()` sets `tokenizer.ggml.model` but does
        // not embed a `tokenizer.ggml.tokens` array (every other consumer in
        // this workspace pairs it with `load_model_from_bytes(bytes, json)`,
        // which takes the tokenizer JSON directly and never touches disk).
        // Driving the real file-based `load_model()` therefore needs the
        // sidecar `tokenizer.json` resolution path — same code a real model
        // shipped without an embedded vocabulary would exercise — so both
        // files are written into a unique per-process temp directory here.
        // Vocabulary is lowercase-ASCII-only (see
        // `oxillama_gguf::test_utils::minimal_tokenizer_json`), so the
        // prompt is restricted to characters that vocabulary actually has.
        let bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();

        let dir = std::env::temp_dir().join(format!(
            "oxillama_bench_real_e2e_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create unique temp dir for synthetic model");
        let model_path = dir.join("model.gguf");
        let tokenizer_path = dir.join("tokenizer.json");
        std::fs::write(&model_path, &bytes).expect("write synthetic GGUF to temp file");
        std::fs::write(&tokenizer_path, tokenizer_json).expect("write sidecar tokenizer.json");

        let config = RealE2eConfig {
            prompt: "the cat sat on the mat".to_string(),
            context_size: None,
        };
        let result = run_real_e2e_bench_for(&model_path.to_string_lossy(), &config);

        std::fs::remove_dir_all(&dir).ok();

        let report = result.expect("real_e2e bench must succeed against a valid synthetic GGUF");

        assert!(report.load_wall_ms >= 0.0);
        assert!(
            report.prompt_tokens > 0,
            "prompt must tokenize to >= 1 token"
        );
        assert_eq!(
            report.decode_tokens, DECODE_TOKENS,
            "decode phase must always run the full fixed length, ignoring EOS"
        );
        assert!(report.prefill_ms >= 0.0);
        assert!(report.decode_ms >= 0.0);
        assert_eq!(
            report.effective_context_size, 128,
            "synthetic model's llama.context_length"
        );
        assert!(
            report.model_file_bytes > 0,
            "model_file_bytes must reflect the real file size on disk"
        );
        // Best-effort on every platform: `MemoryProfiler` returns `0` when
        // RSS reading is unsupported (see `memory.rs`), so only assert peak
        // >= baseline, never a hard nonzero requirement.
        assert!(report.peak_rss_bytes() >= report.memory.baseline_bytes);
    }

    #[test]
    fn nonexistent_model_path_returns_err_not_a_panic() {
        let config = RealE2eConfig::default();
        let missing = std::env::temp_dir().join(format!(
            "oxillama_bench_real_e2e_missing_{}_{}.gguf",
            std::process::id(),
            "does-not-exist"
        ));
        let result = run_real_e2e_bench_for(&missing.to_string_lossy(), &config);
        assert!(result.is_err(), "a missing model file must be a clean Err");
    }

    #[test]
    fn default_config_has_the_mission_fixed_decode_length() {
        // Not a config field on `RealE2eConfig` on purpose — DECODE_TOKENS is
        // the mission-fixed 64-token greedy generation length.
        assert_eq!(DECODE_TOKENS, 64);
        let config = RealE2eConfig::default();
        assert!(!config.prompt.is_empty());
        assert!(config.context_size.is_none());
    }
}
