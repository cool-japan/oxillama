//! Benchmark loop for the `bench` subcommand (feature = `bench`).
//!
//! Extracted from `main.rs` so it can be exercised by a fast, offline unit
//! test against a tiny synthetic model instead of only by hand against a
//! multi-gigabyte GGUF — see the regression tests at the bottom of this file
//! for the C3 fix: the KV cache growing unbounded across iterations (each
//! iteration used to prefill on top of every prior iteration's prompt and
//! output), and tokens/s computed from a whitespace word count instead of
//! the real generated-token count (understates by roughly 1.3x for typical
//! BPE text, and is flatly wrong for vocabularies without word-aligned
//! tokens).

use oxillama_runtime::{GenerationConfig, InferenceEngine, RuntimeResult};

/// Aggregate result of a [`run_benchmark`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchResult {
    /// Number of measured iterations run.
    pub iterations: usize,
    /// Sum of `GenerationOutcome::completion_tokens()` across all measured
    /// iterations — a real generated-token count, never a word count.
    pub total_completion_tokens: usize,
    /// Wall-clock time spent in the measured iterations, in seconds.
    pub elapsed_secs: f64,
}

impl BenchResult {
    /// Generated tokens per second across all measured iterations.
    pub fn tokens_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total_completion_tokens as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

/// Run `warmup` untimed iterations followed by `iterations` timed ones,
/// generating up to `n_predict` tokens from `prompt` each time.
///
/// `engine.reset()` is called before **every** iteration, warmup and
/// measured alike: without it, iteration N prefills into a KV cache that
/// still holds every prior iteration's prompt and output, so later
/// iterations attend over a monotonically longer context — progressively
/// slower for reasons that have nothing to do with steady-state throughput —
/// and once the context limit is reached the decode loop stops early,
/// producing near-zero tokens while still consuming wall time.
///
/// # Errors
///
/// Propagates any `RuntimeError` from generation (e.g. no model loaded).
pub fn run_benchmark(
    engine: &mut InferenceEngine,
    prompt: &str,
    warmup: usize,
    iterations: usize,
    n_predict: usize,
) -> RuntimeResult<BenchResult> {
    // Sample with the engine's own configured sampler, not a fresh
    // `SamplerConfig::default()`: the latter draws an unseeded,
    // unpredictable seed on every call (`sampling::rng::generate_seed`,
    // time-based), so back-to-back runs on the same seeded engine would
    // report different token counts for reasons that have nothing to do
    // with throughput. This also matches `run`/`serve`, which both
    // generate with the engine's configured sampler rather than a
    // throwaway default.
    let gen_config = GenerationConfig::new(n_predict).with_sampler(engine.config().sampler.clone());

    for _ in 0..warmup {
        engine.reset();
        engine.generate_detailed(prompt, &gen_config, |_| {})?;
    }

    let mut total_completion_tokens = 0usize;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        engine.reset();
        let outcome = engine.generate_detailed(prompt, &gen_config, |_| {})?;
        total_completion_tokens += outcome.completion_tokens();
    }
    let elapsed_secs = start.elapsed().as_secs_f64();

    Ok(BenchResult {
        iterations,
        total_completion_tokens,
        elapsed_secs,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny, fully in-memory `InferenceEngine` for fast, offline
    /// tests — no filesystem access, no multi-gigabyte model download.
    fn tiny_engine(ctx_size: usize) -> InferenceEngine {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();

        let config = oxillama_runtime::EngineConfig {
            model_path: "in-memory-test-model".to_string(),
            tokenizer_path: None,
            context_size: Some(ctx_size),
            sampler: oxillama_runtime::SamplerConfig {
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = InferenceEngine::new(config);
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("tiny synthetic model should load");
        engine
    }

    #[test]
    fn resets_kv_cache_between_every_iteration() {
        // Before the fix, the benchmark loop called `generate()` with no
        // `engine.reset()` in between iterations, so the KV cache
        // accumulated every prior iteration's prompt + output. This asserts
        // the cache after N `run_benchmark` iterations reflects only the
        // LAST iteration, never their sum.
        let mut engine = tiny_engine(128);
        let prompt = "hello";
        let n_predict = 5;

        let prompt_tokens = engine.tokenize(prompt).expect("tokenize prompt");

        let result =
            run_benchmark(&mut engine, prompt, 0, 3, n_predict).expect("benchmark should succeed");
        assert_eq!(result.iterations, 3);

        let seq_len = engine.kv_cache_seq_len();
        let max_plausible = prompt_tokens.len() + n_predict;
        assert!(
            seq_len <= max_plausible,
            "KV cache seq_len ({seq_len}) must not exceed one iteration's worth \
             ({max_plausible}) — a growing cache means iterations are not being reset"
        );
    }

    #[test]
    fn without_reset_kv_cache_grows_unbounded_across_iterations() {
        // Companion test proving the *old* behavior really was a bug:
        // reproduces the pre-fix `Commands::Bench` loop (no `reset()`
        // between calls) and shows the KV cache strictly grows.
        let mut engine = tiny_engine(128);
        let prompt = "hello";
        let gen_config = GenerationConfig::new(5);

        engine
            .generate_detailed(prompt, &gen_config, |_| {})
            .expect("first generation should succeed");
        let after_one = engine.kv_cache_seq_len();

        engine
            .generate_detailed(prompt, &gen_config, |_| {})
            .expect("second generation should succeed");
        engine
            .generate_detailed(prompt, &gen_config, |_| {})
            .expect("third generation should succeed");
        let after_three = engine.kv_cache_seq_len();

        assert!(
            after_three > after_one,
            "reproduction of the pre-fix bug: KV cache should grow across \
             unreset iterations ({after_one} -> {after_three})"
        );
    }

    #[test]
    fn total_completion_tokens_uses_real_token_count() {
        // The old code computed `total_tokens += result.split_whitespace().count()`.
        // Assert `run_benchmark`'s total is the sum of each outcome's real
        // `completion_tokens()`, and sanity-check it against a text produced
        // by an *independently seeded* engine (same seed, fresh RNG state —
        // `engine.reset()` only clears the KV cache, not the sampler's RNG,
        // so reusing one engine for two back-to-back generations would give
        // each call a different RNG state and make them incomparable) so the
        // fix cannot silently regress back to counting words.
        let n_predict = 8;
        // Unlike `run_benchmark` (which now builds its `GenerationConfig`
        // from `engine.config().sampler`), calling `generate_detailed`
        // directly here does NOT pick up `tiny_engine`'s seed 42
        // automatically — `GenerationConfig::new` alone defaults to
        // `SamplerConfig::default()` (seed: None, i.e. a fresh, unseeded
        // RNG draw per call). Matching that seed explicitly is what makes
        // this call comparable to `run_benchmark`'s below: same seed, same
        // fresh `tiny_engine` construction, same prompt.
        let fixed_sampler = oxillama_runtime::SamplerConfig {
            seed: Some(42),
            ..Default::default()
        };
        let gen_config = GenerationConfig::new(n_predict).with_sampler(fixed_sampler.clone());

        let mut probe_engine = tiny_engine(128);
        let mut generated_text = String::new();
        let text_outcome = probe_engine
            .generate_detailed("hello", &gen_config, |chunk| generated_text.push_str(chunk))
            .expect("generation should succeed");
        assert_eq!(
            text_outcome.completion_tokens(),
            text_outcome.generated_tokens.len(),
            "completion_tokens() must equal the real per-token count"
        );
        let naive_word_count = generated_text.split_whitespace().count();

        let mut engine = tiny_engine(128);
        let result =
            run_benchmark(&mut engine, "hello", 0, 1, n_predict).expect("benchmark should succeed");
        assert_eq!(
            result.total_completion_tokens,
            text_outcome.completion_tokens(),
            "run_benchmark's total must equal completion_tokens() from an identically \
             seeded, freshly constructed engine, regardless of how many \
             whitespace-separated words the text contains \
             (naive word count for this run: {naive_word_count})"
        );
    }
}
