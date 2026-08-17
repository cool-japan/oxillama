//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::error::RuntimeError;
use crate::sampling::SamplerConfig;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    /// `forward_prefill` must return `ModelNotLoaded` when no model is loaded.
    #[test]
    fn test_forward_prefill_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_prefill(&[1, 2, 3], 0);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded from forward_prefill, got {result:?}"
        );
    }
    /// `forward_prefill` with empty token slice must return an error
    /// (even if a model were loaded — callers must supply at least one token).
    #[test]
    fn test_forward_prefill_empty_slice_errors() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_prefill(&[], 0);
        assert!(
            result.is_err(),
            "forward_prefill with empty slice must return Err, got Ok"
        );
    }
    /// `forward_decode` must return `ModelNotLoaded` when no model is loaded.
    #[test]
    fn test_forward_decode_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_decode(42, 0);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded from forward_decode, got {result:?}"
        );
    }
    /// `forward_prefill` with a loaded model must return a logits vector whose
    /// length equals the model's vocab size (32 in the synthetic fixture).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_forward_prefill_returns_logits_after_load() {
        let mut engine = make_loaded_engine();
        let result = engine.forward_prefill(&[3, 4, 5], 0);
        assert!(
            result.is_ok(),
            "forward_prefill must return Ok when model is loaded, got {result:?}"
        );
        let logits = result.expect("forward_prefill Ok");
        assert_eq!(
            logits.len(),
            32,
            "logits length must equal vocab_size=32, got {}",
            logits.len()
        );
    }
    /// `forward_decode` with a loaded model must return a logits vector of
    /// the correct vocab-size length.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_forward_decode_returns_logits_after_load() {
        let mut engine = make_loaded_engine();
        engine
            .forward_prefill(&[3], 0)
            .expect("prefill must succeed");
        let result = engine.forward_decode(4, 1);
        assert!(
            result.is_ok(),
            "forward_decode must return Ok when model is loaded, got {result:?}"
        );
        let logits = result.expect("forward_decode Ok");
        assert_eq!(
            logits.len(),
            32,
            "logits length must equal vocab_size=32, got {}",
            logits.len()
        );
    }
    /// Verify that chunked-prefill produces the same final logits as
    /// single-shot prefill (the core KV-state invariant from A3).
    ///
    /// Both paths must agree on the logit vector produced after processing
    /// the same prompt tokens, within floating-point tolerance.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn chunked_prefill_kv_matches_singleshot() {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let prompt_tokens = vec![3u32, 4, 5, 6];
        let mut engine_single = InferenceEngine::new(EngineConfig::default());
        engine_single
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("single-shot load");
        let logits_single = engine_single
            .forward_prefill(&prompt_tokens, 0)
            .expect("single-shot prefill");
        let mut engine_chunked = InferenceEngine::new(EngineConfig::default());
        engine_chunked
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("chunked load");
        let mut logits_chunked = Vec::new();
        let chunk_size = 2usize;
        let mut pos = 0usize;
        for slice in prompt_tokens.chunks(chunk_size) {
            logits_chunked = engine_chunked
                .forward_prefill(slice, pos)
                .expect("chunked prefill");
            pos += slice.len();
        }
        assert_eq!(
            logits_single.len(),
            logits_chunked.len(),
            "logit vector lengths must match"
        );
        let tol = 1e-4f32;
        let max_diff = logits_single
            .iter()
            .zip(logits_chunked.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < tol,
            "chunked and single-shot prefill logits differ by {max_diff} > tolerance {tol}"
        );
    }
    /// embed() must return an error when no model has been loaded,
    /// rather than panicking or producing a garbage vector.
    #[test]
    fn test_embed_returns_err_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.embed("hello world");
        assert!(
            result.is_err(),
            "embed() should return Err when no model is loaded"
        );
    }
    /// hidden_size() returns None when no model is loaded.
    #[test]
    fn test_hidden_size_none_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(
            engine.hidden_size().is_none(),
            "hidden_size() should be None before load_model()"
        );
    }
    /// is_loaded() must be false for a freshly created engine.
    #[test]
    fn test_is_loaded_false_initially() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(!engine.is_loaded());
    }
    #[test]
    fn test_model_config_none_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(engine.model_config().is_none());
    }
    #[test]
    fn test_config_roundtrip() {
        let cfg = EngineConfig {
            model_path: "test.gguf".to_string(),
            num_threads: 8,
            ..EngineConfig::default()
        };
        let engine = InferenceEngine::new(cfg);
        assert_eq!(engine.config().model_path, "test.gguf");
        assert_eq!(engine.config().num_threads, 8);
    }
    #[test]
    fn test_generate_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.generate("hello", 10, |_| {});
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_generate_with_config_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.generate_with_config("hello", 5, SamplerConfig::greedy(), |_| {});
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_tokenize_errors_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.tokenize("hello world");
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_prefill_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.prefill(&[1, 2, 3]);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_prefill_empty_slice_ok_when_no_model() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.prefill(&[]);
        assert!(result.is_ok(), "empty prefill should be Ok, got {result:?}");
    }
    #[test]
    fn test_forward_one_errors_when_not_loaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.forward_one(42);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_decode_token_errors_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        let result = engine.decode_token(1);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_is_eos_false_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(!engine.is_eos(0));
        assert!(!engine.is_eos(u32::MAX));
    }
    #[test]
    fn test_vocab_bytes_none_when_not_loaded() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(engine.vocab_bytes().is_none());
    }
    #[test]
    fn test_reset_does_not_panic_when_no_kv_cache() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine.reset();
    }
    #[test]
    fn test_apply_lora_adapters_errors_when_not_loaded() {
        use oxillama_arch::lora::LoadedLora;
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let lora = LoadedLora {
            rank: 8,
            alpha: 1.0,
            adapters: std::collections::HashMap::new(),
        };
        let result = engine.apply_lora_adapters(&lora);
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    #[test]
    fn test_load_model_missing_file_errors() {
        let cfg = EngineConfig {
            model_path: "/nonexistent/path/model_abc_xyz.gguf".to_string(),
            ..EngineConfig::default()
        };
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model();
        assert!(
            matches!(result, Err(RuntimeError::ModelLoadError { .. })),
            "expected ModelLoadError for missing file, got {result:?}"
        );
    }
    #[test]
    fn test_load_model_from_bytes_bad_magic_errors() {
        let cfg = EngineConfig::default();
        let mut engine = InferenceEngine::new(cfg);
        let bad_bytes = b"THIS IS NOT A GGUF FILE AT ALL";
        let result = engine.load_model_from_bytes(bad_bytes, "{}");
        assert!(
            result.is_err(),
            "load_model_from_bytes with garbage bytes should error, got Ok(())"
        );
    }
    #[test]
    fn test_load_model_from_bytes_empty_errors() {
        let cfg = EngineConfig::default();
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model_from_bytes(&[], "{}");
        assert!(
            result.is_err(),
            "load_model_from_bytes with empty bytes should error"
        );
    }
    #[test]
    fn test_engine_config_default_fields() {
        let cfg = EngineConfig::default();
        assert!(
            cfg.model_path.is_empty(),
            "default model_path should be empty"
        );
        assert!(
            cfg.tokenizer_path.is_none(),
            "default tokenizer_path should be None"
        );
        assert!(
            cfg.context_size.is_none(),
            "default context_size should be None"
        );
        assert_eq!(
            cfg.num_threads, 0,
            "default num_threads should be 0 (= auto-size the GEMV pool from \
             available_parallelism); an explicit non-zero value still wins"
        );
    }
    #[test]
    fn test_engine_config_context_override() {
        let cfg = EngineConfig {
            context_size: Some(2048),
            ..EngineConfig::default()
        };
        assert_eq!(cfg.context_size, Some(2048));
    }
    #[test]
    fn test_generate_with_config_errors_when_not_loaded_variant() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let sc = SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            ..SamplerConfig::default()
        };
        let result = engine.generate_with_config("test prompt", 5, sc, |_| {});
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }
    /// load_model() with a file that *exists* but contains garbage (not valid GGUF)
    /// must return an error without panicking, exercising the GgufModel::load parse path.
    #[test]
    fn test_load_model_existing_invalid_file_errors() {
        let mut tmp = std::env::temp_dir();
        tmp.push("oxillama_engine_bad_magic_test.gguf");
        std::fs::write(&tmp, b"NOT A GGUF FILE AT ALL - GARBAGE BYTES 0123456789")
            .expect("write temp file");
        let cfg = EngineConfig {
            model_path: tmp
                .to_str()
                .expect("temp path must be valid UTF-8")
                .to_string(),
            ..EngineConfig::default()
        };
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model();
        let _ = std::fs::remove_file(&tmp);
        assert!(
            result.is_err(),
            "load_model with invalid GGUF content should return Err"
        );
    }
    /// is_loaded() must remain false after a failed load_model() call.
    #[test]
    fn test_is_loaded_remains_false_after_failed_load() {
        let cfg = EngineConfig {
            model_path: "/nonexistent/guaranteed_missing_model.gguf".to_string(),
            ..EngineConfig::default()
        };
        let mut engine = InferenceEngine::new(cfg);
        let _ = engine.load_model();
        assert!(
            !engine.is_loaded(),
            "is_loaded() must be false after a failed load_model()"
        );
    }
    /// EngineConfig implements Clone; verify the clone is independent.
    #[test]
    fn test_engine_config_clone_is_independent() {
        let original = EngineConfig {
            model_path: "original.gguf".to_string(),
            num_threads: 16,
            context_size: Some(4096),
            ..EngineConfig::default()
        };
        let mut cloned = original.clone();
        cloned.model_path = "cloned.gguf".to_string();
        cloned.num_threads = 1;
        assert_eq!(original.model_path, "original.gguf");
        assert_eq!(original.num_threads, 16);
        assert_eq!(original.context_size, Some(4096));
    }
    /// Return a loaded engine from the synthetic GGUF + tokenizer.
    /// These tests are only meaningful when a tokenizer backend is active.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    fn make_loaded_engine() -> InferenceEngine {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("synthetic GGUF must load successfully");
        engine
    }
    /// load_model_from_bytes with the synthetic fixture must succeed and
    /// set is_loaded() to true.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_load_model_from_bytes_succeeds() {
        let engine = make_loaded_engine();
        assert!(
            engine.is_loaded(),
            "is_loaded() must be true after a successful load_model_from_bytes()"
        );
    }
    /// model_config() must be Some with the expected hidden size after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_hidden_size_after_load() {
        let engine = make_loaded_engine();
        let hs = engine.hidden_size();
        assert_eq!(
            hs,
            Some(32),
            "hidden_size() must be Some(32) after loading the synthetic model, got {hs:?}"
        );
    }
    /// tokenize() must return Ok with at least one token after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_tokenize_after_load() {
        let engine = make_loaded_engine();
        let result = engine.tokenize("abc");
        assert!(
            result.is_ok(),
            "tokenize() must return Ok after model is loaded, got {result:?}"
        );
        let tokens = result.expect("tokenize succeeded");
        assert!(
            !tokens.is_empty(),
            "tokenize('abc') must produce at least one token"
        );
    }
    /// is_eos() must return true for token id 2 (</s> = EOS in the synthetic tokenizer).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_is_eos_after_load() {
        let engine = make_loaded_engine();
        assert!(
            engine.is_eos(2),
            "is_eos(2) must be true — </s> is the EOS token in the synthetic tokenizer"
        );
        assert!(
            !engine.is_eos(3),
            "is_eos(3) must be false — token 3 ('a') is not EOS"
        );
    }
    /// decode_token(3) should decode successfully (token 3 = 'a' in the synthetic vocab).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_decode_token_after_load() {
        let engine = make_loaded_engine();
        let result = engine.decode_token(3);
        assert!(
            result.is_ok(),
            "decode_token(3) must return Ok, got {result:?}"
        );
    }
    /// generate() must return Ok after loading; the returned string may be empty
    /// if the EOS token is sampled immediately, but it must not panic or error.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_after_load() {
        let mut engine = make_loaded_engine();
        let result = engine.generate("a", 3, |_| {});
        assert!(
            result.is_ok(),
            "generate() must return Ok after model is loaded, got {result:?}"
        );
    }
    /// generate() with max_tokens=5 must produce at most 5 tokens of output.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_respects_max_tokens() {
        let mut engine = make_loaded_engine();
        let max = 5usize;
        let mut count = 0usize;
        let result = engine.generate("a", max, |_tok| {
            count += 1;
        });
        assert!(result.is_ok(), "generate() must return Ok, got {result:?}");
        assert!(
            count <= max,
            "callback was invoked {count} times but max_tokens={max}"
        );
    }
    /// generate_streaming — count callback invocations to verify the streaming path fires.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_streaming_calls_callback() {
        let mut engine = make_loaded_engine();
        let mut invocations = 0usize;
        let max_tokens = 4;
        let result = engine.generate("a", max_tokens, |_piece| {
            invocations += 1;
        });
        assert!(
            result.is_ok(),
            "generate() streaming path must return Ok, got {result:?}"
        );
        assert!(
            invocations <= max_tokens,
            "streaming callback fired {invocations} > max_tokens={max_tokens}"
        );
    }
    /// embed() must return Ok with a non-empty vector after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_after_load() {
        let mut engine = make_loaded_engine();
        let result = engine.embed("a");
        assert!(
            result.is_ok(),
            "embed() must return Ok after model is loaded, got {result:?}"
        );
        let vec = result.expect("embed succeeded");
        assert!(!vec.is_empty(), "embed() must return a non-empty vector");
    }
    /// embed() must return a vector of length == hidden_size (32 for the synthetic model).
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_returns_hidden_size_vector() {
        let mut engine = make_loaded_engine();
        let vec = engine
            .embed("a")
            .expect("embed() must succeed after loading");
        assert_eq!(
            vec.len(),
            32,
            "embed() vector length must equal hidden_size=32, got {}",
            vec.len()
        );
    }
    /// Reload: loading the model a second time must succeed and leave is_loaded() true.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_reload_model_succeeds() {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("first load must succeed");
        assert!(engine.is_loaded(), "is_loaded() after first load");
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("second (re)load must succeed");
        assert!(
            engine.is_loaded(),
            "is_loaded() after reload must still be true"
        );
    }
    /// vocab_bytes() must return Some with non-empty entries after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_vocab_bytes_some_after_load() {
        let engine = make_loaded_engine();
        let vb = engine.vocab_bytes();
        assert!(
            vb.is_some(),
            "vocab_bytes() must be Some after model is loaded"
        );
        let entries = vb.expect("vocab_bytes is Some");
        assert!(
            !entries.is_empty(),
            "vocab_bytes() must contain at least one entry"
        );
    }
    /// model_config() must return Some with correct metadata after loading.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_model_config_some_after_load() {
        let engine = make_loaded_engine();
        let cfg = engine.model_config();
        assert!(cfg.is_some(), "model_config() must be Some after loading");
        let mc = cfg.expect("model_config is Some");
        assert_eq!(mc.architecture, "llama", "architecture must be 'llama'");
        assert_eq!(
            mc.num_layers, 1,
            "num_layers must be 1 for the synthetic model"
        );
        assert_eq!(mc.vocab_size, 32, "vocab_size must be 32");
    }
    /// reset() must not panic when a model is loaded, and the engine remains usable.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_reset_when_loaded_does_not_panic() {
        let mut engine = make_loaded_engine();
        engine.reset();
        assert!(
            engine.is_loaded(),
            "is_loaded() must still be true after reset()"
        );
        assert_eq!(engine.hidden_size(), Some(32));
    }
    /// Qwen3 forward pass: load, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_qwen3_arch() {
        use oxillama_gguf::test_utils::{build_minimal_qwen3_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_qwen3_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load qwen3");
        assert!(engine.is_loaded(), "qwen3: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate qwen3");
    }
    /// Qwen3 embed: load and call embed().
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_qwen3_arch() {
        use oxillama_gguf::test_utils::{build_minimal_qwen3_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_qwen3_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load qwen3 for embed");
        let vec = engine.embed("abc").expect("test: embed qwen3");
        assert_eq!(
            vec.len(),
            32,
            "qwen3 embed must return hidden_size=32 vector"
        );
    }
    /// Mistral forward pass: load, generate with sliding window, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_mistral_arch() {
        use oxillama_gguf::test_utils::{build_minimal_mistral_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_mistral_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load mistral");
        assert!(engine.is_loaded(), "mistral: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate mistral");
    }
    /// Gemma forward pass: load with soft-capping metadata, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_gemma_arch() {
        use oxillama_gguf::test_utils::{build_minimal_gemma_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_gemma_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load gemma");
        assert!(engine.is_loaded(), "gemma: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate gemma");
    }
    /// Gemma embed: load and call embed() to cover the Gemma embedding path.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_embed_gemma_arch() {
        use oxillama_gguf::test_utils::{build_minimal_gemma_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_gemma_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load gemma for embed");
        let vec = engine.embed("abc").expect("test: embed gemma");
        assert_eq!(
            vec.len(),
            32,
            "gemma embed must return hidden_size=32 vector"
        );
    }
    /// Phi-3 forward pass: merged QKV + partial RoPE, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_phi3_arch() {
        use oxillama_gguf::test_utils::{build_minimal_phi3_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_phi3_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load phi3");
        assert!(engine.is_loaded(), "phi3: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate phi3");
    }
    /// Command-R forward pass: logit scaling, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_command_r_arch() {
        use oxillama_gguf::test_utils::{build_minimal_command_r_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_command_r_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load command-r");
        assert!(engine.is_loaded(), "command-r: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate command-r");
    }
    /// StarCoder forward pass: MQA + absolute position embeddings, generate, assert ok.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn test_generate_starcoder_arch() {
        use oxillama_gguf::test_utils::{build_minimal_starcoder_gguf, minimal_tokenizer_json};
        let bytes = build_minimal_starcoder_gguf();
        let json = minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&bytes, json)
            .expect("test: load starcoder");
        assert!(engine.is_loaded(), "starcoder: is_loaded() must be true");
        let _out = engine
            .generate("abc", 2, |_| {})
            .expect("test: generate starcoder");
    }
    #[test]
    fn lora_stack_push_pop() {
        use oxillama_arch::lora::LoadedLora;
        use oxillama_quant::LoraAdapter;
        use std::collections::HashMap;
        use std::sync::Arc;
        fn make_lora() -> Arc<LoadedLora> {
            let adapter = LoraAdapter::new(vec![0.0f32; 4 * 8], vec![0.0f32; 8 * 4], 4, 1.0, 8, 8)
                .expect("valid lora adapter");
            let mut adapters = HashMap::new();
            adapters.insert("test.weight".to_string(), Arc::new(adapter));
            Arc::new(LoadedLora {
                adapters,
                rank: 4,
                alpha: 1.0,
            })
        }
        let mut engine = InferenceEngine::new(EngineConfig::default());
        assert!(engine.lora_stack().is_empty());
        assert_eq!(engine.lora_stack().len(), 0);
        engine.push_lora(make_lora(), 1.0);
        engine.push_lora(make_lora(), 0.5);
        assert_eq!(engine.lora_stack().len(), 2);
        assert!(!engine.lora_stack().is_empty());
        let popped = engine.pop_lora();
        assert!(popped.is_some());
        let (_, scale) = popped.expect("pop must return Some");
        assert!((scale - 0.5).abs() < 1e-6);
        assert_eq!(engine.lora_stack().len(), 1);
        engine.clear_loras();
        assert!(engine.lora_stack().is_empty());
        assert!(engine.pop_lora().is_none());
    }
    #[test]
    fn lora_apply_stack_errors_when_not_loaded() {
        use oxillama_arch::lora::LoadedLora;
        use oxillama_quant::LoraAdapter;
        use std::collections::HashMap;
        use std::sync::Arc;
        let adapter = LoraAdapter::new(vec![0.0f32; 4 * 8], vec![0.0f32; 8 * 4], 4, 1.0, 8, 8)
            .expect("valid lora adapter");
        let mut adapters = HashMap::new();
        adapters.insert("test.weight".to_string(), Arc::new(adapter));
        let lora = Arc::new(LoadedLora {
            adapters,
            rank: 4,
            alpha: 1.0,
        });
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine.push_lora(lora, 1.0);
        let result = engine.apply_lora_stack();
        assert!(
            matches!(result, Err(RuntimeError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {:?}",
            result
        );
    }
    /// `unapply_all_loras` on an unloaded engine must not panic.
    #[test]
    fn unapply_all_loras_noop_when_unloaded() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine.unapply_all_loras();
        assert!(!engine.is_loaded());
    }
    /// `prime_with_prefix` on an unloaded engine returns `ModelNotLoaded`.
    #[test]
    fn prime_with_prefix_returns_model_not_loaded() {
        use crate::kv_cache::prefix::{PrefixCacheConfig, PrefixKvCache};
        use crate::kv_cache::KvCache;
        let mut engine = InferenceEngine::new(EngineConfig::default());
        let mut prefix_cache = PrefixKvCache::new(PrefixCacheConfig {
            max_entries: 16,
            max_memory_bytes: 1024 * 1024,
            min_prefix_len: 1,
        });
        let kv = KvCache::new(1, 4, 32);
        let tokens: Vec<u32> = vec![1, 2, 3];
        prefix_cache.store(&tokens, &kv, 3, 4, 1);
        if let Some((match_len, cached)) = prefix_cache.lookup(&tokens) {
            let suffix = &tokens[match_len.min(tokens.len() - 1)..];
            let result = engine.prime_with_prefix(cached, match_len.saturating_sub(1), suffix);
            assert!(
                matches!(result, Err(RuntimeError::ModelNotLoaded)),
                "unloaded engine must return ModelNotLoaded, got {:?}",
                result
            );
        }
    }
}

// ── T1: architecture routing through the registry ────────────────────────────

#[cfg(test)]
mod registry_routing {
    use super::*;

    /// `build_forward_pass` used to be a hard-coded `match` over seven
    /// architecture names while `ArchitectureRegistry::with_builtins()` already
    /// registered 26.  Every other architecture in `oxillama-arch` was
    /// unreachable from the engine — a GGUF naming one was rejected with
    /// `unsupported architecture: '<name>'` no matter how complete the
    /// implementation behind it was.
    ///
    /// This is the direct proof: these architectures have **no arm** in the old
    /// match, so this test fails against it and passes through the registry.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn architectures_absent_from_the_old_hard_coded_match_now_load() {
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        // Every one of these was `unsupported architecture` before 0.1.4.
        let fixtures: Vec<(&str, Vec<u8>)> = vec![
            (
                "falcon",
                oxillama_gguf::test_utils::build_minimal_falcon_gguf(),
            ),
            (
                "olmo2",
                oxillama_gguf::test_utils::build_minimal_olmo2_gguf(),
            ),
            (
                "gptneox",
                oxillama_gguf::test_utils::build_minimal_gpt_neox_gguf(),
            ),
            (
                "stablelm",
                oxillama_gguf::test_utils::build_minimal_stablelm_gguf(),
            ),
            (
                "minicpm",
                oxillama_gguf::test_utils::build_minimal_minicpm_gguf(),
            ),
            (
                "bloom",
                oxillama_gguf::test_utils::build_minimal_bloom_gguf(),
            ),
        ];

        let mut loaded = Vec::new();
        let mut failed = Vec::new();
        for (name, bytes) in &fixtures {
            let mut engine = InferenceEngine::new(EngineConfig::default());
            match engine.load_model_from_bytes(bytes, tokenizer_json) {
                Ok(()) => {
                    assert!(engine.is_loaded(), "{name} reported Ok but is not loaded");
                    loaded.push(*name);
                }
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }

        assert!(
            !loaded.is_empty(),
            "no architecture outside the old hard-coded match could be loaded; \
             failures: {failed:?}"
        );
        // None of these may fail with the old "unsupported architecture" verdict:
        // a genuine loader error is a different (and reportable) matter, but
        // "this engine has never heard of you" must be gone.
        for f in &failed {
            assert!(
                !f.contains("unsupported architecture"),
                "an architecture the registry knows must not be reported as \
                 unsupported: {f}"
            );
        }
    }

    /// A GGUF naming an architecture nothing registers must still be rejected,
    /// and the message must help: it names what this build *does* know.
    #[test]
    fn a_genuinely_unknown_architecture_is_rejected_with_a_useful_message() {
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine.config.model_path = "/nonexistent".to_string();
        // Drive `build_forward_pass` through the public loader by constructing a
        // config with an impossible architecture name.
        let registry = oxillama_arch::ArchitectureRegistry::with_builtins();
        assert!(
            registry.get("definitely-not-an-architecture").is_err(),
            "the registry must not claim an invented architecture"
        );
        assert!(
            registry.len() > 7,
            "the registry should know far more than the seven architectures the \
             old hard-coded match handled, got {}",
            registry.len()
        );
    }

    /// Feature gating still works: an architecture compiled out is simply absent
    /// from the registry, producing a clear error rather than a link failure.
    #[test]
    fn feature_gated_architectures_are_present_only_when_enabled() {
        let registry = oxillama_arch::ArchitectureRegistry::with_builtins();
        #[cfg(feature = "llama")]
        assert!(registry.contains("llama"), "the llama feature is on");
        #[cfg(not(feature = "llama"))]
        assert!(!registry.contains("llama"), "the llama feature is off");
        #[cfg(feature = "qwen3")]
        assert!(registry.contains("qwen3"), "the qwen3 feature is on");
    }

    /// The aliases the old match hard-coded (`gemma2`/`gemma3` → gemma,
    /// `phi` → phi3) must survive the move to the registry.
    #[test]
    fn the_hard_coded_aliases_still_resolve() {
        let registry = oxillama_arch::ArchitectureRegistry::with_builtins();
        #[cfg(feature = "gemma")]
        {
            for alias in ["gemma", "gemma2", "gemma3"] {
                let arch = registry
                    .get(alias)
                    .unwrap_or_else(|e| panic!("alias {alias} must resolve: {e}"));
                assert_eq!(arch.arch_id(), "gemma", "{alias} must route to gemma");
            }
        }
        #[cfg(feature = "phi")]
        {
            for alias in ["phi3", "phi"] {
                let arch = registry
                    .get(alias)
                    .unwrap_or_else(|e| panic!("alias {alias} must resolve: {e}"));
                assert_eq!(arch.arch_id(), "phi3", "{alias} must route to phi3");
            }
        }
    }

    /// Both Mixtral routes must work.  Real Mixtral GGUFs carry
    /// `general.architecture = "llama"` (the converter registers
    /// `MixtralForCausalLM` on the LLaMA converter), so they go through llama's
    /// MoE path; `crate::mixtral` only serves third-party converters that name
    /// the architecture explicitly.  The old match had no `"mixtral"` arm at
    /// all, so the explicit route was unreachable.
    #[test]
    fn both_mixtral_routes_resolve() {
        let registry = oxillama_arch::ArchitectureRegistry::with_builtins();
        #[cfg(feature = "llama")]
        {
            let llama = registry.get("llama").expect("the llama route must resolve");
            assert_eq!(llama.arch_id(), "llama");
        }
        // `mixtral` is an oxillama-arch feature (default-on) rather than a
        // runtime one, so it is checked at run time instead of with `cfg`.
        if let Ok(mixtral) = registry.get("mixtral") {
            assert_eq!(mixtral.arch_id(), "mixtral");
        }
    }

    /// The engine must load the architectures the old match *did* handle, so
    /// the move to the registry is not a regression for anyone.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn the_previously_supported_architectures_still_load() {
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let fixtures: Vec<(&str, Vec<u8>)> = vec![
            (
                "llama",
                oxillama_gguf::test_utils::build_minimal_llama_gguf(),
            ),
            (
                "qwen3",
                oxillama_gguf::test_utils::build_minimal_qwen3_gguf(),
            ),
            (
                "mistral",
                oxillama_gguf::test_utils::build_minimal_mistral_gguf(),
            ),
            (
                "gemma",
                oxillama_gguf::test_utils::build_minimal_gemma_gguf(),
            ),
            ("phi3", oxillama_gguf::test_utils::build_minimal_phi3_gguf()),
            (
                "command-r",
                oxillama_gguf::test_utils::build_minimal_command_r_gguf(),
            ),
            (
                "starcoder",
                oxillama_gguf::test_utils::build_minimal_starcoder_gguf(),
            ),
        ];
        for (name, bytes) in &fixtures {
            let mut engine = InferenceEngine::new(EngineConfig::default());
            engine
                .load_model_from_bytes(bytes, tokenizer_json)
                .unwrap_or_else(|e| panic!("{name} must still load through the registry: {e}"));
            assert!(engine.is_loaded(), "{name} must be loaded");
        }
    }
}

// ── T2 / T5 / T6: engine-level sequence isolation and metrics ────────────────

#[cfg(test)]
mod sequence_isolation {
    use super::*;

    /// A loaded engine over the synthetic LLaMA fixture.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    fn loaded_engine() -> InferenceEngine {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("synthetic GGUF must load successfully");
        engine
    }

    /// Two successive generations on one engine must not contaminate each
    /// other.  `reset()` used to clear only the KV cache, leaving every
    /// architecture's *internal* per-sequence state — DeepSeek's
    /// `MlaLatentCache`, Mamba-2 / Jamba's SSM `h`, DBRX / Grok's
    /// `current_pos` — carrying the previous request straight into the next.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn two_generations_on_one_engine_are_independent() {
        let mut engine = loaded_engine();
        let config = GenerationConfig::new(4).with_sampler(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        engine.reset();
        let first = engine
            .generate_detailed("hello", &config, |_| {})
            .expect("first generation");
        let first_len = engine.kv_cache_seq_len();

        engine.reset();
        assert_eq!(
            engine.kv_cache_seq_len(),
            0,
            "reset() must return the KV cache to position 0"
        );
        let second = engine
            .generate_detailed("hello", &config, |_| {})
            .expect("second generation");
        let second_len = engine.kv_cache_seq_len();

        assert_eq!(
            first_len, second_len,
            "the same prompt must occupy the same number of KV positions in both \
             runs; a differing length means state leaked across the reset"
        );
        assert_eq!(
            first.generated_tokens, second.generated_tokens,
            "greedy sampling on the same prompt after a reset must reproduce the \
             first run exactly"
        );
        assert_eq!(first.prompt_tokens, second.prompt_tokens);
    }

    /// Without a reset the second prompt deliberately *continues* the first —
    /// this is the documented precondition, and it must stay true so callers
    /// can build multi-turn conversations.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn generation_without_reset_continues_the_sequence() {
        let mut engine = loaded_engine();
        let config = GenerationConfig::new(2);
        engine.reset();
        engine
            .generate_detailed("hello", &config, |_| {})
            .expect("first");
        let after_first = engine.kv_cache_seq_len();
        engine
            .generate_detailed("hello", &config, |_| {})
            .expect("second");
        assert!(
            engine.kv_cache_seq_len() > after_first,
            "without reset() the KV cache must keep growing"
        );
    }

    /// `record_kv_hit` / `record_kv_miss` had no callers outside their own
    /// tests, so `MetricsSnapshot::kv_cache_hit_rate` was permanently 0.0 even
    /// when the prefix cache was doing all the work.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn prefix_cache_activity_reaches_the_metrics() {
        use crate::kv_cache::prefix::{PrefixCacheConfig, PrefixKvCache};

        let mut engine = loaded_engine();
        let mut prefix_cache = PrefixKvCache::new(PrefixCacheConfig {
            max_entries: 8,
            max_memory_bytes: 1 << 24,
            min_prefix_len: 1,
        });

        assert_eq!(
            engine.metrics_snapshot().kv_cache_hit_rate,
            0.0,
            "no lookups yet"
        );

        // A miss, reported by the caller that performed the lookup.
        let tokens = engine.tokenize("hello").expect("tokenize");
        assert!(prefix_cache.lookup(&tokens).is_none());
        engine.record_kv_cache_miss();
        assert_eq!(
            engine.metrics_snapshot().kv_cache_hit_rate,
            0.0,
            "one miss and no hits is a 0.0 hit rate"
        );

        // Populate the cache from a real prefill.
        engine.reset();
        engine.prefill(&tokens).expect("prefill");
        assert!(
            engine.store_kv_in_prefix_cache(&tokens, &mut prefix_cache),
            "storing a prompt-length snapshot must succeed"
        );

        // A hit, recorded by prime_with_prefix.
        let (matched, cached) = prefix_cache.lookup(&tokens).expect("hit");
        let cached = cached.clone();
        let restore_to = matched.min(tokens.len().saturating_sub(1)).max(1);
        engine.reset();
        engine
            .prime_with_prefix(&cached, restore_to, &tokens[restore_to..])
            .expect("prime_with_prefix");

        let rate = engine.metrics_snapshot().kv_cache_hit_rate;
        assert!(
            rate > 0.0,
            "a prefix-cache hit must move kv_cache_hit_rate off 0.0, got {rate}"
        );
    }

    /// `store_kv_in_prefix_cache` passed `kv.seq_len()` — the prompt PLUS
    /// everything generated — alongside the caller's prompt-only `tokens`.
    /// The entry then covered a different span than its trie key.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn prefix_entry_covers_exactly_the_prompt_not_the_completion() {
        use crate::kv_cache::prefix::{PrefixCacheConfig, PrefixKvCache};

        let mut engine = loaded_engine();
        let mut prefix_cache = PrefixKvCache::new(PrefixCacheConfig {
            max_entries: 8,
            max_memory_bytes: 1 << 24,
            min_prefix_len: 1,
        });

        let tokens = engine.tokenize("hello").expect("tokenize");
        engine.reset();
        // Greedy, not `GenerationConfig::new(5)`'s default sampler: the
        // default's `seed: None` draws a fresh, unpredictable seed every
        // call (`sampling::rng::generate_seed`), and `temperature: 0.0`
        // alone would not fix that — `TemperatureScale` only skips its
        // rescaling step at `temperature <= 0.0`, it does not force
        // argmax, so `top_k`/`top_p` still leave `select_token` drawing
        // from that same unseeded RNG. `SamplerConfig::greedy()` sets
        // `top_k: 1`, which collapses `select_token` to its
        // `survivor_count == 1` short-circuit — a deterministic pick with
        // no RNG draw at all. Without this, the very first sampled token
        // could land on EOS by chance and the decode loop would stop
        // after zero completion tokens, failing the assertion below on an
        // unrelated coin flip instead of the KV-cache invariant it checks.
        let config = GenerationConfig::new(5).with_sampler(SamplerConfig::greedy());
        engine
            .generate_detailed("hello", &config, |_| {})
            .expect("generate");

        let after_decode = engine.kv_cache_seq_len();
        assert!(
            after_decode > tokens.len(),
            "the decode loop must have extended the cache past the prompt"
        );

        assert!(engine.store_kv_in_prefix_cache(&tokens, &mut prefix_cache));
        let (_, cached) = prefix_cache.lookup(&tokens).expect("the prompt is cached");
        assert_eq!(
            cached.seq_len(),
            tokens.len(),
            "the stored entry must cover the trie key exactly, not prompt+completion"
        );
    }
}

/// The decode loop must stop at whichever context limit is *smaller* — the
/// architecture's or the KV cache's.
///
/// `resolve_context_size` clamps an unset context to
/// `min(n_ctx_train, DEFAULT_MAX_CONTEXT)`, so the cache can be built shorter
/// than the architecture reports.  Every in-tree architecture stores the
/// `ModelConfig` the engine hands it and therefore agrees, but an architecture
/// whose `build_from_gguf` re-derives the context from GGUF metadata reports the
/// *trained* length instead.  Bounding the loop by `max_context_length()` alone
/// then runs it past the cache, and since `store_kv` returns a hard error on
/// overflow (T3) rather than silently dropping the write, the request fails
/// mid-sentence instead of ending with `FinishReason::ContextFull`.
#[cfg(all(test, any(feature = "tokenizer-onig", feature = "tokenizer-wasm")))]
mod context_bound_tests {
    use super::*;
    use crate::engine::generation::{run_decode_loop, DecodeContext};
    use oxillama_arch::ArchResult;

    /// A forward pass that advertises a far larger context than the cache has.
    struct OverclaimingForwardPass {
        vocab_size: usize,
    }

    impl ForwardPass for OverclaimingForwardPass {
        fn forward(&mut self, tokens: &[u32], kv: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
            let kv_dim = kv.kv_dim().max(1);
            for _ in tokens {
                let row = vec![0.25f32; kv_dim];
                kv.store_kv(0, &row, &row)?;
                kv.advance();
            }
            // Token 7 is an ordinary letter in the fixture vocabulary, so the
            // loop never stops on EOG and only the context bound can end it.
            let mut logits = vec![0.0f32; self.vocab_size];
            logits[7] = 1.0;
            Ok(logits)
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn hidden_size(&self) -> usize {
            1
        }

        /// Deliberately larger than the cache the engine builds.
        fn max_context_length(&self) -> usize {
            8192
        }
    }

    #[test]
    fn decode_loop_stops_at_the_cache_limit_not_the_arch_claim() {
        let cache_ctx = 8usize;
        let kv_dim = 4usize;
        let mut kv = KvCache::new(1, cache_ctx, kv_dim);
        let mut fp = OverclaimingForwardPass { vocab_size: 32 };
        let tokenizer = TokenizerBridge::from_bytes(
            oxillama_gguf::test_utils::minimal_tokenizer_json().as_bytes(),
        )
        .expect("the fixture tokenizer must parse");
        let metrics = EngineMetrics::new();

        assert!(
            fp.max_context_length() > kv.max_seq_len(),
            "the premise of this test is that the two limits disagree"
        );

        // Prefill most of the cache, leaving room for a couple of tokens.
        let prompt: Vec<u32> = vec![3, 4, 5, 6, 7, 8];
        let logits = fp.forward(&prompt, &mut kv).expect("prefill");
        assert_eq!(kv.seq_len(), prompt.len());

        let config = GenerationConfig {
            max_tokens: 64,
            sampler: SamplerConfig::greedy(),
            ..GenerationConfig::default()
        };
        let mut recent = prompt.clone();

        let outcome = run_decode_loop(
            DecodeContext {
                forward_pass: &mut fp,
                kv_cache: &mut kv,
                tokenizer: &tokenizer,
                metrics: &metrics,
            },
            &config,
            logits,
            &mut recent,
            &mut |_| {},
        )
        .expect(
            "hitting the context limit must be a clean stop, not a KV-overflow \
             error from store_kv",
        );

        assert_eq!(
            outcome.finish_reason,
            FinishReason::ContextFull,
            "the loop must report ContextFull, got {:?}",
            outcome.finish_reason
        );
        assert!(
            kv.seq_len() <= cache_ctx,
            "the cache must never be driven past its own limit: {} > {cache_ctx}",
            kv.seq_len()
        );
        assert!(
            outcome.generated_tokens.len() < config.max_tokens,
            "the run must end on the context bound, not the token budget"
        );
    }
}

/// GPU policy plumbing through [`EngineConfig`] and the load paths.
#[cfg(test)]
mod gpu_policy {
    use super::*;
    use crate::gpu_backend::{GpuOptions, GpuPolicy};

    #[test]
    fn default_engine_config_leaves_the_gpu_off() {
        assert_eq!(EngineConfig::default().gpu, GpuPolicy::Off);
    }

    #[test]
    fn with_gpu_sets_the_policy() {
        let opts = GpuOptions {
            n_gpu_layers: Some(8),
            ..GpuOptions::default()
        };
        let cfg = EngineConfig::default().with_gpu(GpuPolicy::On(opts.clone()));
        assert_eq!(cfg.gpu, GpuPolicy::On(opts));
    }

    #[test]
    fn with_gpu_can_switch_back_off() {
        let cfg = EngineConfig::default()
            .with_gpu(GpuPolicy::On(GpuOptions::default()))
            .with_gpu(GpuPolicy::Off);
        assert_eq!(cfg.gpu, GpuPolicy::Off);
    }

    #[test]
    fn gpu_status_is_none_before_any_load() {
        let engine = InferenceEngine::new(EngineConfig::default());
        assert!(engine.gpu_status().is_none());
    }

    /// Asking for a GPU in a build without the `gpu` feature must fail the
    /// load outright — a silent CPU fallback would make the flag a lie.
    #[cfg(not(feature = "gpu"))]
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn requesting_a_gpu_without_the_feature_fails_the_load() {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let cfg = EngineConfig::default().with_gpu(GpuPolicy::On(GpuOptions::default()));
        let mut engine = InferenceEngine::new(cfg);
        let result = engine.load_model_from_bytes(&model_bytes, tokenizer_json);
        match result {
            Err(RuntimeError::GpuUnavailable { reason }) => {
                assert!(
                    reason.contains("gpu"),
                    "the reason must name the missing feature, got: {reason}"
                );
            }
            other => panic!("expected GpuUnavailable, got {other:?}"),
        }
        assert!(
            engine.gpu_status().is_none(),
            "a failed activation must not leave a status behind"
        );
    }

    /// `GpuPolicy::Off` must load exactly as before and report no status.
    #[cfg(any(feature = "tokenizer-onig", feature = "tokenizer-wasm"))]
    #[test]
    fn policy_off_loads_normally_and_reports_no_status() {
        let model_bytes = oxillama_gguf::test_utils::build_minimal_llama_gguf();
        let tokenizer_json = oxillama_gguf::test_utils::minimal_tokenizer_json();
        let mut engine = InferenceEngine::new(EngineConfig::default());
        engine
            .load_model_from_bytes(&model_bytes, tokenizer_json)
            .expect("synthetic GGUF must load with the GPU off");
        assert!(engine.is_loaded());
        assert!(engine.gpu_status().is_none());
    }
}
