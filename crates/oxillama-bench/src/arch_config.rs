//! Architecture-specific benchmark configurations.
//!
//! Provides pre-built benchmark configurations for known model architectures
//! so users can quickly run meaningful benchmarks without hand-tuning
//! parameters for each model family.

use crate::e2e::E2eBenchConfig;
use crate::prefill_decode::PrefillDecodeConfig;

/// Pre-built benchmark configurations for known architectures.
#[derive(Debug, Clone)]
pub struct ArchBenchConfig {
    /// Architecture name (e.g., "llama3", "qwen3").
    pub arch_name: String,
    /// Model sizes to benchmark (e.g., "1B", "3B", "7B").
    pub model_sizes: Vec<String>,
    /// Recommended prompt lengths for this architecture.
    pub prompt_lengths: Vec<usize>,
    /// Recommended decode token count.
    pub decode_tokens: usize,
    /// Recommended context size for benchmarking.
    pub context_size: usize,
    /// Notes about this architecture's characteristics.
    pub notes: String,
}

impl ArchBenchConfig {
    /// Get the benchmark config for LLaMA-3 models.
    pub fn llama3() -> Self {
        Self {
            arch_name: "llama3".to_string(),
            model_sizes: vec![
                "1B".to_string(),
                "3B".to_string(),
                "8B".to_string(),
                "70B".to_string(),
            ],
            prompt_lengths: vec![128, 256, 512, 1024, 2048],
            decode_tokens: 128,
            context_size: 8192,
            notes: "GQA with 8 KV heads. RoPE with 500k base frequency. \
                    Strong prefill throughput due to parallel attention."
                .to_string(),
        }
    }

    /// Get the benchmark config for Qwen3 models.
    pub fn qwen3() -> Self {
        Self {
            arch_name: "qwen3".to_string(),
            model_sizes: vec![
                "0.6B".to_string(),
                "1.7B".to_string(),
                "4B".to_string(),
                "8B".to_string(),
                "32B".to_string(),
            ],
            prompt_lengths: vec![128, 256, 512, 1024],
            decode_tokens: 128,
            context_size: 32768,
            notes: "Large context window (32k). SwiGLU activation. \
                    Supports both dense and MoE variants."
                .to_string(),
        }
    }

    /// Get the benchmark config for Mistral models.
    pub fn mistral() -> Self {
        Self {
            arch_name: "mistral".to_string(),
            model_sizes: vec!["7B".to_string(), "8x7B".to_string(), "8x22B".to_string()],
            prompt_lengths: vec![128, 256, 512, 1024, 4096],
            decode_tokens: 128,
            context_size: 8192,
            notes: "Sliding-window attention (4096). GQA. \
                    MoE variants use top-2 expert routing."
                .to_string(),
        }
    }

    /// Get the benchmark config for Gemma models.
    pub fn gemma() -> Self {
        Self {
            arch_name: "gemma".to_string(),
            model_sizes: vec!["2B".to_string(), "7B".to_string(), "9B".to_string()],
            prompt_lengths: vec![64, 128, 256, 512],
            decode_tokens: 64,
            context_size: 8192,
            notes: "Multi-query attention on 2B, GQA on larger variants. \
                    GeGLU activation. Tied embeddings."
                .to_string(),
        }
    }

    /// Get the benchmark config for Phi models.
    pub fn phi() -> Self {
        Self {
            arch_name: "phi".to_string(),
            model_sizes: vec!["1.5B".to_string(), "2.7B".to_string(), "3.8B".to_string()],
            prompt_lengths: vec![64, 128, 256, 512],
            decode_tokens: 64,
            context_size: 4096,
            notes: "Partial rotary embeddings. Dense attention (no GQA on small variants). \
                    Compact architecture optimised for lower memory budgets."
                .to_string(),
        }
    }

    /// Look up config by architecture name string (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "llama3" | "llama-3" | "llama" => Some(Self::llama3()),
            "qwen3" | "qwen" => Some(Self::qwen3()),
            "mistral" => Some(Self::mistral()),
            "gemma" => Some(Self::gemma()),
            "phi" => Some(Self::phi()),
            _ => None,
        }
    }

    /// List all known architecture names.
    pub fn known_architectures() -> Vec<&'static str> {
        vec!["llama3", "qwen3", "mistral", "gemma", "phi"]
    }

    /// Convert to a [`PrefillDecodeConfig`].
    pub fn to_prefill_decode_config(&self) -> PrefillDecodeConfig {
        PrefillDecodeConfig {
            warmup_iters: 2,
            measure_iters: 5,
            prompt_lengths: self.prompt_lengths.clone(),
            decode_tokens: self.decode_tokens,
        }
    }

    /// Convert to an [`E2eBenchConfig`].
    ///
    /// Uses the largest prompt length as a representative prompt size and
    /// `decode_tokens` as the max token count.
    pub fn to_e2e_config(&self) -> E2eBenchConfig {
        let prompt_size = self.prompt_lengths.last().copied().unwrap_or(128);
        // Build a representative prompt string by repeating a short token
        let prompt = "token ".repeat(prompt_size);

        E2eBenchConfig {
            warmup_iters: 2,
            measure_iters: 5,
            max_tokens: self.decode_tokens,
            prompt,
            track_memory: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_config_non_empty() {
        let cfg = ArchBenchConfig::llama3();
        assert_eq!(cfg.arch_name, "llama3");
        assert!(!cfg.model_sizes.is_empty());
        assert!(!cfg.prompt_lengths.is_empty());
        assert!(cfg.decode_tokens > 0);
        assert!(cfg.context_size > 0);
        assert!(!cfg.notes.is_empty());
    }

    #[test]
    fn test_qwen3_config_non_empty() {
        let cfg = ArchBenchConfig::qwen3();
        assert_eq!(cfg.arch_name, "qwen3");
        assert!(!cfg.model_sizes.is_empty());
        assert!(!cfg.prompt_lengths.is_empty());
        assert!(cfg.decode_tokens > 0);
        assert_eq!(cfg.context_size, 32768);
        assert!(!cfg.notes.is_empty());
    }

    #[test]
    fn test_mistral_config_non_empty() {
        let cfg = ArchBenchConfig::mistral();
        assert_eq!(cfg.arch_name, "mistral");
        assert!(!cfg.model_sizes.is_empty());
        assert!(!cfg.prompt_lengths.is_empty());
        assert!(cfg.decode_tokens > 0);
        assert!(cfg.context_size > 0);
        assert!(!cfg.notes.is_empty());
    }

    #[test]
    fn test_gemma_config_non_empty() {
        let cfg = ArchBenchConfig::gemma();
        assert_eq!(cfg.arch_name, "gemma");
        assert!(!cfg.model_sizes.is_empty());
        assert!(!cfg.prompt_lengths.is_empty());
        assert_eq!(cfg.decode_tokens, 64);
        assert_eq!(cfg.context_size, 8192);
        assert!(!cfg.notes.is_empty());
    }

    #[test]
    fn test_phi_config_non_empty() {
        let cfg = ArchBenchConfig::phi();
        assert_eq!(cfg.arch_name, "phi");
        assert!(!cfg.model_sizes.is_empty());
        assert!(!cfg.prompt_lengths.is_empty());
        assert_eq!(cfg.decode_tokens, 64);
        assert_eq!(cfg.context_size, 4096);
        assert!(!cfg.notes.is_empty());
    }

    #[test]
    fn test_from_name_exact_match() {
        assert!(ArchBenchConfig::from_name("llama3").is_some());
        assert!(ArchBenchConfig::from_name("qwen3").is_some());
        assert!(ArchBenchConfig::from_name("mistral").is_some());
        assert!(ArchBenchConfig::from_name("gemma").is_some());
        assert!(ArchBenchConfig::from_name("phi").is_some());
    }

    #[test]
    fn test_from_name_aliases() {
        assert!(ArchBenchConfig::from_name("llama-3").is_some());
        assert!(ArchBenchConfig::from_name("llama").is_some());
        assert!(ArchBenchConfig::from_name("qwen").is_some());
    }

    #[test]
    fn test_from_name_case_insensitive() {
        assert!(ArchBenchConfig::from_name("LLaMA3").is_some());
        assert!(ArchBenchConfig::from_name("QWEN3").is_some());
        assert!(ArchBenchConfig::from_name("Mistral").is_some());
        assert!(ArchBenchConfig::from_name("GEMMA").is_some());
        assert!(ArchBenchConfig::from_name("PHI").is_some());
    }

    #[test]
    fn test_from_name_unknown_returns_none() {
        assert!(ArchBenchConfig::from_name("gpt4").is_none());
        assert!(ArchBenchConfig::from_name("").is_none());
        assert!(ArchBenchConfig::from_name("unknown_model").is_none());
    }

    #[test]
    fn test_known_architectures_list() {
        let archs = ArchBenchConfig::known_architectures();
        assert_eq!(archs.len(), 5);
        assert!(archs.contains(&"llama3"));
        assert!(archs.contains(&"qwen3"));
        assert!(archs.contains(&"mistral"));
        assert!(archs.contains(&"gemma"));
        assert!(archs.contains(&"phi"));
    }

    #[test]
    fn test_to_prefill_decode_config() {
        let arch = ArchBenchConfig::llama3();
        let pd = arch.to_prefill_decode_config();

        assert_eq!(pd.warmup_iters, 2);
        assert_eq!(pd.measure_iters, 5);
        assert_eq!(pd.prompt_lengths, vec![128, 256, 512, 1024, 2048]);
        assert_eq!(pd.decode_tokens, 128);
    }

    #[test]
    fn test_to_e2e_config() {
        let arch = ArchBenchConfig::qwen3();
        let e2e = arch.to_e2e_config();

        assert_eq!(e2e.warmup_iters, 2);
        assert_eq!(e2e.measure_iters, 5);
        assert_eq!(e2e.max_tokens, 128);
        assert!(e2e.track_memory);
        // Prompt should be constructed from the largest prompt length (1024)
        assert!(!e2e.prompt.is_empty());
    }

    #[test]
    fn test_to_e2e_config_prompt_from_largest_length() {
        let arch = ArchBenchConfig::phi();
        let e2e = arch.to_e2e_config();
        // Phi's largest prompt length is 512 => "token " × 512 = 3072 chars
        let expected_len = "token ".len() * 512;
        assert_eq!(e2e.prompt.len(), expected_len);
    }

    #[test]
    fn test_all_architectures_produce_valid_prefill_decode_config() {
        for name in ArchBenchConfig::known_architectures() {
            let arch = ArchBenchConfig::from_name(name)
                .unwrap_or_else(|| panic!("from_name failed for {name}"));
            let pd = arch.to_prefill_decode_config();
            assert!(
                !pd.prompt_lengths.is_empty(),
                "{name}: empty prompt_lengths"
            );
            assert!(pd.decode_tokens > 0, "{name}: decode_tokens is 0");
        }
    }

    #[test]
    fn test_all_architectures_produce_valid_e2e_config() {
        for name in ArchBenchConfig::known_architectures() {
            let arch = ArchBenchConfig::from_name(name)
                .unwrap_or_else(|| panic!("from_name failed for {name}"));
            let e2e = arch.to_e2e_config();
            assert!(!e2e.prompt.is_empty(), "{name}: empty prompt");
            assert!(e2e.max_tokens > 0, "{name}: max_tokens is 0");
        }
    }
}
