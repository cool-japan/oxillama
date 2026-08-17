// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Pre-tokenizer regular expressions for byte-level BPE vocabularies.
//!
//! A byte-level BPE tokenizer never applies its merge table to a whole string.
//! It first *pre-tokenizes* the input into words / runs of digits / runs of
//! punctuation, and only then runs BPE independently inside each piece.  Which
//! split pattern is used is model-specific and is recorded in the GGUF metadata
//! key `tokenizer.ggml.pre`.
//!
//! The patterns here are transcribed from llama.cpp's
//! `llm_tokenizer_bpe::llm_tokenizer_bpe` (`src/llama-vocab.cpp`) so that the
//! resulting token IDs match llama.cpp exactly.  llama.cpp hand-codes the two
//! hottest patterns (GPT-2 and LLaMA-3) to avoid `std::regex`; we use
//! `fancy-regex` (pure Rust, supports the `(?!\S)` look-ahead these patterns
//! need) which keeps the definition declarative and therefore auditable.
//!
//! Splitting semantics match llama.cpp's `unicode_regex_split`: every pattern
//! in the list is applied in order to the pieces produced by the previous one,
//! and the *gaps* between matches become pieces of their own.

use fancy_regex::Regex;

use crate::error::{RuntimeError, RuntimeResult};

/// Pre-tokenizer family, resolved from `tokenizer.ggml.pre`.
///
/// Variants are grouped by the regex list they share, exactly as llama.cpp
/// groups its `LLAMA_VOCAB_PRE_TYPE_*` enum in a `switch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PreType {
    /// GPT-2's original pattern — also llama.cpp's `default`.
    #[default]
    Default,
    /// LLaMA-3 / Falcon-3 / Pixtral: digits in runs of at most three.
    Llama3,
    /// Qwen-2 / StableLM-2 / Hunyuan: like LLaMA-3 but one digit at a time.
    Qwen2,
    /// Qwen-3.5: Qwen-2 plus combining-mark awareness.
    Qwen35,
    /// DeepSeek "LLM" series.
    DeepSeekLlm,
    /// DeepSeek "Coder" series.
    DeepSeekCoder,
    /// DeepSeek-V3 / Hunyuan-dense.
    DeepSeek3Llm,
    /// Falcon (the original, not Falcon-3).
    Falcon,
    /// StarCoder / Refact / Command-R / SmolLM / CodeShell / EXAONE / Minerva.
    StarCoder,
    /// GPT-2 proper / MPT / OLMo / Jais / Trillion.
    Gpt2,
    /// BLOOM / Poro / GPT-3 Finnish.
    Bloom,
    /// Viking.
    Viking,
    /// ChatGLM-4.
    ChatGlm4,
    /// Tekken (Mistral NeMo).
    Tekken,
    /// GPT-4o / MiniMax-M2.
    Gpt4o,
    /// Seed-Coder.
    SeedCoder,
    /// BailingMoE.
    BailingMoe,
    /// Grok-2.
    Grok2,
    /// DBRX / Smaug — identical regex to LLaMA-3 but no `ignore_merges`.
    Dbrx,
}

/// LLaMA-3's pattern (adapted by llama.cpp to avoid `(?i:...)`).
const RE_LLAMA3: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Qwen-2 / StableLM-2 pattern — LLaMA-3 with single-digit number runs.
const RE_QWEN2: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Qwen-3.5 pattern — Qwen-2 with `\p{M}` combining marks attached to letters.
const RE_QWEN35: &str = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The original GPT-2 pattern.
const RE_GPT2: &str =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";

/// GPT-4o / MiniMax-M2 (llama.cpp's look-ahead-based case folding).
const RE_GPT4O: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Tekken (Mistral NeMo).
const RE_TEKKEN: &str = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// DeepSeek-LLM's explicit letter class (transcribed verbatim from llama.cpp).
const RE_DEEPSEEK_LLM_LETTERS: &str = "\\s?[A-Za-z\u{00B5}\u{00C0}-\u{00D6}\u{00D8}-\u{00F6}\u{00F8}-\u{01BA}\u{01BC}-\u{01BF}\u{01C4}-\u{0293}\u{0295}-\u{02AF}\u{0370}-\u{0373}\u{0376}\u{0377}\u{037B}-\u{037D}\u{037F}\u{0386}\u{0388}-\u{038A}\u{038C}\u{038E}-\u{03A1}\u{03A3}-\u{03F5}\u{03F7}-\u{0481}\u{048A}-\u{052F}\u{0531}-\u{0556}\u{10A0}-\u{10C5}\u{13A0}-\u{13F5}\u{13F8}-\u{13FD}\u{1C90}-\u{1CBA}\u{1CBD}-\u{1CBF}\u{1D00}-\u{1D2B}\u{1D6B}-\u{1D77}\u{1D79}-\u{1D9A}\u{1E00}-\u{1F15}\u{1F18}-\u{1F1D}\u{1F20}-\u{1F45}\u{1F48}-\u{1F4D}\u{1F50}-\u{1F57}\u{1F59}\u{1F5B}\u{1F5D}\u{1F5F}-\u{1F7D}\u{1F80}-\u{1FB4}\u{1FB6}-\u{1FBC}\u{1FBE}\u{1FC2}-\u{1FC4}\u{1FC6}-\u{1FCC}\u{1FD0}-\u{1FD3}\u{1FD6}-\u{1FDB}\u{1FE0}-\u{1FEC}\u{1FF2}-\u{1FF4}\u{1FF6}-\u{1FFC}\u{2102}\u{2107}\u{210A}-\u{2113}\u{2115}\u{2119}-\u{211D}\u{2124}\u{2126}\u{2128}\u{212A}-\u{212D}\u{212F}-\u{2134}\u{2139}\u{213C}-\u{213F}\u{2145}-\u{2149}\u{214E}\u{2183}\u{2184}\u{2C00}-\u{2C7B}\u{2C7E}-\u{2CE4}\u{2CEB}-\u{2CEE}\u{2CF2}\u{2CF3}\u{A640}-\u{A66D}\u{A680}-\u{A69B}\u{A722}-\u{A76F}\u{A771}-\u{A787}\u{A78B}-\u{A78E}\u{AB70}-\u{ABBF}\u{FB00}-\u{FB06}\u{FB13}-\u{FB17}\u{FF21}-\u{FF3A}\u{FF41}-\u{FF5A}\u{10400}-\u{1044F}\u{104B0}-\u{104D3}\u{104D8}-\u{104FB}\u{10C80}-\u{10CB2}\u{10CC0}-\u{10CF2}\u{118A0}-\u{118DF}\u{1E900}-\u{1E943}]+";

/// DeepSeek's CJK / Hangul class.
const RE_DEEPSEEK_CJK: &str = "[\u{4E00}-\u{9FA5}\u{0800}-\u{4E00}\u{AC00}-\u{D7FF}]+";

/// Returns the ordered regex list for `pre`.
pub fn regex_exprs(pre: PreType) -> &'static [&'static str] {
    match pre {
        PreType::Default | PreType::Gpt2 => &[RE_GPT2],
        PreType::Llama3 | PreType::Dbrx => &[RE_LLAMA3],
        PreType::Qwen2 => &[RE_QWEN2],
        PreType::Qwen35 => &[RE_QWEN35],
        PreType::ChatGlm4 => &[RE_LLAMA3],
        PreType::DeepSeekLlm => &[
            "[\r\n]",
            RE_DEEPSEEK_LLM_LETTERS,
            "\\s?[!-/:-~\u{FF01}-\u{FF0F}\u{FF1A}-\u{FF5E}\u{2018}-\u{201F}\u{3000}-\u{3002}]+",
            "\\s+$",
            RE_DEEPSEEK_CJK,
            "\\p{N}+",
        ],
        PreType::DeepSeekCoder => &[
            "[\r\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            RE_DEEPSEEK_CJK,
            "\\p{N}",
        ],
        PreType::DeepSeek3Llm => &[
            "\\p{N}{1,3}",
            "[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::Falcon => &[
            "[\\p{P}\\$\\+<=>\\^~\\|`]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "[0-9][0-9][0-9]",
        ],
        PreType::StarCoder => &[
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        ],
        PreType::Bloom => &[" ?[^(\\s|.,!?…。，、।۔،)]+"],
        PreType::Viking => &[" ?[^(\\s|.,!?…。，、।۔،)]+", "\\p{N}"],
        PreType::Tekken => &[RE_TEKKEN],
        PreType::Gpt4o => &[RE_GPT4O],
        PreType::SeedCoder => &[
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ],
        PreType::BailingMoe => &[
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        ],
        PreType::Grok2 => &[RE_QWEN2],
    }
}

/// Resolve `tokenizer.ggml.pre` to a [`PreType`].
///
/// Returns `None` for names this build does not know; the caller is expected to
/// fall back to [`PreType::Default`] and log a warning, which mirrors what
/// llama.cpp prints when the key is missing entirely.
pub fn pre_type_from_name(name: &str) -> Option<PreType> {
    let t = match name {
        "default" => PreType::Default,
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" => PreType::Llama3,
        "dbrx" | "smaug-bpe" => PreType::Dbrx,
        "deepseek-llm" => PreType::DeepSeekLlm,
        "deepseek-coder" => PreType::DeepSeekCoder,
        "deepseek-v3" | "hunyuan-dense" => PreType::DeepSeek3Llm,
        "falcon" => PreType::Falcon,
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "exaone4"
        | "minerva-7b" => PreType::StarCoder,
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "jina-v1-en" | "jina-v2-es" | "jina-v2-de"
        | "jina-v2-code" | "roberta-bpe" | "mpt" | "olmo" | "jais" | "trillion"
        | "granite-docling" => PreType::Gpt2,
        "qwen2" | "stablelm2" | "hunyuan" | "solar-open" => PreType::Qwen2,
        "qwen35" => PreType::Qwen35,
        "poro-chat" | "bloom" | "gpt3-finnish" => PreType::Bloom,
        "viking" => PreType::Viking,
        "chatglm-bpe" | "glm4" => PreType::ChatGlm4,
        "tekken" => PreType::Tekken,
        "gpt-4o" | "llama4" | "minimax-m2" => PreType::Gpt4o,
        "seed-coder" => PreType::SeedCoder,
        "bailingmoe" | "bailingmoe2" => PreType::BailingMoe,
        "grok-2" => PreType::Grok2,
        _ => return None,
    };
    Some(t)
}

/// `true` when llama.cpp sets `ignore_merges` for this pre-tokenizer.
///
/// With `ignore_merges`, a pre-token that is *itself* a vocabulary entry is
/// emitted as-is instead of being re-derived through the merge table.  Omitting
/// this produces subtly different — and wrong — IDs for LLaMA-3.
pub fn ignore_merges(pre: PreType) -> bool {
    matches!(pre, PreType::Llama3 | PreType::Tekken)
}

/// `true` when this pre-tokenizer family implies `add_bos` in the absence of an
/// explicit `tokenizer.ggml.add_bos_token`.
///
/// Only the LLaMA-3 and Tekken families do.  It matters because GGUFs converted
/// before `tokenizer.ggml.add_bos_token` existed — Meta-Llama-3-8B among them —
/// carry no flag at all, and dropping BOS from a LLaMA-3 prompt measurably
/// degrades the output.
///
/// # Deliberate divergence from llama.cpp
///
/// llama.cpp reaches the same `add_bos = true` **only when
/// `tokenizer.ggml.pre` is present and names `llama3`/`llama-bpe`/`tekken`/…**
/// (`src/llama-vocab.cpp`, the `tokenizer_pre == "llama3"` and
/// `tokenizer_pre == "tekken"` arms).  When the key is *missing* it takes the
/// warning branch instead — `missing pre-tokenizer type, using: 'default'` /
/// `GENERATION QUALITY WILL BE DEGRADED!` — which selects
/// `LLAMA_VOCAB_PRE_TYPE_DEFAULT` and leaves the BPE default `add_bos = false`
/// untouched.
///
/// OxiLLaMa instead infers [`PreType::Llama3`] from the vocabulary's special
/// tokens and then applies this function, so on a GGUF with no
/// `tokenizer.ggml.pre` key (e.g. the 2024-vintage
/// `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`) OxiLLaMa prepends `128000` where
/// llama.cpp prepends nothing.  Both tokenize the prompt body identically;
/// only the leading BOS differs.  Measured consequence: byte-identical logits
/// are unobtainable from the *string* prompt on such a file, because the two
/// models are conditioned on different sequences.  Feed explicit ids
/// (`oxillama run --dump-logits … --prompt-tokens …`) when byte-level
/// comparability matters.
pub fn implies_add_bos(pre: PreType) -> bool {
    matches!(pre, PreType::Llama3 | PreType::Tekken)
}

/// A compiled pre-tokenizer.
pub struct PreTokenizer {
    regexes: Vec<Regex>,
    pre: PreType,
}

impl core::fmt::Debug for PreTokenizer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreTokenizer")
            .field("pre", &self.pre)
            .field("n_regexes", &self.regexes.len())
            .finish()
    }
}

impl PreTokenizer {
    /// Compile the regex list for `pre`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::TokenizerError`] if a pattern fails to compile,
    /// which would indicate a bug in the transcribed table rather than bad
    /// model data.
    pub fn new(pre: PreType) -> RuntimeResult<Self> {
        let mut regexes = Vec::new();
        for expr in regex_exprs(pre) {
            let re = Regex::new(expr).map_err(|e| RuntimeError::TokenizerError {
                message: format!("failed to compile pre-tokenizer regex for {pre:?}: {e}"),
            })?;
            regexes.push(re);
        }
        Ok(Self { regexes, pre })
    }

    /// The pre-tokenizer family this instance implements.
    pub fn pre_type(&self) -> PreType {
        self.pre
    }

    /// Split `text` into pre-token pieces.
    ///
    /// Each regex is applied in turn to the pieces produced by the previous
    /// one; both the matches and the gaps between them survive as pieces, so
    /// the concatenation of the result always reproduces `text` exactly.
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces: Vec<&'a str> = vec![text];
        for re in &self.regexes {
            let mut next: Vec<&'a str> = Vec::with_capacity(pieces.len());
            for piece in pieces.drain(..) {
                split_one(re, piece, &mut next);
            }
            pieces = next;
        }
        pieces.retain(|p| !p.is_empty());
        pieces
    }
}

/// Split `piece` by `re`, pushing matches and gaps onto `out` in order.
fn split_one<'a>(re: &Regex, piece: &'a str, out: &mut Vec<&'a str>) {
    if piece.is_empty() {
        return;
    }
    let base = out.len();
    let mut last_end = 0usize;
    for m in re.find_iter(piece) {
        let Ok(m) = m else {
            // Backtrack limit or catastrophic pattern: keep the piece intact
            // rather than corrupting the token stream.
            out.truncate(base);
            out.push(piece);
            return;
        };
        if m.start() > last_end {
            out.push(&piece[last_end..m.start()]);
        }
        if m.end() > m.start() {
            out.push(&piece[m.start()..m.end()]);
        }
        last_end = m.end();
    }
    if last_end < piece.len() {
        out.push(&piece[last_end..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(pre: PreType, text: &str) -> Vec<String> {
        let pt = PreTokenizer::new(pre).expect("test: regex must compile");
        pt.split(text).into_iter().map(str::to_string).collect()
    }

    #[test]
    fn every_pre_type_compiles() {
        for pre in [
            PreType::Default,
            PreType::Llama3,
            PreType::Qwen2,
            PreType::Qwen35,
            PreType::DeepSeekLlm,
            PreType::DeepSeekCoder,
            PreType::DeepSeek3Llm,
            PreType::Falcon,
            PreType::StarCoder,
            PreType::Gpt2,
            PreType::Bloom,
            PreType::Viking,
            PreType::ChatGlm4,
            PreType::Tekken,
            PreType::Gpt4o,
            PreType::SeedCoder,
            PreType::BailingMoe,
            PreType::Grok2,
            PreType::Dbrx,
        ] {
            PreTokenizer::new(pre)
                .unwrap_or_else(|e| panic!("test: {pre:?} must compile, got {e}"));
        }
    }

    #[test]
    fn llama3_splits_digits_in_groups_of_three() {
        assert_eq!(split(PreType::Llama3, "1234567"), ["123", "456", "7"]);
    }

    #[test]
    fn qwen2_splits_digits_one_at_a_time() {
        assert_eq!(split(PreType::Qwen2, "1234"), ["1", "2", "3", "4"]);
    }

    #[test]
    fn gpt2_keeps_digit_runs_together() {
        assert_eq!(split(PreType::Gpt2, "1234"), ["1234"]);
    }

    #[test]
    fn leading_space_attaches_to_word() {
        assert_eq!(
            split(PreType::Llama3, "The capital of France is"),
            ["The", " capital", " of", " France", " is"]
        );
    }

    #[test]
    fn split_is_lossless_for_unicode() {
        for pre in [PreType::Llama3, PreType::Qwen2, PreType::Gpt2] {
            let text = "こんにちは世界 🚀 Hello, y'all! 42\n\tend";
            let joined: String = split(pre, text).concat();
            assert_eq!(joined, text, "{pre:?} split must be lossless");
        }
    }

    #[test]
    fn unknown_pre_name_is_none() {
        assert!(pre_type_from_name("definitely-not-a-real-pretokenizer").is_none());
        assert_eq!(pre_type_from_name("llama-bpe"), Some(PreType::Llama3));
        assert_eq!(pre_type_from_name("qwen2"), Some(PreType::Qwen2));
    }

    #[test]
    fn llama3_family_ignores_merges() {
        assert!(ignore_merges(PreType::Llama3));
        assert!(!ignore_merges(PreType::Qwen2));
        assert!(!ignore_merges(PreType::Default));
    }
}
