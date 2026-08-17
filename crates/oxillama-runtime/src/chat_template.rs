// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Chat template detection and rendering, shared by every chat frontend.
//!
//! ## Why this lives in `oxillama-runtime`
//!
//! This logic started life inside `oxillama-cli` (used by the `chat` REPL and
//! the TUI). `oxillama-server` could not reach it — the server depends on
//! `oxillama-runtime`, never on the CLI — so `/v1/chat/completions` rendered
//! every request through a hardcoded, model-agnostic
//! `<|system|>…<|end|>` skeleton that no real model was ever trained on. On
//! Qwen3 that surfaced as literal `<|end|#>` text in the returned content:
//! the model, having never seen that framing, imitated the fake marker as
//! ordinary output.
//!
//! Both frontends now share this one implementation. `oxillama-cli` re-exports
//! it (keeping its `SessionSnapshot`-coupled helpers CLI-side) and
//! `oxillama-server` caches a resolved [`ChatTemplate`] in its `AppState` at
//! model-load time.
//!
//! ## Approach
//!
//! Writing a full Jinja2 interpreter to execute the GGUF-embedded
//! `tokenizer.chat_template` string is out of scope here. Instead this module
//! *fingerprints* the template: real-world chat templates for a given family
//! are textually stable (the literal control-token strings `<|im_start|>` /
//! `<|start_header_id|>` / `[INST]` appear verbatim in the Jinja source
//! because the template's whole job is to interleave them with message
//! content). Detecting those literals is enough to pick the matching
//! pure-Rust formatter, without evaluating the template's control flow.
//!
//! When no `tokenizer.chat_template` is present at all, the same fingerprint
//! scan runs over the model's vocabulary (`tokenizer.ggml.tokens`): if the
//! vocabulary itself defines `<|im_start|>` as a token, the model was almost
//! certainly trained with ChatML-shaped data even if the GGUF conversion
//! dropped the template string. Only when neither source yields a match does
//! this fall back to a documented default (ChatML) — never to a raw
//! `User:`/`Assistant:` transcript, which no instruct model was trained on.
//!
//! ## Verified against
//!
//! - `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` — `tokenizer.chat_template`
//!   contains the literal `<|start_header_id|>` → [`ChatTemplate::Llama3`].
//! - `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` — `tokenizer.chat_template`
//!   contains the literal `<|im_start|>` → [`ChatTemplate::ChatMl`].

use oxillama_gguf::{MetadataStore, MetadataValue};

/// One turn to render: a role (`"system"` / `"user"` / `"assistant"` /
/// `"tool"`) and its text content.
#[derive(Debug, Clone, Copy)]
pub struct Turn<'a> {
    /// The speaker of this turn.
    pub role: &'a str,
    /// The turn's text content.
    pub content: &'a str,
}

/// A detected (or defaulted) chat template family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatTemplate {
    /// Meta LLaMA-3 / LLaMA-3.1 instruct format
    /// (`<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`).
    Llama3,
    /// ChatML, used by Qwen, Mistral-Instruct (new), Yi, and many others
    /// (`<|im_start|>role\ncontent<|im_end|>`).
    ChatMl,
    /// Mistral-style `[INST] … [/INST]` instruction format.
    Mistral,
    /// Alpaca-style `### Role:\ncontent` format (WizardLM, Alpaca fine-tunes).
    Alpaca,
}

impl ChatTemplate {
    /// Render `turns` into a single prompt string.
    ///
    /// `add_generation_prompt` appends the marker that elicits an assistant
    /// reply (the model has not produced it yet — this is what tells the
    /// model it is now its turn to speak).
    ///
    /// Control tokens are emitted as their literal text (e.g. `<|im_end|>`).
    /// Callers MUST encode the result with `parse_special = true` so these
    /// are recognised as single tokens rather than split byte-by-byte.
    ///
    /// Whether callers should also pass `add_special = false` to the
    /// tokenizer depends on the template: see [`Self::emits_literal_bos`].
    pub fn render(self, turns: &[Turn<'_>], add_generation_prompt: bool) -> String {
        match self {
            ChatTemplate::Llama3 => render_llama3(turns, add_generation_prompt),
            ChatTemplate::ChatMl => render_chatml(turns, add_generation_prompt),
            ChatTemplate::Mistral => render_mistral(turns, add_generation_prompt),
            ChatTemplate::Alpaca => render_alpaca(turns, add_generation_prompt),
        }
    }

    /// `true` when [`Self::render`]'s output already contains the model's
    /// begin-of-text/BOS marker as literal text.
    ///
    /// [`ChatTemplate::Llama3`] emits `<|begin_of_text|>` and
    /// [`ChatTemplate::Mistral`] emits `<s>` up front, so encoding their
    /// output with `add_special = true` would duplicate the BOS token.
    /// [`ChatTemplate::ChatMl`] and [`ChatTemplate::Alpaca`] emit no such
    /// marker — encoding their output needs `add_special = true` so the
    /// tokenizer's own `tokenizer.ggml.add_bos_token` policy still applies;
    /// passing `add_special = false` unconditionally for every template (as
    /// an earlier version of this module did) would otherwise silently drop
    /// BOS for any ChatML/Alpaca model that requires it.
    pub fn emits_literal_bos(self) -> bool {
        matches!(self, ChatTemplate::Llama3 | ChatTemplate::Mistral)
    }

    /// A short, stable identifier for this family, suitable for logs and for
    /// diagnostic fields on an API response.
    pub fn as_str(self) -> &'static str {
        match self {
            ChatTemplate::Llama3 => "llama3",
            ChatTemplate::ChatMl => "chatml",
            ChatTemplate::Mistral => "mistral",
            ChatTemplate::Alpaca => "alpaca",
        }
    }

    /// Detect a template family from a raw Jinja `chat_template` string by
    /// fingerprinting the literal control-token text it contains.
    ///
    /// Order matters: check the most specific / least ambiguous markers
    /// first. Returns `None` when nothing recognisable is found.
    pub fn detect_from_jinja(template: &str) -> Option<Self> {
        if template.contains("<|start_header_id|>") {
            Some(ChatTemplate::Llama3)
        } else if template.contains("<|im_start|>") {
            Some(ChatTemplate::ChatMl)
        } else if template.contains("[INST]") {
            Some(ChatTemplate::Mistral)
        } else if template.contains("### Instruction") || template.contains("### Response") {
            Some(ChatTemplate::Alpaca)
        } else {
            None
        }
    }

    /// Detect a template family from the model's vocabulary strings, used
    /// when no `tokenizer.chat_template` metadata is present.
    pub fn detect_from_vocab<'a>(tokens: impl IntoIterator<Item = &'a str>) -> Option<Self> {
        let mut saw_im_start = false;
        let mut saw_header_id = false;
        for tok in tokens {
            if tok == "<|start_header_id|>" {
                saw_header_id = true;
            } else if tok == "<|im_start|>" {
                saw_im_start = true;
            }
        }
        if saw_header_id {
            Some(ChatTemplate::Llama3)
        } else if saw_im_start {
            Some(ChatTemplate::ChatMl)
        } else {
            None
        }
    }

    /// Resolve the chat template for a model from its GGUF metadata.
    ///
    /// Priority:
    /// 1. Fingerprint `tokenizer.chat_template`, if present.
    /// 2. Fingerprint the vocabulary itself (`tokenizer.ggml.tokens`).
    /// 3. Fall back to [`ChatTemplate::ChatMl`] — documented default, never
    ///    the historical `User:`/`Assistant:` transcript format.
    pub fn resolve(metadata: &MetadataStore) -> Self {
        if let Some(MetadataValue::String(template)) = metadata.get("tokenizer.chat_template") {
            if let Some(detected) = Self::detect_from_jinja(template) {
                return detected;
            }
        }
        if let Some(MetadataValue::Array(tokens)) = metadata.get("tokenizer.ggml.tokens") {
            let as_str = tokens.iter().filter_map(MetadataValue::as_str);
            if let Some(detected) = Self::detect_from_vocab(as_str) {
                return detected;
            }
        }
        ChatTemplate::ChatMl
    }

    /// Best-effort resolution straight from a GGUF file path.
    ///
    /// Any failure to open/parse the file (including "the file does not
    /// exist", which happens in unit tests that construct a `TuiApp` with a
    /// placeholder path) falls back to [`ChatTemplate::ChatMl`] rather than
    /// propagating an error — chat template selection is a UX nicety, not a
    /// precondition for starting a chat session.
    pub fn resolve_from_path(model_path: &std::path::Path) -> Self {
        match oxillama_gguf::GgufModel::load(model_path) {
            Ok(model) => Self::resolve(&model.file.metadata),
            Err(_) => ChatTemplate::ChatMl,
        }
    }
}

impl core::fmt::Display for ChatTemplate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Default for ChatTemplate {
    /// [`ChatTemplate::ChatMl`], matching [`ChatTemplate::resolve`]'s
    /// documented fallback.
    fn default() -> Self {
        ChatTemplate::ChatMl
    }
}

fn render_llama3(turns: &[Turn<'_>], add_generation_prompt: bool) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for turn in turns {
        out.push_str("<|start_header_id|>");
        out.push_str(turn.role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(turn.content.trim());
        out.push_str("<|eot_id|>");
    }
    if add_generation_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

fn render_chatml(turns: &[Turn<'_>], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for turn in turns {
        out.push_str("<|im_start|>");
        out.push_str(turn.role);
        out.push('\n');
        out.push_str(turn.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

fn render_mistral(turns: &[Turn<'_>], add_generation_prompt: bool) -> String {
    let mut out = String::from("<s>");
    for turn in turns {
        match turn.role {
            "system" => {
                // Mistral has no dedicated system turn; fold it into the
                // next instruction block as leading text.
                out.push_str(turn.content.trim());
                out.push_str("\n\n");
            }
            "user" => {
                out.push_str("[INST] ");
                out.push_str(turn.content.trim());
                out.push_str(" [/INST]");
            }
            "assistant" => {
                out.push(' ');
                out.push_str(turn.content.trim());
                out.push_str("</s>");
            }
            _ => {}
        }
    }
    // Mistral has no explicit generation-prompt marker beyond `[/INST]`
    // already having been emitted for the trailing user turn; nothing to
    // append when `add_generation_prompt` is set.
    let _ = add_generation_prompt;
    out
}

fn render_alpaca(turns: &[Turn<'_>], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for turn in turns {
        let header = match turn.role {
            "system" => "System",
            "user" => "Instruction",
            "assistant" => "Response",
            _ => "Unknown",
        };
        out.push_str("### ");
        out.push_str(header);
        out.push_str(":\n");
        out.push_str(turn.content.trim());
        out.push_str("\n\n");
    }
    if add_generation_prompt {
        out.push_str("### Response:\n");
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_llama3_from_real_template_fragment() {
        // Fragment lifted (fingerprint only) from Meta-Llama-3-8B-Instruct.
        let template = "{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>' %}{{ content }}{% endfor %}";
        assert_eq!(
            ChatTemplate::detect_from_jinja(template),
            Some(ChatTemplate::Llama3)
        );
    }

    #[test]
    fn detects_chatml_from_real_template_fragment() {
        // Fragment lifted (fingerprint only) from Qwen3-4B-Instruct.
        let template = "{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}{%- endif %}";
        assert_eq!(
            ChatTemplate::detect_from_jinja(template),
            Some(ChatTemplate::ChatMl)
        );
    }

    #[test]
    fn detects_mistral_inst_style() {
        let template = "{{ bos_token }}{% for message in messages %}[INST] {{ message['content'] }} [/INST]{% endfor %}";
        assert_eq!(
            ChatTemplate::detect_from_jinja(template),
            Some(ChatTemplate::Mistral)
        );
    }

    #[test]
    fn unknown_template_detects_none() {
        assert_eq!(ChatTemplate::detect_from_jinja("{{ nonsense }}"), None);
    }

    #[test]
    fn vocab_fallback_detects_chatml() {
        let tokens = vec!["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"];
        assert_eq!(
            ChatTemplate::detect_from_vocab(tokens),
            Some(ChatTemplate::ChatMl)
        );
    }

    #[test]
    fn vocab_fallback_detects_llama3() {
        let tokens = vec!["<|begin_of_text|>", "<|start_header_id|>", "<|eot_id|>"];
        assert_eq!(
            ChatTemplate::detect_from_vocab(tokens),
            Some(ChatTemplate::Llama3)
        );
    }

    #[test]
    fn vocab_fallback_detects_none_for_plain_vocab() {
        let tokens = vec!["<unk>", "<s>", "</s>", "hello", "world"];
        assert_eq!(ChatTemplate::detect_from_vocab(tokens), None);
    }

    #[test]
    fn resolve_from_missing_path_defaults_to_chatml() {
        let bogus = std::path::PathBuf::from("/nonexistent/oxillama_test_model_xyz.gguf");
        assert_eq!(
            ChatTemplate::resolve_from_path(&bogus),
            ChatTemplate::ChatMl
        );
    }

    #[test]
    fn default_is_chatml() {
        assert_eq!(ChatTemplate::default(), ChatTemplate::ChatMl);
    }

    #[test]
    fn as_str_and_display_agree() {
        for t in [
            ChatTemplate::Llama3,
            ChatTemplate::ChatMl,
            ChatTemplate::Mistral,
            ChatTemplate::Alpaca,
        ] {
            assert_eq!(t.to_string(), t.as_str());
        }
        assert_eq!(ChatTemplate::Llama3.as_str(), "llama3");
        assert_eq!(ChatTemplate::ChatMl.as_str(), "chatml");
    }

    #[test]
    fn llama3_render_single_user_turn() {
        let turns = [Turn {
            role: "user",
            content: "Hello!",
        }];
        let rendered = ChatTemplate::Llama3.render(&turns, true);
        assert!(rendered.starts_with("<|begin_of_text|>"));
        assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|>"));
        assert!(rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama3_render_system_emitted_once() {
        let turns = [
            Turn {
                role: "system",
                content: "Be terse.",
            },
            Turn {
                role: "user",
                content: "Hi",
            },
            Turn {
                role: "assistant",
                content: "Hey.",
            },
            Turn {
                role: "user",
                content: "Again?",
            },
        ];
        let rendered = ChatTemplate::Llama3.render(&turns, true);
        // The system content must appear exactly once, not once per turn.
        assert_eq!(rendered.matches("Be terse.").count(), 1);
    }

    #[test]
    fn chatml_render_matches_expected_shape() {
        let turns = [
            Turn {
                role: "system",
                content: "You are helpful.",
            },
            Turn {
                role: "user",
                content: "2+2?",
            },
        ];
        let rendered = ChatTemplate::ChatMl.render(&turns, true);
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn alpaca_render_shape() {
        let turns = [Turn {
            role: "user",
            content: "What is 2+2?",
        }];
        let rendered = ChatTemplate::Alpaca.render(&turns, true);
        assert!(rendered.contains("### Instruction:\nWhat is 2+2?"));
        assert!(rendered.ends_with("### Response:\n"));
    }

    #[test]
    fn mistral_render_shape() {
        let turns = [Turn {
            role: "user",
            content: "hi",
        }];
        let rendered = ChatTemplate::Mistral.render(&turns, true);
        assert!(rendered.starts_with("<s>"));
        assert!(rendered.contains("[INST] hi [/INST]"));
    }

    // ── emits_literal_bos regression tests ──────────────────────────────────
    //
    // Before this fix, both chat frontends passed `add_special: false`
    // unconditionally for every template. That is correct for Llama3
    // (`<|begin_of_text|>`) and Mistral (`<s>`), whose rendered output
    // already contains a literal BOS marker — but ChatML and Alpaca render
    // no such marker, so `add_special: false` there silently dropped BOS on
    // any model whose `tokenizer.ggml.add_bos_token` policy expects the
    // tokenizer to add it automatically.

    #[test]
    fn llama3_and_mistral_emit_literal_bos() {
        assert!(
            ChatTemplate::Llama3.emits_literal_bos(),
            "Llama3's rendered output starts with the literal <|begin_of_text|> marker"
        );
        assert!(
            ChatTemplate::Mistral.emits_literal_bos(),
            "Mistral's rendered output starts with the literal <s> marker"
        );
    }

    #[test]
    fn chatml_and_alpaca_do_not_emit_literal_bos() {
        assert!(
            !ChatTemplate::ChatMl.emits_literal_bos(),
            "ChatML's rendered output has no literal BOS marker — the tokenizer's own \
             add_bos_token policy must still run, so callers must NOT pass add_special: false"
        );
        assert!(
            !ChatTemplate::Alpaca.emits_literal_bos(),
            "Alpaca's rendered output has no literal BOS marker either"
        );
    }

    #[test]
    fn llama3_render_actually_starts_with_the_literal_bos_text() {
        // Cross-check emits_literal_bos() against the real render() output
        // rather than trusting the two to agree by convention.
        let rendered = ChatTemplate::Llama3.render(&[], false);
        assert!(rendered.starts_with("<|begin_of_text|>"));
    }

    #[test]
    fn chatml_render_never_contains_a_bos_marker() {
        let rendered = ChatTemplate::ChatMl.render(
            &[Turn {
                role: "user",
                content: "hi",
            }],
            true,
        );
        assert!(!rendered.contains("<|begin_of_text|>"));
        assert!(!rendered.starts_with("<s>"));
    }

    /// The exact defect this module's relocation fixes: no renderer may emit
    /// the fabricated `<|system|>` / `<|user|>` / `<|end|>` markers the
    /// server used to hardcode. Those are not part of any of the four real
    /// families, and a model that never saw them imitates them as literal
    /// output text.
    #[test]
    fn no_renderer_emits_the_fabricated_generic_markers() {
        let turns = [
            Turn {
                role: "system",
                content: "sys",
            },
            Turn {
                role: "user",
                content: "hi",
            },
        ];
        for template in [
            ChatTemplate::Llama3,
            ChatTemplate::ChatMl,
            ChatTemplate::Mistral,
            ChatTemplate::Alpaca,
        ] {
            let rendered = template.render(&turns, true);
            assert!(
                !rendered.contains("<|end|>"),
                "{template} must not emit the fabricated <|end|> marker: {rendered}"
            );
            assert!(
                !rendered.contains("<|system|>"),
                "{template} must not emit the fabricated <|system|> marker: {rendered}"
            );
            assert!(
                !rendered.contains("<|user|>"),
                "{template} must not emit the fabricated <|user|> marker: {rendered}"
            );
        }
    }
}
