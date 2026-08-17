//! Chat template rendering shared by the `chat` REPL and the TUI.
//!
//! ## Where the logic lives now
//!
//! The model-agnostic half — [`ChatTemplate`], [`Turn`], the family detection
//! (`detect_from_jinja` / `detect_from_vocab` / `resolve` /
//! `resolve_from_path`) and the four renderers — moved to
//! [`oxillama_runtime::chat_template`] so `oxillama-server` can share exactly
//! one implementation with the CLI. The server previously had no way to reach
//! it (it depends on `oxillama-runtime`, never on `oxillama-cli`) and so
//! rendered every `/v1/chat/completions` request through a hardcoded, fake
//! template; see that module's header for the full story.
//!
//! This module re-exports those types unchanged and keeps only the two
//! helpers that are coupled to the CLI's own [`SessionSnapshot`] transcript
//! type, which the server has no equivalent of.

use crate::session::{ChatMessage, SessionSnapshot};

pub use oxillama_runtime::chat_template::{ChatTemplate, Turn};

/// Ensure `session.messages[0]` is exactly the system message described by
/// `system_prompt` (replacing any previous leading system message), or
/// contains no leading system message when `system_prompt` is `None`.
///
/// Both chat frontends call this — at startup (`--system` / a profile's
/// `system_prompt`), on `/system <text>`, and when resetting/clearing a
/// conversation while keeping the system prompt — so a system prompt is
/// recorded exactly **once** in the transcript. Before this fix the plain
/// REPL instead re-prepended the system text to the prompt string on every
/// turn (main.rs's old `format!("{sys}\n\nUser: {input}\nAssistant:")`),
/// which injected it once per turn into the ever-growing context.
pub fn seed_system_message(session: &mut SessionSnapshot, system_prompt: &Option<String>) {
    let has_leading_system = session.messages.first().is_some_and(|m| m.role == "system");
    if has_leading_system {
        session.messages.remove(0);
    }
    if let Some(text) = system_prompt {
        session.messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: text.clone(),
            },
        );
    }
}

/// Render a full [`SessionSnapshot`] through `template`, eliciting an
/// assistant reply at the end.
///
/// Any trailing empty-content assistant message (the TUI's streaming
/// placeholder) is skipped — it is the render *target*, not input.
pub fn render_session(session: &SessionSnapshot, template: ChatTemplate) -> String {
    let turns: Vec<Turn<'_>> = session
        .messages
        .iter()
        .filter(|m| !(m.role == "assistant" && m.content.is_empty()))
        .map(|m: &ChatMessage| Turn {
            role: &m.role,
            content: &m.content,
        })
        .collect();
    template.render(&turns, true)
}

// ── Tests ────────────────────────────────────────────────────────────────────
//
// The detection/rendering tests moved with the code into
// `oxillama-runtime/src/chat_template.rs`.  What stays here covers (a) the two
// `SessionSnapshot`-coupled helpers that did *not* move, and (b) a thin
// re-export smoke check, so a future refactor that silently drops the
// `pub use` or changes a rendered shape is caught on the CLI side too — the
// CLI's observable behaviour must not change just because the code moved
// crates.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reexported_types_still_detect_and_render() {
        // Same fingerprints the CLI relied on before the move.
        assert_eq!(
            ChatTemplate::detect_from_jinja("{{ '<|start_header_id|>' }}"),
            Some(ChatTemplate::Llama3)
        );
        assert_eq!(
            ChatTemplate::detect_from_jinja("{{ '<|im_start|>' }}"),
            Some(ChatTemplate::ChatMl)
        );
        let rendered = ChatTemplate::ChatMl.render(
            &[Turn {
                role: "user",
                content: "hi",
            }],
            true,
        );
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn reexported_emits_literal_bos_still_drives_add_special() {
        // main.rs builds `add_special: !template.emits_literal_bos()`.
        assert!(ChatTemplate::Llama3.emits_literal_bos());
        assert!(!ChatTemplate::ChatMl.emits_literal_bos());
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
    fn render_session_skips_empty_assistant_placeholder() {
        let mut session = SessionSnapshot::new("test");
        session.messages.push(ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        });
        session.messages.push(ChatMessage {
            role: "assistant".into(),
            content: String::new(),
        });
        let rendered = render_session(&session, ChatTemplate::ChatMl);
        assert!(rendered.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_session_includes_completed_turns() {
        let mut session = SessionSnapshot::new("test");
        session.messages.push(ChatMessage {
            role: "user".into(),
            content: "Turn 1".into(),
        });
        session.messages.push(ChatMessage {
            role: "assistant".into(),
            content: "Reply 1".into(),
        });
        let rendered = render_session(&session, ChatTemplate::Llama3);
        assert!(rendered.contains("Turn 1"));
        assert!(rendered.contains("Reply 1"));
    }

    #[test]
    fn seed_system_message_inserts_once() {
        let mut session = SessionSnapshot::new("test");
        seed_system_message(&mut session, &Some("Be terse.".to_string()));
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].role, "system");
        assert_eq!(session.messages[0].content, "Be terse.");
    }

    #[test]
    fn seed_system_message_replaces_previous_not_duplicates() {
        let mut session = SessionSnapshot::new("test");
        seed_system_message(&mut session, &Some("First.".to_string()));
        seed_system_message(&mut session, &Some("Second.".to_string()));
        assert_eq!(
            session.messages.len(),
            1,
            "re-seeding must replace, not accumulate, the system message"
        );
        assert_eq!(session.messages[0].content, "Second.");
    }

    #[test]
    fn seed_system_message_none_clears_leading_system() {
        let mut session = SessionSnapshot::new("test");
        seed_system_message(&mut session, &Some("Be terse.".to_string()));
        seed_system_message(&mut session, &None);
        assert!(session.messages.is_empty());
    }

    #[test]
    fn seed_system_message_preserves_other_turns() {
        let mut session = SessionSnapshot::new("test");
        session.messages.push(ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        });
        seed_system_message(&mut session, &Some("Be terse.".to_string()));
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, "system");
        assert_eq!(session.messages[1].role, "user");
    }

    /// The system prompt must be rendered exactly once no matter how many
    /// turns follow — the REPL regression that `seed_system_message` exists
    /// to prevent, asserted end-to-end through the re-exported renderer.
    #[test]
    fn seeded_system_prompt_renders_exactly_once() {
        let mut session = SessionSnapshot::new("test");
        seed_system_message(&mut session, &Some("Be terse.".to_string()));
        for i in 0..3 {
            session.messages.push(ChatMessage {
                role: "user".into(),
                content: format!("q{i}"),
            });
            session.messages.push(ChatMessage {
                role: "assistant".into(),
                content: format!("a{i}"),
            });
        }
        let rendered = render_session(&session, ChatTemplate::Llama3);
        assert_eq!(rendered.matches("Be terse.").count(), 1);
    }
}
