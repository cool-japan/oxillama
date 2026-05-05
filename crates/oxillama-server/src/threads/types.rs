//! OpenAI Assistants v2-compatible request/response types.
//!
//! This module defines the wire-format structs used by the Assistants API:
//! threads, messages, runs, and the request bodies that create them.
//!
//! Field names and `object` strings match the OpenAI v2 specification so
//! that clients written against the official SDK work without modification.

use serde::{Deserialize, Serialize};

// ── Thread ────────────────────────────────────────────────────────────────────

/// An OpenAI-compatible thread object (persisted to disk).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thread {
    /// Stable identifier (`thread_<uuid>`).
    pub id: String,
    /// Always `"thread"`.
    pub object: String,
    /// Unix timestamp (seconds) when the thread was created.
    pub created_at: i64,
    /// Caller-supplied free-form metadata (JSON object).
    pub metadata: serde_json::Value,
}

// ── Message types ─────────────────────────────────────────────────────────────

/// Role of the message author.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User-authored message.
    User,
    /// Assistant-authored message.
    Assistant,
}

/// Annotation placeholder inside a `TextContent` (reserved for future use).
pub type Annotation = serde_json::Value;

/// Text payload of a content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    /// The text string.
    pub value: String,
    /// Annotations (empty for now).
    pub annotations: Vec<Annotation>,
}

/// A single content block within a message.
///
/// Currently only the `text` type is supported; the `type` field is always `"text"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    /// Content type — always `"text"` in this implementation.
    pub r#type: String,
    /// The text payload.
    pub text: TextContent,
}

/// A message stored inside a thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMessage {
    /// Stable identifier (`msg_<uuid>`).
    pub id: String,
    /// Always `"thread.message"`.
    pub object: String,
    /// Unix timestamp (seconds) when the message was created.
    pub created_at: i64,
    /// ID of the owning thread.
    pub thread_id: String,
    /// Role of the message author.
    pub role: MessageRole,
    /// Message content blocks.
    pub content: Vec<ContentBlock>,
    /// Run ID that produced this message (`None` for user messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

impl ThreadMessage {
    /// Construct a new user message.
    pub fn new_user(id: String, thread_id: String, content: String) -> Self {
        Self {
            id,
            object: "thread.message".to_string(),
            created_at: unix_now(),
            thread_id,
            role: MessageRole::User,
            content: vec![ContentBlock {
                r#type: "text".to_string(),
                text: TextContent {
                    value: content,
                    annotations: vec![],
                },
            }],
            run_id: None,
        }
    }

    /// Construct a new assistant message produced by a run.
    pub fn new_assistant(id: String, thread_id: String, run_id: String, content: String) -> Self {
        Self {
            id,
            object: "thread.message".to_string(),
            created_at: unix_now(),
            thread_id,
            role: MessageRole::Assistant,
            content: vec![ContentBlock {
                r#type: "text".to_string(),
                text: TextContent {
                    value: content,
                    annotations: vec![],
                },
            }],
            run_id: Some(run_id),
        }
    }

    /// Extract the plain text value from the first content block.
    pub fn text_content(&self) -> &str {
        self.content
            .first()
            .map(|b| b.text.value.as_str())
            .unwrap_or("")
    }
}

// ── Run types ─────────────────────────────────────────────────────────────────

/// Lifecycle status of a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Submitted, not yet picked up.
    Queued,
    /// Currently processing.
    InProgress,
    /// Finished successfully.
    Completed,
    /// Cancelled by a user request.
    Cancelled,
    /// Failed with an error.
    Failed,
    /// Timed out before completion.
    Expired,
}

impl RunStatus {
    /// Whether this status represents a terminal (non-resumable) state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            RunStatus::Completed | RunStatus::Cancelled | RunStatus::Failed | RunStatus::Expired
        )
    }
}

/// A structured error attached to a failed run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunError {
    /// Short machine-readable code (e.g. `"server_error"`).
    pub code: String,
    /// Human-readable description.
    pub message: String,
}

/// A run object — one inference job against the messages in a thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Stable identifier (`run_<uuid>`).
    pub id: String,
    /// Always `"thread.run"`.
    pub object: String,
    /// Unix timestamp (seconds) when the run was created.
    pub created_at: i64,
    /// ID of the thread this run belongs to.
    pub thread_id: String,
    /// Current lifecycle status.
    pub status: RunStatus,
    /// Model override (empty = use server default).
    pub model: String,
    /// Error details if `status` is `Failed` or `Cancelled`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<RunError>,
}

// ── Run Step types ────────────────────────────────────────────────────────────

/// Type of a run step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStepType {
    /// The step created an assistant message.
    MessageCreation,
    /// The step invoked one or more tools.
    ToolCalls,
}

/// Lifecycle status of a run step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStepStatus {
    /// Step is currently executing.
    InProgress,
    /// Step finished successfully.
    Completed,
    /// Step failed with an error.
    Failed,
    /// Step was cancelled.
    Cancelled,
}

impl RunStepStatus {
    /// Whether this status represents a terminal (non-resumable) state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            RunStepStatus::Completed | RunStepStatus::Failed | RunStepStatus::Cancelled
        )
    }
}

/// Details attached to a `MessageCreation` step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageCreationStepDetails {
    /// The ID of the message created by this step.
    pub message_id: String,
}

/// A single step within a run.
///
/// Exposes the sub-task breakdown of a run to clients: for the current
/// implementation each run produces exactly one `MessageCreation` step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStep {
    /// Stable identifier (`step-<uuid>`).
    pub id: String,
    /// Always `"thread.run.step"`.
    pub object: String,
    /// ID of the owning run.
    pub run_id: String,
    /// ID of the owning thread.
    pub thread_id: String,
    /// Type of this step.
    pub step_type: RunStepType,
    /// Current lifecycle status.
    pub status: RunStepStatus,
    /// Unix timestamp when the step was created.
    pub created_at: u64,
    /// Unix timestamp when the step completed successfully, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    /// Unix timestamp when the step failed, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_at: Option<u64>,
    /// Human-readable error message when `status` is `Failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Details for `MessageCreation` steps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_details: Option<MessageCreationStepDetails>,
}

impl RunStep {
    /// Create a new `MessageCreation` step in `InProgress` state.
    pub fn new_message_creation(step_id: String, run_id: String, thread_id: String) -> Self {
        Self {
            id: step_id,
            object: "thread.run.step".to_string(),
            run_id,
            thread_id,
            step_type: RunStepType::MessageCreation,
            status: RunStepStatus::InProgress,
            created_at: unix_now() as u64,
            completed_at: None,
            failed_at: None,
            error: None,
            step_details: None,
        }
    }
}

// ── Request bodies ────────────────────────────────────────────────────────────

/// Request body for `POST /v1/threads/:thread_id/messages`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessageRequest {
    /// Role of the message author (must be `user` for API-created messages).
    pub role: MessageRole,
    /// Plain text content of the message.
    pub content: String,
}

/// Request body for `POST /v1/threads`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateThreadRequest {
    /// Optional initial messages to seed the thread with.
    #[serde(default)]
    pub messages: Option<Vec<CreateMessageRequest>>,
    /// Optional caller-supplied metadata (arbitrary JSON object).
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Request body for `POST /v1/threads/:thread_id/runs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRunRequest {
    /// Optional model override.  When absent the server's default model is used.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional system-level instructions prepended to the thread context.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Maximum tokens to generate.  Defaults to 512 when absent.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// When `true`, emit SSE events instead of returning the run object directly.
    #[serde(default)]
    pub stream: bool,
}

// ── Shared timestamp helper ───────────────────────────────────────────────────

pub(crate) fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_status_terminal_set_is_correct() {
        assert!(RunStatus::Completed.is_terminal());
        assert!(RunStatus::Cancelled.is_terminal());
        assert!(RunStatus::Failed.is_terminal());
        assert!(RunStatus::Expired.is_terminal());
        assert!(!RunStatus::Queued.is_terminal());
        assert!(!RunStatus::InProgress.is_terminal());
    }

    #[test]
    fn thread_message_new_user_sets_fields() {
        let msg = ThreadMessage::new_user("msg_1".into(), "thread_1".into(), "hello".into());
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.text_content(), "hello");
        assert!(msg.run_id.is_none());
    }

    #[test]
    fn thread_message_new_assistant_sets_run_id() {
        let msg = ThreadMessage::new_assistant(
            "msg_2".into(),
            "thread_1".into(),
            "run_1".into(),
            "hi!".into(),
        );
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.run_id, Some("run_1".into()));
    }

    #[test]
    fn run_status_serde_roundtrip() {
        let s = serde_json::to_string(&RunStatus::InProgress).expect("serialize");
        let d: RunStatus = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(d, RunStatus::InProgress);
    }

    #[test]
    fn message_role_serde_lowercase() {
        let json = serde_json::to_string(&MessageRole::User).expect("serialize");
        assert_eq!(json, r#""user""#);
        let back: MessageRole = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, MessageRole::User);
    }
}
