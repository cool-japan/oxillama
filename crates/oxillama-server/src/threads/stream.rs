//! SSE streaming for run creation.
//!
//! When `POST /v1/threads/:id/runs` receives `"stream": true`, the route
//! handler subscribes to the broadcast channel and returns an SSE stream
//! that emits events as the run progresses through its lifecycle:
//!
//! ```text
//! event: thread.run.created
//! data: {"id":"run-xxx","status":"queued",...}
//!
//! event: thread.run.in_progress
//! data: {"id":"run-xxx","status":"in_progress",...}
//!
//! event: thread.message.delta
//! data: {"delta":{"content":[{"type":"text","text":{"value":"..."}}]}}
//!
//! event: thread.run.completed
//! data: {"id":"run-xxx","status":"completed",...}
//!
//! event: done
//! data: [DONE]
//! ```
//!
//! Implementation details:
//! - The broadcast channel capacity is 256 events.
//! - The run worker acquires a sender from `AppState.run_event_tx` and
//!   broadcasts each lifecycle event.
//! - The SSE handler subscribes after the run is created and forwards events
//!   until it sees a terminal `RunEvent` or the channel closes.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::{self, Stream, StreamExt as _};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::threads::types::Run;

/// Capacity of the run-event broadcast channel.
pub const RUN_EVENT_BROADCAST_CAPACITY: usize = 256;

/// All events that can be emitted during a run's lifecycle.
///
/// The worker broadcasts these; the SSE handler (or other consumers) subscribe
/// to receive them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "data", rename_all = "snake_case")]
pub enum RunEvent {
    /// Run was created and is now in the queue.
    Created(Run),
    /// Run has been picked up and inference is in progress.
    InProgress(Run),
    /// A chunk of the assistant's response is available.
    MessageDelta {
        /// The run that generated this delta.
        run_id: String,
        /// The text chunk.
        content: String,
    },
    /// Run completed successfully.
    Completed(Run),
    /// Run failed with an error.
    Failed(Run),
}

impl RunEvent {
    /// Return the SSE event name for this variant.
    pub fn sse_event_name(&self) -> &'static str {
        match self {
            RunEvent::Created(_) => "thread.run.created",
            RunEvent::InProgress(_) => "thread.run.in_progress",
            RunEvent::MessageDelta { .. } => "thread.message.delta",
            RunEvent::Completed(_) => "thread.run.completed",
            RunEvent::Failed(_) => "thread.run.failed",
        }
    }

    /// Whether this event represents a terminal state (no more events follow).
    pub fn is_terminal(&self) -> bool {
        matches!(self, RunEvent::Completed(_) | RunEvent::Failed(_))
    }

    /// The run ID associated with this event.
    pub fn run_id(&self) -> &str {
        match self {
            RunEvent::Created(r)
            | RunEvent::InProgress(r)
            | RunEvent::Completed(r)
            | RunEvent::Failed(r) => &r.id,
            RunEvent::MessageDelta { run_id, .. } => run_id,
        }
    }
}

/// A broadcast sender for run events.
///
/// Stored in `AppState.run_event_tx`.  The worker clones this to send events;
/// route handlers subscribe with `.subscribe()`.
pub type RunEventSender = Arc<broadcast::Sender<RunEvent>>;

/// Create a new run-event broadcast channel.
pub fn new_run_event_channel() -> (RunEventSender, broadcast::Receiver<RunEvent>) {
    let (tx, rx) = broadcast::channel(RUN_EVENT_BROADCAST_CAPACITY);
    (Arc::new(tx), rx)
}

/// Build an SSE stream that forwards `RunEvent`s for a specific run.
///
/// The stream subscribes to `event_tx` and filters events by `run_id`.
/// It emits events until it sees a terminal event (`Completed` / `Failed`),
/// then appends a `[DONE]` sentinel and closes.
///
/// Returns an `axum::response::Sse` response.
pub fn build_run_sse_stream(
    event_tx: &RunEventSender,
    run_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = event_tx.subscribe();

    // Use `stream::unfold` with a state machine to avoid lifetime issues
    // that arise from async closures capturing mutable references.
    // State: (receiver, done_flag)
    let event_stream = stream::unfold(
        (BroadcastStream::new(rx), false),
        move |(mut stream, done)| {
            let run_id_inner = run_id.clone();
            async move {
                if done {
                    // Stream exhausted; emit [DONE] one more time and stop.
                    return None;
                }
                // Pull the next item from the broadcast stream.
                match stream.next().await {
                    None => {
                        // Channel closed without a terminal event.
                        let done_event =
                            Ok::<Event, Infallible>(Event::default().event("done").data("[DONE]"));
                        Some((done_event, (stream, true)))
                    }
                    Some(result) => {
                        let event = match result {
                            Ok(e) => e,
                            Err(_) => {
                                // Lagged; skip this item and continue.
                                let placeholder = Ok::<Event, Infallible>(
                                    Event::default().event("keep-alive").data(""),
                                );
                                return Some((placeholder, (stream, false)));
                            }
                        };
                        // Skip events for other runs.
                        if event.run_id() != run_id_inner {
                            // Return a comment-style keep-alive that clients ignore.
                            let placeholder = Ok::<Event, Infallible>(
                                Event::default().event("keep-alive").data(""),
                            );
                            return Some((placeholder, (stream, false)));
                        }
                        let is_terminal = event.is_terminal();
                        let event_name = event.sse_event_name();
                        let data = match serde_json::to_string(&event) {
                            Ok(s) => s,
                            Err(_) => {
                                let placeholder = Ok::<Event, Infallible>(
                                    Event::default().event("keep-alive").data(""),
                                );
                                return Some((placeholder, (stream, false)));
                            }
                        };
                        let sse_event = Ok(Event::default().event(event_name).data(data));
                        // If terminal, mark done so the next poll returns None.
                        Some((sse_event, (stream, is_terminal)))
                    }
                }
            }
        },
    );

    Sse::new(event_stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threads::types::{Run, RunStatus};

    fn make_run(id: &str) -> Run {
        Run {
            id: id.to_string(),
            object: "thread.run".to_string(),
            created_at: 1_000_000,
            thread_id: "thread_test".to_string(),
            status: RunStatus::Queued,
            model: "test-model".to_string(),
            last_error: None,
        }
    }

    /// `RunEvent::Created` serializes with the correct event_type discriminant.
    #[test]
    fn run_event_serialize_created() {
        let run = make_run("run_create_test");
        let event = RunEvent::Created(run.clone());

        let json_str = serde_json::to_string(&event).expect("serialize");
        let val: serde_json::Value = serde_json::from_str(&json_str).expect("parse");

        assert_eq!(val["event_type"], "created");
        // The data field contains the run object.
        assert_eq!(val["data"]["id"], run.id);
        assert_eq!(val["data"]["status"], "queued");
    }

    /// `RunEvent::Completed` serializes with status set to "completed".
    #[test]
    fn run_event_serialize_completed() {
        let mut run = make_run("run_complete_test");
        run.status = RunStatus::Completed;
        let event = RunEvent::Completed(run.clone());

        let json_str = serde_json::to_string(&event).expect("serialize");
        let val: serde_json::Value = serde_json::from_str(&json_str).expect("parse");

        assert_eq!(val["event_type"], "completed");
        assert_eq!(val["data"]["id"], run.id);
        assert_eq!(val["data"]["status"], "completed");
    }

    /// `RunEvent::MessageDelta` has the correct format.
    #[test]
    fn run_event_message_delta() {
        let event = RunEvent::MessageDelta {
            run_id: "run_delta_test".to_string(),
            content: "Hello, world!".to_string(),
        };

        let json_str = serde_json::to_string(&event).expect("serialize");
        let val: serde_json::Value = serde_json::from_str(&json_str).expect("parse");

        assert_eq!(val["event_type"], "message_delta");
        assert_eq!(val["data"]["run_id"], "run_delta_test");
        assert_eq!(val["data"]["content"], "Hello, world!");
    }

    /// `is_terminal` returns true only for Completed and Failed.
    #[test]
    fn run_event_is_terminal_variants() {
        let run = make_run("r");
        assert!(!RunEvent::Created(run.clone()).is_terminal());
        assert!(!RunEvent::InProgress(run.clone()).is_terminal());
        assert!(!RunEvent::MessageDelta {
            run_id: "r".into(),
            content: "x".into()
        }
        .is_terminal());
        assert!(RunEvent::Completed(run.clone()).is_terminal());
        assert!(RunEvent::Failed(run).is_terminal());
    }

    /// `sse_event_name` returns the correct OpenAI-compatible event names.
    #[test]
    fn run_event_sse_names() {
        let run = make_run("r");
        assert_eq!(
            RunEvent::Created(run.clone()).sse_event_name(),
            "thread.run.created"
        );
        assert_eq!(
            RunEvent::InProgress(run.clone()).sse_event_name(),
            "thread.run.in_progress"
        );
        assert_eq!(
            RunEvent::MessageDelta {
                run_id: "r".into(),
                content: "x".into()
            }
            .sse_event_name(),
            "thread.message.delta"
        );
        assert_eq!(
            RunEvent::Completed(run.clone()).sse_event_name(),
            "thread.run.completed"
        );
        assert_eq!(RunEvent::Failed(run).sse_event_name(), "thread.run.failed");
    }

    /// The broadcast channel can send and receive `RunEvent`s.
    #[tokio::test]
    async fn run_event_channel_roundtrip() {
        let (tx, mut rx) = new_run_event_channel();
        let run = make_run("run_channel");
        let event = RunEvent::Created(run.clone());

        tx.send(event).expect("send");
        let received = rx.recv().await.expect("recv");

        if let RunEvent::Created(r) = received {
            assert_eq!(r.id, run.id);
        } else {
            panic!("unexpected event type");
        }
    }
}
