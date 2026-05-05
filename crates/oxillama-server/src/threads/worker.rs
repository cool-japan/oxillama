//! Background run worker for the Assistants API.
//!
//! The worker loops over the `RunQueueReceiver`, processing one run at a time:
//!
//! 1. Transitions the run to `InProgress`.
//! 2. Reads the thread's messages and formats them as a prompt using the same
//!    chat-template logic as `routes/chat.rs`.
//! 3. Sends a `BatchRequest::Generate` to the inference engine via the shared
//!    queue in `AppState`.
//! 4. Appends the assistant's response as a new `ThreadMessage`.
//! 5. Transitions the run to `Completed`.
//!
//! On any error, the run is transitioned to `Failed` with a descriptive error
//! record attached.

use std::sync::Arc;

use tracing::{debug, error, info, warn};

use crate::queue::BatchRequest;
use crate::state::AppState;
use crate::threads::queue::RunQueueReceiver;
use crate::threads::store::ThreadStore;
use crate::threads::stream::{RunEvent, RunEventSender};
use crate::threads::types::{
    MessageRole, Run, RunError, RunStatus, RunStep, RunStepStatus, ThreadMessage,
};
use oxillama_runtime::sampling::SamplerConfig;

/// Spawn the background run worker task.
///
/// The worker runs for the lifetime of the server; it stops when the queue's
/// sender side is dropped (server shutdown).
pub fn spawn_run_worker(store: Arc<ThreadStore>, mut rx: RunQueueReceiver, state: Arc<AppState>) {
    tokio::spawn(async move {
        info!("assistants run worker started");
        while let Some(item) = rx.recv().await {
            let thread_id = item.thread_id.clone();
            let run_id = item.run_id.clone();
            debug!(thread_id, run_id, "run worker picked up item");

            let event_tx = state.run_event_tx_broadcast.clone();

            let result = process_run(
                &item.thread_id,
                &item.run_id,
                item.instructions.as_deref(),
                item.max_tokens,
                &store,
                &state,
                event_tx.as_ref(),
            )
            .await;

            if let Err(e) = result {
                error!(thread_id, run_id, error = %e, "run processing failed");
                // Best-effort: mark the run as failed.
                let store_c = Arc::clone(&store);
                let tid = item.thread_id.clone();
                let rid = item.run_id.clone();
                let err_msg = e.clone();

                // Broadcast failure event if we have a channel.
                if let Some(ref tx) = state.run_event_tx_broadcast {
                    // Attempt to get the run for the broadcast.
                    let store_for_event = Arc::clone(&store);
                    let tid_ev = tid.clone();
                    let rid_ev = rid.clone();
                    if let Some(run) = tokio::task::spawn_blocking(move || {
                        store_for_event.get_run(&tid_ev, &rid_ev)
                    })
                    .await
                    .ok()
                    .and_then(|r| r.ok())
                    {
                        let _ = tx.send(RunEvent::Failed(run));
                    }
                }

                tokio::task::spawn_blocking(move || {
                    let _ = store_c.force_update_run_status(
                        &tid,
                        &rid,
                        RunStatus::Failed,
                        Some(RunError {
                            code: "server_error".to_string(),
                            message: err_msg,
                        }),
                    );
                })
                .await
                .ok();
            }
        }
        info!("assistants run worker queue closed — exiting");
    });
}

/// Broadcast a `RunEvent` if the sender is available (best-effort; ignores lag).
fn maybe_broadcast(event_tx: Option<&RunEventSender>, event: RunEvent) {
    if let Some(tx) = event_tx {
        let _ = tx.send(event);
    }
}

/// Process a single run end-to-end.
///
/// `event_tx` is an optional broadcast sender for run lifecycle events.
/// When `Some`, lifecycle events are published for SSE streaming consumers.
async fn process_run(
    thread_id: &str,
    run_id: &str,
    instructions: Option<&str>,
    max_tokens: usize,
    store: &Arc<ThreadStore>,
    state: &Arc<AppState>,
    event_tx: Option<&RunEventSender>,
) -> Result<(), String> {
    // Step 1 — transition to InProgress.
    {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            store_c.update_run_status(&tid, &rid, RunStatus::InProgress, None)
        })
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
        .map_err(|e| format!("update InProgress: {e}"))?;
    }

    // Broadcast InProgress event.
    if event_tx.is_some() {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        if let Ok(Ok(run)) = tokio::task::spawn_blocking(move || store_c.get_run(&tid, &rid)).await
        {
            maybe_broadcast(event_tx, RunEvent::InProgress(run));
        }
    }

    // Step 2 — read messages and format prompt.
    let messages = {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        tokio::task::spawn_blocking(move || store_c.list_messages(&tid))
            .await
            .map_err(|e| format!("spawn_blocking join: {e}"))?
            .map_err(|e| format!("list_messages: {e}"))?
    };

    let prompt = format_thread_prompt(instructions, &messages);

    if prompt.is_empty() {
        warn!(
            thread_id,
            run_id, "run has empty prompt — completing with empty response"
        );
    }

    // Step 2b — create a MessageCreation RunStep (InProgress).
    let step_id = format!("step-{}", uuid::Uuid::new_v4().as_simple());
    {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        let sid = step_id.clone();
        let step = RunStep::new_message_creation(sid, rid, tid.clone());
        let step_c = step.clone();
        tokio::task::spawn_blocking(move || store_c.append_step(&tid, &step_c.run_id, &step_c))
            .await
            .map_err(|e| format!("spawn_blocking join: {e}"))?
            .map_err(|e| format!("append_step: {e}"))?;
    }

    // Step 3 — send to the inference engine.
    let (reply_tx, reply_rx) =
        tokio::sync::oneshot::channel::<Result<(String, crate::queue::UsageStats), String>>();

    let sampler = SamplerConfig::default();

    state
        .queue
        .send(BatchRequest::Generate {
            prompt,
            max_tokens,
            config: sampler,
            cache_prompt: true,
            lora_selection: vec![],
            reply: reply_tx,
        })
        .await
        .map_err(|_| "inference queue closed during run".to_string())?;

    let generated_text = match reply_rx.await {
        Ok(Ok((text, _usage))) => text,
        Ok(Err(e)) => return Err(format!("inference engine error: {e}")),
        Err(e) => return Err(format!("reply channel closed: {e}")),
    };

    // Broadcast MessageDelta event with the generated text.
    maybe_broadcast(
        event_tx,
        RunEvent::MessageDelta {
            run_id: run_id.to_string(),
            content: generated_text.clone(),
        },
    );

    // Step 4 — retrieve the run to get its ID for the assistant message.
    let run: Run = {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        tokio::task::spawn_blocking(move || store_c.get_run(&tid, &rid))
            .await
            .map_err(|e| format!("spawn_blocking join: {e}"))?
            .map_err(|e| format!("get_run: {e}"))?
    };

    // Append the assistant's message and capture the message ID.
    let assistant_msg_id = format!("msg_{}", uuid::Uuid::new_v4().as_simple());
    {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let mid = assistant_msg_id.clone();
        let assistant_msg =
            ThreadMessage::new_assistant(mid, tid.clone(), run.id.clone(), generated_text);
        tokio::task::spawn_blocking(move || store_c.append_message(&tid, &assistant_msg))
            .await
            .map_err(|e| format!("spawn_blocking join: {e}"))?
            .map_err(|e| format!("append_message: {e}"))?;
    }

    // Mark the RunStep as Completed and attach the message ID as step_details.
    {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        let sid = step_id.clone();
        let msg_id = assistant_msg_id.clone();
        tokio::task::spawn_blocking(move || {
            // Update status to Completed.
            store_c.update_step_status(&tid, &rid, &sid, RunStepStatus::Completed)?;
            // Set step_details to include the message ID.
            let mut step = store_c.get_step(&tid, &rid, &sid)?;
            step.step_details =
                Some(crate::threads::types::MessageCreationStepDetails { message_id: msg_id });
            let steps_dir = store_c.steps_dir(&tid, &rid);
            let filename = format!("{sid}.json");
            let json = serde_json::to_string_pretty(&step)
                .map_err(crate::error::ServerError::Serialization)?;
            let mut tmp = tempfile::NamedTempFile::new_in(&steps_dir).map_err(|e| {
                crate::error::ServerError::IoError {
                    context: "create temp file for step details".to_string(),
                    source: e,
                }
            })?;
            use std::io::Write as _;
            tmp.write_all(json.as_bytes())
                .map_err(|e| crate::error::ServerError::IoError {
                    context: "write step details".to_string(),
                    source: e,
                })?;
            tmp.flush()
                .map_err(|e| crate::error::ServerError::IoError {
                    context: "flush step details".to_string(),
                    source: e,
                })?;
            let target = steps_dir.join(&filename);
            tmp.persist(&target)
                .map_err(|e| crate::error::ServerError::IoError {
                    context: format!("persist step details to {}", target.display()),
                    source: e.error,
                })?;
            Ok::<(), crate::error::ServerError>(())
        })
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
        .map_err(|e| format!("update step details: {e}"))?;
    }

    // Step 5 — transition to Completed.
    {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            store_c.update_run_status(&tid, &rid, RunStatus::Completed, None)
        })
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
        .map_err(|e| format!("update Completed: {e}"))?;
    }

    // Broadcast Completed event.
    if event_tx.is_some() {
        let store_c = Arc::clone(store);
        let tid = thread_id.to_string();
        let rid = run_id.to_string();
        if let Ok(Ok(completed_run)) =
            tokio::task::spawn_blocking(move || store_c.get_run(&tid, &rid)).await
        {
            maybe_broadcast(event_tx, RunEvent::Completed(completed_run));
        }
    }

    info!(thread_id, run_id, "run completed successfully");
    Ok(())
}

/// Format the thread's messages as a single prompt string.
///
/// Mirrors the chat-template logic from `routes/chat.rs::format_chat_prompt`.
/// If `instructions` is provided it is prepended as a `system` message.
fn format_thread_prompt(instructions: Option<&str>, messages: &[ThreadMessage]) -> String {
    let mut prompt = String::new();

    if let Some(sys) = instructions {
        if !sys.is_empty() {
            prompt.push_str("<|system|>\n");
            prompt.push_str(sys);
            prompt.push_str("\n<|end|>\n");
        }
    }

    for msg in messages {
        let content = msg.text_content();
        match msg.role {
            MessageRole::User => {
                prompt.push_str("<|user|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
            MessageRole::Assistant => {
                prompt.push_str("<|assistant|>\n");
                prompt.push_str(content);
                prompt.push_str("\n<|end|>\n");
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::UsageStats;
    use crate::threads::types::{Thread, ThreadMessage};
    use std::env::temp_dir;
    use std::sync::Arc;
    use uuid::Uuid;

    fn make_store(tag: &str) -> Arc<ThreadStore> {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_thread_worker_test_{tag}_{id}"));
        Arc::new(ThreadStore::new(dir).expect("ThreadStore::new"))
    }

    fn make_state_with_mock_worker() -> (Arc<AppState>, tokio::task::JoinHandle<()>) {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);

        let handle = tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                if let BatchRequest::Generate { reply, .. } = req {
                    let _ = reply.send(Ok((
                        "mock assistant response".to_string(),
                        UsageStats {
                            prompt_tokens: 5,
                            completion_tokens: 4,
                            total_tokens: 9,
                        },
                    )));
                }
            }
        });

        let state = Arc::new(AppState::new(
            tx,
            "test-model".to_string(),
            oxillama_runtime::sampling::SamplerConfig::default(),
            None,
            0,
        ));

        (state, handle)
    }

    #[test]
    fn format_thread_prompt_with_instructions() {
        let msgs = vec![ThreadMessage::new_user(
            "m1".into(),
            "t1".into(),
            "hi there".into(),
        )];
        let prompt = format_thread_prompt(Some("Be helpful."), &msgs);
        assert!(prompt.contains("<|system|>"), "should have system block");
        assert!(prompt.contains("Be helpful."));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("hi there"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn format_thread_prompt_without_instructions() {
        let msgs = vec![ThreadMessage::new_user(
            "m1".into(),
            "t1".into(),
            "question".into(),
        )];
        let prompt = format_thread_prompt(None, &msgs);
        assert!(!prompt.contains("<|system|>"));
        assert!(prompt.contains("question"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn format_thread_prompt_mixed_roles() {
        let msgs = vec![
            ThreadMessage::new_user("m1".into(), "t1".into(), "hello".into()),
            ThreadMessage::new_assistant("m2".into(), "t1".into(), "run_1".into(), "hi".into()),
            ThreadMessage::new_user("m3".into(), "t1".into(), "follow up".into()),
        ];
        let prompt = format_thread_prompt(None, &msgs);
        // All three messages + trailing assistant prompt.
        assert_eq!(prompt.matches("<|user|>").count(), 2);
        assert_eq!(prompt.matches("<|assistant|>").count(), 2); // 1 from history + 1 trailing
    }

    #[tokio::test]
    async fn worker_processes_run_to_completed() {
        let store = make_store("worker_complete");
        let (state, _worker_handle) = make_state_with_mock_worker();

        // Set up thread and run.
        let thread = Thread {
            id: "thread_wc".to_string(),
            object: "thread".to_string(),
            created_at: 0,
            metadata: serde_json::json!({}),
        };
        store.create_thread(&thread).expect("create thread");

        let msg = ThreadMessage::new_user("msg_1".into(), "thread_wc".into(), "hello".into());
        store.append_message("thread_wc", &msg).expect("append");

        let run = Run {
            id: "run_wc".to_string(),
            object: "thread.run".to_string(),
            created_at: 0,
            thread_id: "thread_wc".to_string(),
            status: RunStatus::Queued,
            model: "test-model".to_string(),
            last_error: None,
        };
        store.create_run("thread_wc", &run).expect("create run");

        // Spawn worker and submit item.
        let (tx, rx) = crate::threads::queue::new_run_queue();
        spawn_run_worker(Arc::clone(&store), rx, Arc::clone(&state));

        tx.send(crate::threads::queue::RunWorkItem {
            thread_id: "thread_wc".to_string(),
            run_id: "run_wc".to_string(),
            model: None,
            instructions: None,
            max_tokens: 64,
        })
        .expect("send work item");

        // Poll until terminal state.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let current_run = store.get_run("thread_wc", "run_wc").expect("get run");
            if current_run.status.is_terminal() {
                assert_eq!(current_run.status, RunStatus::Completed);
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("run did not complete within deadline");
            }
        }

        // Verify assistant message was appended.
        let msgs = store.list_messages("thread_wc").expect("list messages");
        assert_eq!(msgs.len(), 2, "should have user + assistant message");
        assert_eq!(msgs[1].role, MessageRole::Assistant);
        assert_eq!(msgs[1].text_content(), "mock assistant response");
    }
}
