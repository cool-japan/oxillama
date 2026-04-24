//! Background batch processing worker.
//!
//! The `BatchWorker` runs as a dedicated Tokio task.  It receives job IDs
//! from the `BatchQueue`, reads the input JSONL from disk line-by-line, sends
//! each line as a generate request to the inference engine via the existing
//! `BatchRequest` queue, collects results, and writes output JSONL back to
//! disk atomically.
//!
//! The worker checks `cancel_requested` in `status.json` after every line and
//! marks remaining lines as cancelled if the flag is set.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::batch_spool::queue::BatchQueueReceiver;
use crate::batch_spool::store::{BatchJobStatus, BatchStore};
use crate::queue::{BatchRequest, UsageStats};

/// Spawn the batch background worker task.
///
/// - `rx` — work-item receiver from `BatchQueue`.
/// - `inference_tx` — sender into the inference queue (shared with route handlers).
/// - `store` — disk-backed store reference.
pub fn spawn_batch_worker(
    mut rx: BatchQueueReceiver,
    inference_tx: mpsc::Sender<BatchRequest>,
    store: Arc<BatchStore>,
) {
    tokio::spawn(async move {
        info!("batch worker started");
        while let Some(item) = rx.recv().await {
            let job_id = item.job_id.clone();
            debug!(job_id, "batch worker picked up job");

            if let Err(e) = process_job(&job_id, &inference_tx, &store).await {
                error!(job_id, error = %e, "batch job failed");
                // Best-effort: mark as failed on disk.
                if let Ok(mut meta) = store.read_status(&job_id) {
                    meta.status = BatchJobStatus::Failed;
                    meta.updated_at = unix_now();
                    let _ = store.update_status(&job_id, &meta);
                }
            }
        }
        info!("batch worker queue closed — exiting");
    });
}

/// Process a single batch job end-to-end.
async fn process_job(
    job_id: &str,
    inference_tx: &mpsc::Sender<BatchRequest>,
    store: &Arc<BatchStore>,
) -> Result<(), String> {
    // Read input lines.
    let input_lines = tokio::task::spawn_blocking({
        let store = Arc::clone(store);
        let job_id = job_id.to_string();
        move || store.read_input_lines(&job_id)
    })
    .await
    .map_err(|e| format!("spawn_blocking join error: {e}"))?
    .map_err(|e| format!("read_input_lines: {e}"))?;

    // Update status to InProgress.
    {
        let store = Arc::clone(store);
        let job_id = job_id.to_string();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            let mut meta = store.read_status(&job_id)?;
            meta.status = BatchJobStatus::InProgress;
            meta.updated_at = unix_now();
            store.update_status(&job_id, &meta)
        })
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
        .map_err(|e| format!("update InProgress: {e}"))?;
    }

    let total = input_lines.len() as u32;
    let mut completed = 0_u32;
    let mut failed = 0_u32;

    for (line_idx, line) in input_lines.iter().enumerate() {
        // Check cancel flag before each line.
        {
            let store_c = Arc::clone(store);
            let job_id_c = job_id.to_string();
            let cancelled = tokio::task::spawn_blocking(move || -> bool {
                store_c
                    .read_status(&job_id_c)
                    .map(|m| m.cancel_requested)
                    .unwrap_or(false)
            })
            .await
            .unwrap_or(false);

            if cancelled {
                // Mark all remaining lines as cancelled.
                let remaining = total - completed - failed;
                let store_c = Arc::clone(store);
                let job_id_c = job_id.to_string();
                tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                    let mut meta = store_c.read_status(&job_id_c)?;
                    meta.status = BatchJobStatus::Cancelled;
                    meta.failed_lines += remaining;
                    meta.updated_at = unix_now();
                    store_c.update_status(&job_id_c, &meta)
                })
                .await
                .ok();

                // Write cancelled markers for remaining lines.
                for i in (line_idx as u32)..total {
                    let cancelled_line = serde_json::json!({
                        "custom_id": format!("line-{i}"),
                        "status": "cancelled",
                    });
                    let store_c = Arc::clone(store);
                    let job_id_c = job_id.to_string();
                    let line_str = cancelled_line.to_string();
                    tokio::task::spawn_blocking(move || {
                        let _ = store_c.append_output(&job_id_c, &line_str);
                    })
                    .await
                    .ok();
                }
                return Ok(());
            }
        }

        // Parse the line as a BatchRequestItem.
        let request_body = match parse_request_line(line) {
            Ok(body) => body,
            Err(e) => {
                warn!(
                    job_id,
                    line = line_idx,
                    error = %e,
                    "skipping malformed batch input line"
                );
                let error_record = serde_json::json!({
                    "custom_id": format!("line-{line_idx}"),
                    "error": e,
                });
                let store_c = Arc::clone(store);
                let job_id_c = job_id.to_string();
                let line_str = error_record.to_string();
                tokio::task::spawn_blocking(move || {
                    let _ = store_c.append_error(&job_id_c, &line_str);
                })
                .await
                .ok();
                failed += 1;
                continue;
            }
        };

        // Submit to the inference queue and wait for the result.
        let (reply_tx, reply_rx) =
            tokio::sync::oneshot::channel::<Result<(String, UsageStats), String>>();

        let prompt = extract_prompt(&request_body);
        let max_tokens = extract_max_tokens(&request_body);
        let sampler = oxillama_runtime::sampling::SamplerConfig::default();

        if inference_tx
            .send(BatchRequest::Generate {
                prompt,
                max_tokens,
                config: sampler,
                reply: reply_tx,
            })
            .await
            .is_err()
        {
            error!(
                job_id,
                line = line_idx,
                "inference queue closed during batch"
            );
            failed += 1;
            continue;
        }

        let result = match reply_rx.await {
            Ok(r) => r,
            Err(e) => {
                error!(job_id, line = line_idx, error = %e, "reply channel closed");
                failed += 1;
                continue;
            }
        };

        let (output_line, success) = match result {
            Ok((text, usage)) => {
                let record = serde_json::json!({
                    "custom_id": format!("line-{line_idx}"),
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"role": "assistant", "content": text}}],
                            "usage": {
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.total_tokens,
                            }
                        }
                    }
                });
                (record.to_string(), true)
            }
            Err(e) => {
                let record = serde_json::json!({
                    "custom_id": format!("line-{line_idx}"),
                    "error": e,
                });
                (record.to_string(), false)
            }
        };

        if success {
            completed += 1;
        } else {
            failed += 1;
        }

        // Append output line and update status.
        {
            let store_c = Arc::clone(store);
            let job_id_c = job_id.to_string();
            let line_str = output_line.clone();
            tokio::task::spawn_blocking(move || {
                let _ = store_c.append_output(&job_id_c, &line_str);
            })
            .await
            .ok();
        }

        {
            let store_c = Arc::clone(store);
            let job_id_c = job_id.to_string();
            let c = completed;
            let f = failed;
            tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                let mut meta = store_c.read_status(&job_id_c)?;
                meta.completed_lines = c;
                meta.failed_lines = f;
                meta.updated_at = unix_now();
                store_c.update_status(&job_id_c, &meta)
            })
            .await
            .ok();
        }
    }

    // Finalize job status.
    {
        let store_c = Arc::clone(store);
        let job_id_c = job_id.to_string();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            let mut meta = store_c.read_status(&job_id_c)?;
            meta.status = if meta.failed_lines == 0 {
                BatchJobStatus::Completed
            } else {
                BatchJobStatus::Failed
            };
            meta.completed_lines = completed;
            meta.failed_lines = failed;
            meta.updated_at = unix_now();
            store_c.update_status(&job_id_c, &meta)
        })
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
        .map_err(|e| format!("finalize status: {e}"))?;
    }

    info!(job_id, completed, failed, "batch job complete");
    Ok(())
}

/// Parse a JSONL line into a request body `Value`.
fn parse_request_line(line: &str) -> Result<Value, String> {
    serde_json::from_str(line.trim()).map_err(|e| format!("JSON parse error: {e}"))
}

/// Extract a prompt string from a batch request body.
///
/// Supports both `{ "prompt": "..." }` (completions) and
/// `{ "messages": [...] }` (chat completions) formats.
fn extract_prompt(body: &Value) -> String {
    // Chat completions format.
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        let mut prompt = String::new();
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            prompt.push_str(&format!("<|{role}|>\n{content}\n<|end|>\n"));
        }
        prompt.push_str("<|assistant|>\n");
        return prompt;
    }
    // Completions format.
    body.get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string()
}

/// Extract `max_tokens` from a batch request body, defaulting to 256.
fn extract_max_tokens(body: &Value) -> usize {
    body.get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(256)
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_spool::store::BatchStore;
    use crate::queue::UsageStats;
    use std::env::temp_dir;

    fn temp_store(tag: &str) -> Arc<BatchStore> {
        let id = uuid::Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_batch_worker_test_{tag}_{id}"));
        Arc::new(BatchStore::new(dir).expect("store"))
    }

    /// (a) worker_processes_all_lines — submit a 5-line batch with a mock
    ///     inference engine; output.jsonl should have 5 lines when complete.
    #[tokio::test]
    async fn worker_processes_all_lines() {
        let store = temp_store("processes_all");
        let job_id = "worker_job_a";

        // Build 5 input lines.
        let mut input = String::new();
        for i in 0..5_u32 {
            input.push_str(&format!(r#"{{"prompt":"hello {i}","max_tokens":5}}"#));
            input.push('\n');
        }
        store
            .create_job(job_id, &input, "/v1/completions", 5)
            .expect("create_job");

        // Spin up a mock inference worker.
        let (inference_tx, mut inference_rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);

        tokio::spawn(async move {
            while let Some(req) = inference_rx.recv().await {
                if let BatchRequest::Generate { reply, .. } = req {
                    let _ = reply.send(Ok((
                        "mock output".to_string(),
                        UsageStats {
                            prompt_tokens: 2,
                            completion_tokens: 3,
                            total_tokens: 5,
                        },
                    )));
                }
            }
        });

        // Create queue and batch worker.
        let (batch_tx, batch_rx) = crate::batch_spool::queue::new_batch_queue(8);
        spawn_batch_worker(batch_rx, inference_tx, Arc::clone(&store));

        // Submit the job.
        batch_tx
            .send(crate::batch_spool::queue::BatchWorkItem {
                job_id: job_id.to_string(),
            })
            .await
            .expect("send job");

        // Wait for completion (poll with timeout).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let meta = store.read_status(job_id).expect("read status");
            if matches!(
                meta.status,
                BatchJobStatus::Completed | BatchJobStatus::Failed
            ) {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("batch job did not complete within deadline");
            }
        }

        let output_lines = store.read_output_lines(job_id).expect("read output");
        assert_eq!(
            output_lines.len(),
            5,
            "output.jsonl should have exactly 5 lines, got: {output_lines:?}"
        );
    }

    /// (b) worker_cancels_remaining — cancel after line 2; lines 3-5 are cancelled.
    #[tokio::test]
    async fn worker_cancels_remaining() {
        let store = temp_store("cancels");
        let job_id = "worker_job_b";

        // Build 5 slow input lines.
        let mut input = String::new();
        for i in 0..5_u32 {
            input.push_str(&format!(r#"{{"prompt":"item {i}","max_tokens":5}}"#));
            input.push('\n');
        }
        store
            .create_job(job_id, &input, "/v1/completions", 5)
            .expect("create_job");

        // Slow mock inference worker — allows time to set cancel flag.
        let (inference_tx, mut inference_rx) = tokio::sync::mpsc::channel::<BatchRequest>(16);

        let store_for_cancel = Arc::clone(&store);
        let job_id_str = job_id.to_string();

        tokio::spawn(async move {
            let mut count = 0_u32;
            while let Some(req) = inference_rx.recv().await {
                count += 1;
                if let BatchRequest::Generate { reply, .. } = req {
                    // After the 2nd line, set cancel flag.
                    if count == 2 {
                        if let Ok(mut meta) = store_for_cancel.read_status(&job_id_str) {
                            meta.cancel_requested = true;
                            let _ = store_for_cancel.update_status(&job_id_str, &meta);
                        }
                    }
                    // Small delay to give worker a chance to check the flag.
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    let _ = reply.send(Ok((
                        "output".to_string(),
                        UsageStats {
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            total_tokens: 2,
                        },
                    )));
                }
            }
        });

        let (batch_tx, batch_rx) = crate::batch_spool::queue::new_batch_queue(4);
        spawn_batch_worker(batch_rx, inference_tx, Arc::clone(&store));

        batch_tx
            .send(crate::batch_spool::queue::BatchWorkItem {
                job_id: job_id.to_string(),
            })
            .await
            .expect("send");

        // Wait for cancellation.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let meta = store.read_status(job_id).expect("read status");
            if matches!(
                meta.status,
                BatchJobStatus::Cancelled | BatchJobStatus::Completed | BatchJobStatus::Failed
            ) {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("batch job did not reach terminal status within deadline");
            }
        }

        let final_meta = store.read_status(job_id).expect("read final status");
        assert_eq!(
            final_meta.status,
            BatchJobStatus::Cancelled,
            "job should be cancelled: {final_meta:?}"
        );
    }
}
