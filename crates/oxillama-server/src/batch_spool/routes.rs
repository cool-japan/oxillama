//! HTTP route handlers for the OpenAI Batch API (`/v1/batches`).
//!
//! This module replaces the in-memory `batch.rs` with a disk-spooled
//! implementation.  All jobs persist across server restarts.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs::File as TokioFile;
use uuid::Uuid;

use crate::batch_spool::queue::{BatchQueueSender, BatchWorkItem};
use crate::batch_spool::store::{BatchJobMeta, BatchJobStatus, BatchStore};
use crate::state::AppState;

/// Maximum number of request lines per batch.
pub const MAX_BATCH_LINES: usize = 50_000;
/// Maximum total JSONL payload size in bytes (10 MiB).
pub const MAX_BATCH_BYTES: usize = 10 * 1024 * 1024;

// ── Request / Response types ─────────────────────────────────────────────────

/// Body for `POST /v1/batches`.
#[derive(Debug, Deserialize)]
pub struct CreateBatchBody {
    /// Inline JSONL content (one request object per line).
    /// Used when `/v1/files` is not available.
    #[serde(default)]
    pub input_jsonl: String,
    /// OpenAI-compatible file ID (ignored for now; use `input_jsonl` instead).
    #[serde(default)]
    pub input_file_id: String,
    /// Target endpoint (e.g. `/v1/chat/completions`).
    pub endpoint: String,
    /// Completion window hint (e.g. `"24h"`).
    #[serde(default)]
    pub completion_window: Option<String>,
    /// Optional caller-supplied metadata.
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Pagination parameters for `GET /v1/batches`.
#[derive(Debug, Deserialize, Default)]
pub struct ListBatchesQuery {
    /// Maximum number of results to return (default 20).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Cursor: return jobs created after this ID.
    pub after: Option<String>,
}

fn default_limit() -> usize {
    20
}

/// OpenAI-compatible batch object returned by the API.
#[derive(Debug, Serialize)]
pub struct BatchObject {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub status: String,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub failed_at: Option<i64>,
    pub request_counts: BatchRequestCounts,
    pub metadata: Option<Value>,
}

/// Counts of requests within a batch.
#[derive(Debug, Serialize, Default)]
pub struct BatchRequestCounts {
    pub total: u32,
    pub completed: u32,
    pub failed: u32,
}

/// Shared router state passed to handlers.
pub struct BatchRouterState {
    pub store: Arc<BatchStore>,
    pub queue_tx: BatchQueueSender,
}

// ── Conversion helpers ───────────────────────────────────────────────────────

fn meta_to_object(meta: &BatchJobMeta) -> BatchObject {
    let status_str = match meta.status {
        BatchJobStatus::Pending => "validating",
        BatchJobStatus::InProgress => "in_progress",
        BatchJobStatus::Completed => "completed",
        BatchJobStatus::Failed => "failed",
        BatchJobStatus::Cancelled => "cancelled",
    };

    let completed_at = matches!(
        meta.status,
        BatchJobStatus::Completed | BatchJobStatus::Failed | BatchJobStatus::Cancelled
    )
    .then_some(meta.updated_at);

    let failed_at = matches!(meta.status, BatchJobStatus::Failed).then_some(meta.updated_at);

    BatchObject {
        id: meta.id.clone(),
        object: "batch".to_string(),
        endpoint: meta.endpoint.clone(),
        status: status_str.to_string(),
        created_at: meta.created_at,
        expires_at: Some(meta.created_at + 86400),
        completed_at,
        failed_at,
        request_counts: BatchRequestCounts {
            total: meta.total_lines,
            completed: meta.completed_lines,
            failed: meta.failed_lines,
        },
        metadata: None,
    }
}

fn not_found(id: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!("Batch '{id}' not found"),
            "type": "invalid_request_error",
            "code": "batch_not_found"
        }
    });
    (StatusCode::NOT_FOUND, Json(body)).into_response()
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `POST /v1/batches` — create a new batch job.
pub async fn create_batch(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateBatchBody>,
) -> Response {
    let input_jsonl = if !body.input_jsonl.is_empty() {
        body.input_jsonl.clone()
    } else {
        String::new()
    };

    if input_jsonl.len() > MAX_BATCH_BYTES {
        let err = serde_json::json!({
            "error": {
                "message": format!("input exceeds max size of {} bytes", MAX_BATCH_BYTES),
                "type": "invalid_request_error",
            }
        });
        return (StatusCode::PAYLOAD_TOO_LARGE, Json(err)).into_response();
    }

    let line_count = input_jsonl.lines().filter(|l| !l.trim().is_empty()).count();

    if line_count > MAX_BATCH_LINES {
        let err = serde_json::json!({
            "error": {
                "message": format!("input exceeds max {} lines", MAX_BATCH_LINES),
                "type": "invalid_request_error",
            }
        });
        return (StatusCode::PAYLOAD_TOO_LARGE, Json(err)).into_response();
    }

    let job_id = format!("batch_{}", Uuid::new_v4().as_simple());

    // Write to disk.
    let store = Arc::clone(&state.batch_disk_store);
    let job_id_c = job_id.clone();
    let endpoint = body.endpoint.clone();
    let jsonl = input_jsonl.clone();
    let total = line_count as u32;

    let meta_result =
        tokio::task::spawn_blocking(move || store.create_job(&job_id_c, &jsonl, &endpoint, total))
            .await;

    match meta_result {
        Ok(Ok(meta)) => {
            // Enqueue for background processing.
            if let Err(e) = state
                .batch_queue_tx
                .send(BatchWorkItem {
                    job_id: job_id.clone(),
                })
                .await
            {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("failed to enqueue batch job: {e}"),
                )
            } else {
                (StatusCode::ACCEPTED, Json(meta_to_object(&meta))).into_response()
            }
        }
        Ok(Err(e)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to create batch job on disk: {e}"),
        ),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("task join error: {e}"),
        ),
    }
}

/// `GET /v1/batches/:id` — retrieve a batch by ID.
pub async fn get_batch(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = Arc::clone(&state.batch_disk_store);
    let id_c = id.clone();

    match tokio::task::spawn_blocking(move || store.read_status(&id_c)).await {
        Ok(Ok(meta)) => (StatusCode::OK, Json(meta_to_object(&meta))).into_response(),
        Ok(Err(_)) => not_found(&id),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("task join error: {e}"),
        ),
    }
}

/// `GET /v1/batches` — list all batches with optional pagination.
pub async fn list_batches(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListBatchesQuery>,
) -> Response {
    let store = Arc::clone(&state.batch_disk_store);

    let ids_result = tokio::task::spawn_blocking(move || store.list_jobs()).await;

    let ids = match ids_result {
        Ok(Ok(ids)) => ids,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("list_jobs error: {e}"),
            );
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("task join: {e}"),
            );
        }
    };

    // Apply pagination: skip until after `query.after`.
    let mut skip = query.after.is_some();
    let mut objects = Vec::new();

    for id in &ids {
        if skip {
            if query.after.as_deref() == Some(id.as_str()) {
                skip = false;
            }
            continue;
        }
        if objects.len() >= query.limit {
            break;
        }

        let store2 = Arc::clone(&state.batch_disk_store);
        let id2 = id.clone();
        if let Ok(Ok(meta)) = tokio::task::spawn_blocking(move || store2.read_status(&id2)).await {
            objects.push(meta_to_object(&meta));
        }
    }

    let body = serde_json::json!({
        "object": "list",
        "data": objects,
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// `GET /v1/batches/:id/output` — stream output JSONL as a file response.
pub async fn get_batch_output(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let output_path = state.batch_disk_store.job_dir(&id).join("output.jsonl");

    match TokioFile::open(&output_path).await {
        Ok(file) => {
            use axum::http::header;
            use tokio_util::io::ReaderStream;
            let stream = ReaderStream::new(file);
            let body = Body::from_stream(stream);
            (
                StatusCode::OK,
                [(
                    header::CONTENT_TYPE,
                    "application/jsonl"
                        .parse::<axum::http::HeaderValue>()
                        .unwrap_or_else(|_| "text/plain".parse().expect("header")),
                )],
                body,
            )
                .into_response()
        }
        Err(_) => not_found(&id),
    }
}

/// `POST /v1/batches/:id/cancel` — request cancellation.
pub async fn cancel_batch(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let store = Arc::clone(&state.batch_disk_store);
    let id_c = id.clone();

    let result = tokio::task::spawn_blocking(move || -> std::io::Result<BatchJobMeta> {
        let mut meta = store.read_status(&id_c)?;
        meta.cancel_requested = true;
        meta.updated_at = unix_now();
        store.update_status(&id_c, &meta)?;
        Ok(meta)
    })
    .await;

    match result {
        Ok(Ok(meta)) => (StatusCode::OK, Json(meta_to_object(&meta))).into_response(),
        Ok(Err(_)) => not_found(&id),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("task join: {e}"),
        ),
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
        }
    });
    (status, Json(body)).into_response()
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
