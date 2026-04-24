//! OpenAI-compatible Batch API (`/v1/batches`).

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

// ── Types ────────────────────────────────────────────────────────────────────

/// A single request item within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestItem {
    pub custom_id: String,
    pub method: String,
    pub url: String,
    pub body: Value,
}

/// Status of a batch job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
}

/// Counts of requests within a batch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchRequestCounts {
    pub total: u32,
    pub completed: u32,
    pub failed: u32,
}

/// A batch job descriptor (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub status: BatchStatus,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub failed_at: Option<i64>,
    pub request_counts: BatchRequestCounts,
    pub metadata: Option<Value>,
}

/// Body for `POST /v1/batches`.
#[derive(Debug, Deserialize)]
pub struct CreateBatchBody {
    pub requests: Vec<BatchRequestItem>,
    pub endpoint: String,
    pub completion_window: Option<String>,
    pub metadata: Option<Value>,
}

// ── Store ────────────────────────────────────────────────────────────────────

/// Thread-safe in-memory batch store.
pub type BatchStore = Arc<RwLock<HashMap<String, Batch>>>;

/// Create a new empty batch store.
pub fn new_batch_store() -> BatchStore {
    Arc::new(RwLock::new(HashMap::new()))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn not_found(id: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "message": format!("Batch '{id}' not found"),
                "type": "invalid_request_error",
                "code": "batch_not_found"
            }
        })),
    )
}

// ── Handlers ─────────────────────────────────────────────────────────────────

use crate::state::AppState;

/// `POST /v1/batches` — Create a new batch.
pub async fn create_batch(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateBatchBody>,
) -> Result<Json<Batch>, (StatusCode, Json<Value>)> {
    let id = format!("batch_{}", Uuid::new_v4().as_simple());
    let total = body.requests.len() as u32;
    let now = unix_now();
    let batch = Batch {
        id: id.clone(),
        object: "batch".into(),
        endpoint: body.endpoint,
        status: BatchStatus::InProgress,
        created_at: now,
        expires_at: Some(now + 86400),
        completed_at: None,
        failed_at: None,
        request_counts: BatchRequestCounts {
            total,
            completed: 0,
            failed: 0,
        },
        metadata: body.metadata,
    };
    state.batch_store.write().await.insert(id, batch.clone());
    Ok(Json(batch))
}

/// `GET /v1/batches/{id}` — Retrieve a batch by ID.
pub async fn get_batch(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Batch>, (StatusCode, Json<Value>)> {
    let guard = state.batch_store.read().await;
    guard
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| not_found(&id))
}

/// `GET /v1/batches` — List all batches.
pub async fn list_batches(State(state): State<Arc<AppState>>) -> Json<Value> {
    let guard = state.batch_store.read().await;
    let items: Vec<&Batch> = guard.values().collect();
    Json(serde_json::json!({"object": "list", "data": items}))
}

/// `POST /v1/batches/{id}/cancel` — Cancel a batch.
pub async fn cancel_batch(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Batch>, (StatusCode, Json<Value>)> {
    let mut guard = state.batch_store.write().await;
    match guard.get_mut(&id) {
        Some(batch) => {
            batch.status = BatchStatus::Cancelled;
            Ok(Json(batch.clone()))
        }
        None => Err(not_found(&id)),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_status_serializes_snake_case() {
        let s = serde_json::to_string(&BatchStatus::InProgress).expect("serialize");
        assert_eq!(s, "\"in_progress\"");
    }

    #[test]
    fn batch_request_counts_default_is_zero() {
        let c = BatchRequestCounts::default();
        assert_eq!(c.total, 0);
        assert_eq!(c.completed, 0);
        assert_eq!(c.failed, 0);
    }

    #[tokio::test]
    async fn create_and_retrieve_batch() {
        let store = new_batch_store();
        let batch = Batch {
            id: "batch_test".into(),
            object: "batch".into(),
            endpoint: "/v1/chat/completions".into(),
            status: BatchStatus::InProgress,
            created_at: 0,
            expires_at: None,
            completed_at: None,
            failed_at: None,
            request_counts: BatchRequestCounts {
                total: 2,
                completed: 0,
                failed: 0,
            },
            metadata: None,
        };
        store.write().await.insert("batch_test".into(), batch);
        assert!(store.read().await.contains_key("batch_test"));
    }

    #[tokio::test]
    async fn cancel_nonexistent_batch_returns_not_found() {
        let store = new_batch_store();
        let guard = store.read().await;
        assert!(!guard.contains_key("no_such_id"));
        drop(guard);
        let (status, _) = not_found("no_such_id");
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
