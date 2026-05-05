//! In-memory store for Responses API objects.
//!
//! `ResponseStore` holds [`ResponseRecord`] entries keyed by their stable `id`.
//! All operations are O(1) for single-record access; [`ResponseStore::list`]
//! sorts by `created_at` descending so the newest records appear first.
//!
//! The store is wrapped in `Arc<RwLock<…>>` so it can be shared across the
//! axum thread-pool without additional synchronisation at the call site.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ServerError, ServerResult};

// ── Public types ──────────────────────────────────────────────────────────────

/// Lifecycle status of a response object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// The response is being processed.
    InProgress,
    /// The response completed successfully.
    Completed,
    /// The response failed.
    Failed,
}

/// A stored response object as returned by `GET /v1/responses/:id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseRecord {
    /// Stable identifier (`resp_<uuid>`).
    pub id: String,
    /// Always `"response"`.
    pub object: String,
    /// Unix timestamp (seconds) when the record was created.
    pub created_at: u64,
    /// Model identifier used for this response.
    pub model: String,
    /// Current lifecycle status.
    pub status: ResponseStatus,
    /// Input messages submitted with the request.
    pub input: Vec<serde_json::Value>,
    /// Generated output text (`None` while `InProgress`).
    pub output: Option<String>,
    /// ID of the previous response this one continues from, if any.
    pub previous_response_id: Option<String>,
    /// System-level instructions prepended to the context.
    pub instructions: Option<String>,
    /// Tool definitions supplied with the request (reserved; not yet executed).
    pub tools: Vec<serde_json::Value>,
}

impl ResponseRecord {
    /// Create a new record in the `InProgress` state.
    ///
    /// The `id` is generated as `resp_<uuid_v4>` and `created_at` is set to
    /// the current UNIX timestamp.
    pub fn new_in_progress(
        model: String,
        input: Vec<serde_json::Value>,
        previous_response_id: Option<String>,
        instructions: Option<String>,
        tools: Vec<serde_json::Value>,
    ) -> Self {
        let id = format!("resp_{}", Uuid::new_v4().simple());
        let created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            id,
            object: "response".to_string(),
            created_at,
            model,
            status: ResponseStatus::InProgress,
            input,
            output: None,
            previous_response_id,
            instructions,
            tools,
        }
    }
}

// ── ResponseStore ─────────────────────────────────────────────────────────────

/// Thread-safe in-memory store for [`ResponseRecord`] objects.
///
/// All mutations acquire the inner `RwLock` for writing; reads use a shared
/// read lock so concurrent GET requests do not block each other.
#[derive(Debug, Clone)]
pub struct ResponseStore {
    records: Arc<RwLock<HashMap<String, ResponseRecord>>>,
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert `rec` into the store and return its stable `id`.
    ///
    /// # Errors
    /// Returns [`ServerError::FileStoreError`] if the write lock is poisoned
    /// (should not occur under normal operation).
    pub fn create(&self, rec: ResponseRecord) -> ServerResult<String> {
        let id = rec.id.clone();
        let mut map = self.records.write().map_err(|e| {
            ServerError::FileStoreError(format!("response store lock poisoned: {e}"))
        })?;
        map.insert(id.clone(), rec);
        Ok(id)
    }

    /// Retrieve a single record by `id`.
    ///
    /// # Errors
    /// - [`ServerError::ResponseNotFound`] when no record with `id` exists.
    /// - [`ServerError::FileStoreError`] if the read lock is poisoned.
    pub fn get(&self, id: &str) -> ServerResult<ResponseRecord> {
        let map = self.records.read().map_err(|e| {
            ServerError::FileStoreError(format!("response store lock poisoned: {e}"))
        })?;
        map.get(id)
            .cloned()
            .ok_or_else(|| ServerError::ResponseNotFound(id.to_string()))
    }

    /// Update the `output` and `status` of an existing record.
    ///
    /// # Errors
    /// - [`ServerError::ResponseNotFound`] when no record with `id` exists.
    /// - [`ServerError::FileStoreError`] if the write lock is poisoned.
    pub fn update_output(
        &self,
        id: &str,
        output: String,
        status: ResponseStatus,
    ) -> ServerResult<()> {
        let mut map = self.records.write().map_err(|e| {
            ServerError::FileStoreError(format!("response store lock poisoned: {e}"))
        })?;
        let rec = map
            .get_mut(id)
            .ok_or_else(|| ServerError::ResponseNotFound(id.to_string()))?;
        rec.output = Some(output);
        rec.status = status;
        Ok(())
    }

    /// Return all stored records sorted by `created_at` descending (newest first).
    pub fn list(&self) -> Vec<ResponseRecord> {
        let map = self.records.read().unwrap_or_else(|e| e.into_inner());
        let mut records: Vec<ResponseRecord> = map.values().cloned().collect();
        records.sort_by_key(|b| std::cmp::Reverse(b.created_at));
        records
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(model: &str) -> ResponseRecord {
        ResponseRecord::new_in_progress(
            model.to_string(),
            vec![serde_json::json!({"role": "user", "content": "hello"})],
            None,
            None,
            vec![],
        )
    }

    #[test]
    fn responses_create_returns_id() {
        let store = ResponseStore::new();
        let rec = make_record("gpt-test");
        let id = store.create(rec).expect("create should succeed");
        assert!(
            id.starts_with("resp_"),
            "id should start with 'resp_': {id}"
        );
    }

    #[test]
    fn responses_get_retrieves_record() {
        let store = ResponseStore::new();
        let rec = make_record("test-model");
        let original_id = rec.id.clone();
        let stored_id = store.create(rec).expect("create should succeed");
        assert_eq!(stored_id, original_id);

        let retrieved = store.get(&stored_id).expect("get should succeed");
        assert_eq!(retrieved.id, stored_id);
        assert_eq!(retrieved.model, "test-model");
        assert_eq!(retrieved.status, ResponseStatus::InProgress);
        assert!(retrieved.output.is_none());
    }

    #[test]
    fn responses_unknown_id_returns_not_found() {
        let store = ResponseStore::new();
        let err = store.get("resp_does_not_exist").unwrap_err();
        assert!(
            matches!(err, ServerError::ResponseNotFound(_)),
            "expected ResponseNotFound, got: {err:?}"
        );
    }

    #[test]
    fn responses_list_returns_descending() {
        let store = ResponseStore::new();

        // Insert three records with distinct, manually-controlled created_at
        // values by constructing them directly (bypassing `new_in_progress`).
        for i in 0u64..3 {
            let mut rec = make_record("model");
            // Override created_at so ordering is deterministic.
            rec.created_at = i + 1; // 1, 2, 3
            store.create(rec).expect("create");
        }

        let list = store.list();
        assert_eq!(list.len(), 3);
        // Sorted newest → oldest.
        assert!(
            list[0].created_at >= list[1].created_at,
            "list should be sorted descending: {:?}",
            list.iter().map(|r| r.created_at).collect::<Vec<_>>()
        );
        assert!(
            list[1].created_at >= list[2].created_at,
            "list should be sorted descending"
        );
    }

    #[test]
    fn responses_update_output_changes_status() {
        let store = ResponseStore::new();
        let rec = make_record("m");
        let id = store.create(rec).expect("create");

        store
            .update_output(&id, "hello world".to_string(), ResponseStatus::Completed)
            .expect("update_output should succeed");

        let updated = store.get(&id).expect("get after update");
        assert_eq!(updated.status, ResponseStatus::Completed);
        assert_eq!(updated.output.as_deref(), Some("hello world"));
    }
}
