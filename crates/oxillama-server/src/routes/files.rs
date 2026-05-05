//! HTTP route handlers for the OpenAI Files API (`/v1/files`).
//!
//! | Method | Path | Handler |
//! |--------|------|---------|
//! | POST   | `/v1/files`               | [`create_file_handler`]  |
//! | GET    | `/v1/files`               | [`list_files_handler`]   |
//! | GET    | `/v1/files/:file_id`      | [`get_file_handler`]     |
//! | GET    | `/v1/files/:file_id/content` | [`get_file_content_handler`] |
//! | DELETE | `/v1/files/:file_id`      | [`delete_file_handler`]  |

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::multipart::Field;
use axum::extract::{Multipart, Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::error::ServerError;
use crate::files_store::{FilePurpose, FilesStore, MAX_FILE_BYTES};
use crate::state::AppState;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Extract `Arc<FilesStore>` from state, returning 503 if not configured.
fn require_files_store(state: &AppState) -> Option<Arc<FilesStore>> {
    state.files_store.clone()
}

/// Build a 503 "Files API not enabled" response.
fn files_unavailable_response() -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "Files API is not enabled on this server",
            "type": "service_unavailable",
        }
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

fn not_found_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "not_found_error",
        }
    });
    (StatusCode::NOT_FOUND, Json(body)).into_response()
}

fn server_error_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
        }
    });
    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

fn payload_too_large_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "payload_too_large",
        }
    });
    (StatusCode::PAYLOAD_TOO_LARGE, Json(body)).into_response()
}

// ── POST /v1/files ────────────────────────────────────────────────────────────

/// Upload a new file.
///
/// Accepts `multipart/form-data` with fields:
/// - `file`    — the file bytes (required)
/// - `purpose` — one of `"assistants"`, `"batch"`, `"fine-tune"` (required)
///
/// Maximum upload size: 512 MiB.
pub async fn create_file_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let store = match require_files_store(&state) {
        Some(s) => s,
        None => return files_unavailable_response(),
    };

    let mut file_bytes: Option<(String, Bytes)> = None;
    let mut purpose_str: Option<String> = None;

    // Drain all multipart fields; collect `file` and `purpose`.
    loop {
        let field_result = multipart.next_field().await;
        let field: Field = match field_result {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return server_error_response(&format!("multipart error: {e}"));
            }
        };

        let field_name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().unwrap_or("upload").to_string();

        match field_name.as_str() {
            "file" => {
                let bytes = match field.bytes().await {
                    Ok(b) => b,
                    Err(e) => {
                        return server_error_response(&format!("read file field: {e}"));
                    }
                };
                if bytes.len() > MAX_FILE_BYTES {
                    return payload_too_large_response(&format!(
                        "file exceeds maximum upload size of {} bytes",
                        MAX_FILE_BYTES
                    ));
                }
                file_bytes = Some((file_name, bytes));
            }
            "purpose" => {
                let text = match field.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        return server_error_response(&format!("read purpose field: {e}"));
                    }
                };
                purpose_str = Some(text);
            }
            _ => {
                // Consume and ignore unknown fields.
                let _ = field.bytes().await;
            }
        }
    }

    let (filename, raw_bytes) = match file_bytes {
        Some(f) => f,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": "missing required 'file' field",
                        "type": "invalid_request_error",
                    }
                })),
            )
                .into_response();
        }
    };

    let purpose = match purpose_str
        .as_deref()
        .and_then(FilePurpose::from_purpose_str)
    {
        Some(p) => p,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "missing or invalid 'purpose' field; must be one of 'assistants', 'batch', 'fine-tune' — got {:?}",
                            purpose_str
                        ),
                        "type": "invalid_request_error",
                    }
                })),
            )
                .into_response();
        }
    };

    let data = raw_bytes.to_vec();
    let result = tokio::task::spawn_blocking(move || store.create(&filename, purpose, &data)).await;

    match result {
        Ok(Ok(meta)) => (StatusCode::OK, Json(meta)).into_response(),
        Ok(Err(ServerError::FileTooLarge(msg))) => payload_too_large_response(&msg),
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── GET /v1/files ─────────────────────────────────────────────────────────────

/// List all uploaded files.
pub async fn list_files_handler(State(state): State<Arc<AppState>>) -> Response {
    let store = match require_files_store(&state) {
        Some(s) => s,
        None => return files_unavailable_response(),
    };

    match tokio::task::spawn_blocking(move || store.list()).await {
        Ok(Ok(files)) => {
            let body = serde_json::json!({
                "object": "list",
                "data": files,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── GET /v1/files/:file_id ────────────────────────────────────────────────────

/// Retrieve metadata for a single file.
pub async fn get_file_handler(
    State(state): State<Arc<AppState>>,
    Path(file_id): Path<String>,
) -> Response {
    let store = match require_files_store(&state) {
        Some(s) => s,
        None => return files_unavailable_response(),
    };

    let fid = file_id.clone();
    match tokio::task::spawn_blocking(move || store.get(&fid)).await {
        Ok(Ok(meta)) => (StatusCode::OK, Json(meta)).into_response(),
        Ok(Err(ServerError::FileNotFound(_))) => {
            not_found_response(&format!("File '{}' not found", file_id))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── GET /v1/files/:file_id/content ────────────────────────────────────────────

/// Download the raw bytes of a file.
pub async fn get_file_content_handler(
    State(state): State<Arc<AppState>>,
    Path(file_id): Path<String>,
) -> Response {
    let store = match require_files_store(&state) {
        Some(s) => s,
        None => return files_unavailable_response(),
    };

    let fid = file_id.clone();
    match tokio::task::spawn_blocking(move || store.get_content(&fid)).await {
        Ok(Ok(data)) => (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/octet-stream"),
            )],
            data,
        )
            .into_response(),
        Ok(Err(ServerError::FileNotFound(_))) => {
            not_found_response(&format!("File '{}' not found", file_id))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}

// ── DELETE /v1/files/:file_id ─────────────────────────────────────────────────

/// Delete a file.
///
/// Returns `{"id": "<file_id>", "object": "file", "deleted": true}` on success.
pub async fn delete_file_handler(
    State(state): State<Arc<AppState>>,
    Path(file_id): Path<String>,
) -> Response {
    let store = match require_files_store(&state) {
        Some(s) => s,
        None => return files_unavailable_response(),
    };

    let fid = file_id.clone();
    match tokio::task::spawn_blocking(move || store.delete(&fid)).await {
        Ok(Ok(())) => {
            let body = serde_json::json!({
                "id": file_id,
                "object": "file",
                "deleted": true,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(Err(ServerError::FileNotFound(_))) => {
            not_found_response(&format!("File '{}' not found", file_id))
        }
        Ok(Err(e)) => server_error_response(&e.to_string()),
        Err(e) => server_error_response(&format!("task join: {e}")),
    }
}
