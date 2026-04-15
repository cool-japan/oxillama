//! GET /v1/models handler.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::error::ServerResult;
use crate::state::AppState;

/// Model list response (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Information about a loaded model.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// List available models.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> ServerResult<Json<ModelListResponse>> {
    let response = ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: state.loaded_at,
            owned_by: "oxillama".to_string(),
        }],
    };

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use crate::test_helpers::{build_test_app, get};

    /// `/v1/models` returns HTTP 200 with a `data` array.
    #[tokio::test]
    async fn test_list_models_returns_200() {
        let app = build_test_app().await;
        let (status, body) = get(app, "/v1/models").await;
        assert_eq!(status.as_u16(), 200);
        assert_eq!(body["object"], "list");
    }

    /// The `data` array must contain exactly the model that was registered at
    /// startup (`"test-model"` in tests).
    #[tokio::test]
    async fn test_list_models_data_contains_test_model() {
        let app = build_test_app().await;
        let (_status, body) = get(app, "/v1/models").await;
        let data = body["data"].as_array().expect("data must be an array");
        assert_eq!(data.len(), 1, "exactly one model should be listed");
        assert_eq!(data[0]["id"], "test-model");
        assert_eq!(data[0]["owned_by"], "oxillama");
    }

    /// Confirm the `object` type on a model entry is `"model"`.
    #[tokio::test]
    async fn test_list_models_object_type_is_model() {
        let app = build_test_app().await;
        let (_status, body) = get(app, "/v1/models").await;
        let data = body["data"].as_array().expect("data must be an array");
        assert_eq!(data[0]["object"], "model");
    }
}
