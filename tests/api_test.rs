//! Integration tests for ollama-rs API endpoints.
//!
//! These tests build the full Axum router with a temporary store
//! and exercise endpoints via tower::ServiceExt::oneshot.

use axum::{
    body::{Body, HttpBody},
    http::{Request, StatusCode},
};
use serde_json::Value;
use tower::ServiceExt;

/// Build a test app with an isolated temp directory for storage.
fn test_app() -> axum::Router {
    let dir = std::env::temp_dir().join(format!("ollama_rs_itest_{}", uuid::Uuid::new_v4()));
    let store = ollama_rs::ModelStore::with_base(dir).unwrap();
    let state = ollama_rs::AppState::new(store);
    ollama_rs::router().with_state(state)
}

/// Read the full response body into bytes.
async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
    let body = resp.into_body();
    let collected = body
        .collect()
        .await
        .expect("failed to collect body");
    collected.to_bytes().to_vec()
}

async fn get(app: &mut axum::Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = body_bytes(resp).await;
    let json: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, json)
}

async fn post(app: &mut axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .uri(uri)
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = body_bytes(resp).await;
    let json: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, json)
}

// ── Health & Info ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_root() {
    let req = Request::builder().uri("/").body(Body::empty()).unwrap();
    let resp = test_app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_bytes(resp).await;
    let text = String::from_utf8(body).unwrap();
    assert_eq!(text, "Ollama is running");
}

#[tokio::test]
async fn test_health() {
    let (status, json) = get(&mut test_app(), "/api/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn test_version() {
    let (status, json) = get(&mut test_app(), "/api/version").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["version"], "0.1.0");
}

// ── Model Management ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_models_empty() {
    let (status, json) = get(&mut test_app(), "/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["models"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_create_and_list_model() {
    let mut app = test_app();

    // Create a model
    let (status, _) = post(&mut app, "/api/create", serde_json::json!({
        "name": "test-model:v1",
        "modelfile": "FROM llama3:8b\nSYSTEM You are helpful."
    })).await;
    assert_eq!(status, StatusCode::OK);

    // List should now include it
    let (_, json) = get(&mut app, "/api/tags").await;
    let models = json["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "test-model:v1");
}

#[tokio::test]
async fn test_show_existing_model() {
    let mut app = test_app();

    // Create first
    post(&mut app, "/api/create", serde_json::json!({
        "name": "show-test:latest",
        "modelfile": "FROM llama3:8b"
    })).await;

    // Show it
    let (status, json) = post(&mut app, "/api/show", serde_json::json!({
        "name": "show-test:latest"
    })).await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["modelfile"].as_str().unwrap().contains("show-test"));
}

#[tokio::test]
async fn test_show_nonexistent_returns_404() {
    let (status, _) = post(&mut test_app(), "/api/show", serde_json::json!({
        "name": "ghost:1b"
    })).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_copy_model() {
    let mut app = test_app();

    // Create source
    post(&mut app, "/api/create", serde_json::json!({
        "name": "source:latest",
        "modelfile": "FROM llama3:8b"
    })).await;

    // Copy
    let (status, _) = post(&mut app, "/api/copy", serde_json::json!({
        "source": "source:latest",
        "destination": "dest:latest"
    })).await;
    assert_eq!(status, StatusCode::OK);

    // Verify both exist
    let (_, json) = get(&mut app, "/api/tags").await;
    let names: Vec<&str> = json["models"].as_array().unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"source:latest"));
    assert!(names.contains(&"dest:latest"));
}

#[tokio::test]
async fn test_delete_model() {
    let mut app = test_app();

    // Create
    post(&mut app, "/api/create", serde_json::json!({
        "name": "delete-me:latest",
        "modelfile": "FROM llama3:8b"
    })).await;

    // Delete
    let req = Request::builder()
        .uri("/api/delete")
        .method("DELETE")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"name":"delete-me:latest"}"#))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify gone
    let (_, json) = get(&mut app, "/api/tags").await;
    assert_eq!(json["models"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_delete_nonexistent_returns_404() {
    let req = Request::builder()
        .uri("/api/delete")
        .method("DELETE")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"name":"nothing:1b"}"#))
        .unwrap();
    let resp = test_app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── Inference ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_generate_missing_model_returns_404() {
    let (status, _) = post(&mut test_app(), "/api/generate", serde_json::json!({
        "model": "ghost:1b",
        "prompt": "Hello"
    })).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_chat_missing_model_returns_404() {
    let (status, _) = post(&mut test_app(), "/api/chat", serde_json::json!({
        "model": "ghost:1b",
        "messages": [{"role": "user", "content": "Hi"}]
    })).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_embeddings_missing_model_returns_404() {
    let (status, _) = post(&mut test_app(), "/api/embeddings", serde_json::json!({
        "model": "ghost:1b",
        "prompt": "Hello"
    })).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ── OpenAI Compatible ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_openai_list_models() {
    let mut app = test_app();

    // Create a model
    post(&mut app, "/api/create", serde_json::json!({
        "name": "openai-test:v1",
        "modelfile": "FROM llama3:8b"
    })).await;

    let (status, json) = get(&mut app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    let data = json["data"].as_array().unwrap();
    assert!(data.iter().any(|m| m["id"] == "openai-test:v1"));
}

#[tokio::test]
async fn test_openai_chat_missing_model_field() {
    let (status, _) = post(&mut test_app(), "/v1/chat/completions", serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}]
    })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// ── Pull / Push / PS ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ps_empty() {
    let (status, json) = get(&mut test_app(), "/api/ps").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["models"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_push() {
    let (status, _) = post(&mut test_app(), "/api/push", serde_json::json!({
        "name": "test:latest"
    })).await;
    assert_eq!(status, StatusCode::OK);
}
