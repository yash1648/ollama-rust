use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    routing::{delete, get, post},
    Json, Router,
};
use chrono::Utc;
use futures::stream::{self, Stream, StreamExt};
use serde_json::{json, Value};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::model::{types::*, registry::normalize_name};
use crate::server::{error::ApiError, inference, state::AppState};

pub fn router() -> Router<AppState> {
    Router::new()
        // Core API
        .route("/", get(root))
        .route("/api/version", get(version))
        .route("/api/tags", get(list_models))
        .route("/api/show", post(show_model))
        .route("/api/pull", post(pull_model))
        .route("/api/push", post(push_model))
        .route("/api/create", post(create_model))
        .route("/api/copy", post(copy_model))
        .route("/api/delete", delete(delete_model))
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/embeddings", post(embeddings))
        .route("/api/ps", get(ps))
        // OpenAI-compatible
        .route("/v1/models", get(openai_list_models))
        .route("/v1/chat/completions", post(openai_chat))
}

async fn root() -> impl IntoResponse {
    "Ollama is running"
}

async fn version() -> Json<Value> {
    Json(json!({ "version": "0.1.0" }))
}

async fn list_models(State(state): State<AppState>) -> Json<ListResponse> {
    let models = state.registry.list().await;
    let entries: Vec<ModelListEntry> = models.into_iter().map(|m| ModelListEntry {
        name: m.full_name(),
        modified_at: m.modified_at,
        size: m.size,
        digest: m.digest,
        details: m.details,
    }).collect();
    Json(ListResponse { models: entries })
}

async fn show_model(
    State(state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> Result<Json<ShowResponse>, ApiError> {
    let info = state.registry.get(&req.name).await
        .ok_or_else(|| ApiError::not_found(format!("model '{}' not found", req.name)))?;

    Ok(Json(ShowResponse {
        modelfile: format!("FROM {}", info.full_name()),
        parameters: format!(
            "temperature 0.8\ntop_p 0.9\nnum_ctx 2048"
        ),
        template: "{{ .System }}\n\n{{ .Prompt }}".to_string(),
        details: info.details,
    }))
}

async fn pull_model(
    State(state): State<AppState>,
    Json(req): Json<PullRequest>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let stream_flag = req.stream.unwrap_or(true);
    let name = req.name.clone();
    let loader = state.loader.clone();
    let registry = state.registry.clone();

    info!("Pulling model: {}", name);

    let (tx, rx) = mpsc::channel::<PullProgress>(64);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        match loader.pull(&name, tx.clone()).await {
            Ok(info) => {
                let _ = registry.register(info).await;
            }
            Err(e) => {
                let _ = tx.send(PullProgress {
                    status: format!("error: {}", e),
                    digest: None,
                    total: None,
                    completed: None,
                }).await;
            }
        }
    });

    let event_stream = rx_stream.map(|progress| {
        let data = serde_json::to_string(&progress).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}

async fn push_model(
    State(_state): State<AppState>,
    Json(req): Json<PushRequest>,
) -> Json<Value> {
    Json(json!({ "status": format!("pushing {}", req.name) }))
}

async fn create_model(
    State(state): State<AppState>,
    Json(req): Json<CreateRequest>,
) -> Result<Json<Value>, ApiError> {
    info!("Creating model: {}", req.name);

    // Parse FROM line from modelfile
    let from_line = req.modelfile.lines()
        .find(|l| l.to_uppercase().starts_with("FROM "))
        .ok_or_else(|| ApiError::bad_request("Modelfile must contain FROM directive"))?;

    let base = from_line[5..].trim().to_string();
    let (name, tag) = crate::model::registry::split_name(&req.name);

    let info = ModelInfo {
        name,
        tag,
        digest: format!("sha256:{}", hex::encode(&[0u8; 32])),
        size: 0,
        modified_at: Utc::now(),
        details: ModelDetails {
            format: "gguf".to_string(),
            family: "custom".to_string(),
            families: None,
            parameter_size: "unknown".to_string(),
            quantization_level: "unknown".to_string(),
        },
    };

    state.registry.register(info).await.map_err(ApiError::from)?;
    Ok(Json(json!({ "status": "success" })))
}

async fn copy_model(
    State(state): State<AppState>,
    Json(req): Json<CopyRequest>,
) -> Result<StatusCode, ApiError> {
    state.registry.copy(&req.source, &req.destination).await
        .map_err(ApiError::from)?;
    Ok(StatusCode::OK)
}

async fn delete_model(
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> Result<StatusCode, ApiError> {
    state.registry.remove(&req.name).await.map_err(|e| {
        if e.to_string().contains("not found") {
            ApiError::not_found(e.to_string())
        } else {
            ApiError::from(e)
        }
    })?;
    Ok(StatusCode::OK)
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    if !state.registry.exists(&req.model).await {
        return ApiError::not_found(format!("model '{}' not found", req.model))
            .into_response();
    }

    let stream_flag = req.stream.unwrap_or(true);
    let model_name = req.model.clone();
    let start = Instant::now();

    if stream_flag {
        let tokens = inference::generate_tokens(&req).await;
        let token_count = tokens.len() as u32;

        let event_stream = stream::iter(tokens.into_iter().enumerate()).then(
            move |(_i, token)| {
                let model = model_name.clone();
                async move {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    let resp = GenerateResponse {
                        model: model,
                        created_at: Utc::now(),
                        response: token,
                        done: false,
                        context: None,
                        total_duration: None,
                        load_duration: None,
                        prompt_eval_count: None,
                        prompt_eval_duration: None,
                        eval_count: None,
                        eval_duration: None,
                    };
                    let data = serde_json::to_string(&resp).unwrap_or_default();
                    Ok::<Event, axum::Error>(Event::default().data(data))
                }
            }
        ).chain(stream::once(async move {
            let (total, load, eval) = inference::timing_stats(start, token_count);
            let done_resp = GenerateResponse {
                model: req.model.clone(),
                created_at: Utc::now(),
                response: "".to_string(),
                done: true,
                context: Some(vec![1, 2, 3]),
                total_duration: Some(total),
                load_duration: Some(load),
                prompt_eval_count: Some(req.prompt.split_whitespace().count() as u32),
                prompt_eval_duration: Some(load),
                eval_count: Some(token_count),
                eval_duration: Some(eval),
            };
            let data = serde_json::to_string(&done_resp).unwrap_or_default();
            Ok::<Event, axum::Error>(Event::default().data(data))
        }));

        Sse::new(event_stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let tokens = inference::generate_tokens(&req).await;
        let full_response = tokens.join("");
        let token_count = tokens.len() as u32;
        let (total, load, eval) = inference::timing_stats(start, token_count);

        let resp = GenerateResponse {
            model: req.model.clone(),
            created_at: Utc::now(),
            response: full_response,
            done: true,
            context: Some(vec![1, 2, 3]),
            total_duration: Some(total),
            load_duration: Some(load),
            prompt_eval_count: Some(req.prompt.split_whitespace().count() as u32),
            prompt_eval_duration: Some(load),
            eval_count: Some(token_count),
            eval_duration: Some(eval),
        };
        Json(resp).into_response()
    }
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !state.registry.exists(&req.model).await {
        return ApiError::not_found(format!("model '{}' not found", req.model))
            .into_response();
    }

    let stream_flag = req.stream.unwrap_or(true);
    let model_name = req.model.clone();
    let start = Instant::now();

    if stream_flag {
        let tokens = inference::chat_tokens(&req).await;
        let token_count = tokens.len() as u32;

        let event_stream = stream::iter(tokens.into_iter()).then(move |token| {
            let model = model_name.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                let resp = ChatResponse {
                    model,
                    created_at: Utc::now(),
                    message: Message {
                        role: "assistant".to_string(),
                        content: token,
                        images: None,
                    },
                    done: false,
                    total_duration: None,
                    load_duration: None,
                    prompt_eval_count: None,
                    eval_count: None,
                    eval_duration: None,
                };
                let data = serde_json::to_string(&resp).unwrap_or_default();
                Ok::<Event, axum::Error>(Event::default().data(data))
            }
        }).chain(stream::once(async move {
            let (total, load, eval) = inference::timing_stats(start, token_count);
            let done_resp = ChatResponse {
                model: req.model.clone(),
                created_at: Utc::now(),
                message: Message {
                    role: "assistant".to_string(),
                    content: "".to_string(),
                    images: None,
                },
                done: true,
                total_duration: Some(total),
                load_duration: Some(load),
                prompt_eval_count: Some(10),
                eval_count: Some(token_count),
                eval_duration: Some(eval),
            };
            let data = serde_json::to_string(&done_resp).unwrap_or_default();
            Ok::<Event, axum::Error>(Event::default().data(data))
        }));

        Sse::new(event_stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let tokens = inference::chat_tokens(&req).await;
        let full = tokens.join("");
        let token_count = tokens.len() as u32;
        let (total, load, eval) = inference::timing_stats(start, token_count);

        let resp = ChatResponse {
            model: req.model.clone(),
            created_at: Utc::now(),
            message: Message {
                role: "assistant".to_string(),
                content: full,
                images: None,
            },
            done: true,
            total_duration: Some(total),
            load_duration: Some(load),
            prompt_eval_count: Some(10),
            eval_count: Some(token_count),
            eval_duration: Some(eval),
        };
        Json(resp).into_response()
    }
}

async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    if !state.registry.exists(&req.model).await {
        return Err(ApiError::not_found(format!("model '{}' not found", req.model)));
    }

    let embedding = inference::compute_embedding(&req.prompt, 4096);
    Ok(Json(EmbeddingResponse { embedding }))
}

async fn ps(State(state): State<AppState>) -> Json<PsResponse> {
    // Return empty — no models actively loaded in RAM for stub
    Json(PsResponse { models: vec![] })
}

// ── OpenAI-compatible endpoints ──────────────────────────────────────────────

async fn openai_list_models(State(state): State<AppState>) -> Json<Value> {
    let models = state.registry.list().await;
    let data: Vec<Value> = models.into_iter().map(|m| json!({
        "id": m.full_name(),
        "object": "model",
        "created": m.modified_at.timestamp(),
        "owned_by": "ollama-rs",
    })).collect();

    Json(json!({ "object": "list", "data": data }))
}

async fn openai_chat(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let model = body["model"].as_str()
        .ok_or_else(|| ApiError::bad_request("missing 'model' field"))?
        .to_string();

    let messages = body["messages"].as_array()
        .cloned()
        .unwrap_or_default();

    let last_content = messages.iter().rev()
        .find(|m| m["role"] == "user")
        .and_then(|m| m["content"].as_str())
        .unwrap_or("")
        .to_string();

    let req = ChatRequest {
        model: model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: last_content,
            images: None,
        }],
        stream: Some(false),
        format: None,
        options: None,
        keep_alive: None,
    };

    let tokens = inference::chat_tokens(&req).await;
    let content = tokens.join("");

    Ok(Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": tokens.len(),
            "total_tokens": tokens.len() + 10
        }
    })))
}
