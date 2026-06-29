mod api;
mod model;
mod server;
mod storage;

use anyhow::Result;
use server::backend::BackendKind;
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Default bind address for the server.
const DEFAULT_HOST: &str = "0.0.0.0:11434";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ollama_rs=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let addr: SocketAddr = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| DEFAULT_HOST.to_string())
        .parse()
        .expect("Invalid OLLAMA_HOST: expected IP:port (e.g. 0.0.0.0:11434)");

    // Select inference backend via OLLAMA_BACKEND env var
    // "candle" → real inference via HuggingFace Candle
    // (anything else) → stub/simulated inference
    let backend_kind = match std::env::var("OLLAMA_BACKEND")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "candle" => {
            tracing::info!("Using Candle inference backend");
            BackendKind::Candle
        }
        _ => {
            tracing::info!("Using stub inference backend (set OLLAMA_BACKEND=candle for real inference)");
            BackendKind::Stub
        }
    };

    let store = storage::ModelStore::new()?;
    let state = server::AppState::with_backend(store, backend_kind)?;

    server::run(state, addr).await
}
