#![allow(dead_code, unused_imports, unused_variables)]

mod api;
mod model;
mod server;
mod storage;

use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ollama_rs=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let store = storage::ModelStore::new()?;
    let state = server::AppState::new(store);

    server::run(state).await
}
