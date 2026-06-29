pub mod routes;
pub mod state;
pub mod error;
pub mod inference;

pub use state::AppState;

use anyhow::Result;
use axum::Router;
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tracing::info;
use std::net::SocketAddr;

pub async fn run(state: AppState) -> Result<()> {
    state.registry.load_from_disk().await?;

    let app = Router::new()
        .merge(routes::router())
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:11434".parse()?;
    info!("Ollama-RS listening on http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
