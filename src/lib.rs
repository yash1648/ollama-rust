//! ollama-rs — Ollama-compatible API server library.
//!
//! Provides public types and builders for embedding or testing.

pub mod api;
pub mod model;
pub mod server;
pub mod storage;

// Re-export key types for external consumers and integration tests.
pub use server::AppState;
pub use server::routes::router;
pub use storage::ModelStore;
