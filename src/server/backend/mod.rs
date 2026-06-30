//! Inference backend abstraction.
//!
//! Defines the [`InferenceBackend`] trait that all inference
//! engines must implement. Currently two backends exist:
//!
//! - **Stub** — simulated tokens, used as fallback when no model is loaded
//! - **Candle** — real inference via [candle](https://github.com/huggingface/candle)

pub mod candle;
pub mod chat_template;
pub mod stub;

use crate::model::types::*;
use anyhow::Result;

/// Common interface for generating tokens from a model.
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Generate response tokens for a text generation request.
    async fn generate(&self, req: &GenerateRequest) -> Result<Vec<String>>;

    /// Generate response tokens for a chat request.
    async fn chat(&self, req: &ChatRequest) -> Result<Vec<String>>;

    /// Compute an embedding vector for the given prompt text.
    fn embed(&self, prompt: &str, dim: usize) -> Vec<f32>;
}

/// Which inference backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Simulated/stub inference (no model needed).
    Stub,
    /// Real inference via Candle (HuggingFace Rust ML framework).
    Candle,
}
