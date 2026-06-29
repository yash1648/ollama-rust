//! Stub inference backend — simulated tokens.
//!
//! Used as a fallback when no real model is loaded or when
//! operating in demo mode. Returns templated responses based
//! on the prompt text.

use super::InferenceBackend;
use crate::model::types::*;
use anyhow::Result;

/// Stub inference engine that returns simulated responses.
pub struct StubBackend;

#[async_trait::async_trait]
impl InferenceBackend for StubBackend {
    async fn generate(&self, req: &GenerateRequest) -> Result<Vec<String>> {
        let response = format!(
            "I am a simulated response from model '{}'. \
             You asked: '{}'. \
             To enable real inference, download a GGUF model and use --backend candle.",
            req.model, req.prompt
        );
        Ok(tokenize_response(&response))
    }

    async fn chat(&self, req: &ChatRequest) -> Result<Vec<String>> {
        let last_user = req
            .messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("(empty)");

        let response = format!(
            "I am '{}' running on ollama-rs. You said: '{}'. \
             Download a GGUF model and use --backend candle for real inference.",
            req.model, last_user
        );
        Ok(tokenize_response(&response))
    }

    fn embed(&self, prompt: &str, dim: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vec: Vec<f32> = (0..dim)
            .map(|i| {
                let mut h = DefaultHasher::new();
                format!("{}{}", prompt, i).hash(&mut h);
                (h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }
        vec
    }
}

fn tokenize_response(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| format!("{} ", w)).collect()
}
