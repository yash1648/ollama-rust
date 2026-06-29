//! Simulated inference engine.
//! Real impl would call llama.cpp via FFI or subprocess.
//! This provides a working stub that returns structured responses
//! compatible with the Ollama API so clients work immediately.

use std::time::Instant;
use crate::model::types::*;

/// Simulate token generation. In production, replace with llama.cpp FFI call.
pub async fn generate_tokens(req: &GenerateRequest) -> Vec<String> {
    let response = format!(
        "I am a simulated response from model '{}'. \
        You asked: '{}'. \
        To enable real inference, link this server to a llama.cpp backend via FFI or subprocess.",
        req.model, req.prompt
    );
    tokenize_response(&response)
}

/// Simulate chat response tokens.
pub async fn chat_tokens(req: &ChatRequest) -> Vec<String> {
    let last_user = req.messages.iter()
        .filter(|m| m.role == "user")
        .last()
        .map(|m| m.content.as_str())
        .unwrap_or("(empty)");

    let response = format!(
        "I am '{}' running on ollama-rs. You said: '{}'. \
        Wire up llama.cpp or candle for real inference.",
        req.model, last_user
    );
    tokenize_response(&response)
}

/// Simulate embeddings (random unit vector for stub).
pub fn compute_embedding(prompt: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut vec: Vec<f32> = (0..dim).map(|i| {
        let mut h = DefaultHasher::new();
        format!("{}{}", prompt, i).hash(&mut h);
        let v = (h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        v
    }).collect();

    // Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }
    vec
}

fn tokenize_response(text: &str) -> Vec<String> {
    // Word-level tokenization for demo
    text.split_whitespace()
        .map(|w| format!("{} ", w))
        .collect()
}

pub fn timing_stats(start: Instant, _token_count: u32) -> (u64, u64, u64) {
    let total_ns = start.elapsed().as_nanos() as u64;
    let eval_duration = total_ns * 8 / 10;
    let load_duration = total_ns / 10;
    (total_ns, load_duration, eval_duration)
}
