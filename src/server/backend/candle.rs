//! Real inference via [Candle](https://github.com/huggingface/candle).
//!
//! Loads GGUF model files from the Ollama blob store and runs
//! inference using HuggingFace's pure-Rust ML framework.
//!
//! ## Supported architectures
//! - LLaMA (llama, llama2, llama3, codellama, deepseek-llm, yi)
//!
//! More architectures (Mistral, Phi, Qwen, Gemma) will be added
//! in follow-up commits.

use super::InferenceBackend;
use crate::model::types::*;
use crate::storage::ModelStore;
use anyhow::{anyhow, Context, Result};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::path::PathBuf;

/// Candle-based inference backend.
///
/// Loads GGUF models on demand and runs generation using
/// quantized transformer implementations from `candle-transformers`.
pub struct CandleBackend {
    store: ModelStore,
    device: Device,
}

/// A loaded model, keeping the GGUF content, reader, and
/// architecture-specific model weights alive together.
struct LoadedModel {
    /// The quantized model weights (architecture-specific).
    model: Box<dyn ModelForward>,
    /// Tokenizer for encoding/decoding text.
    tokenizer: Option<tokenizers::Tokenizer>,
    /// Maximum sequence length.
    max_seq_len: usize,
}

/// Trait abstracting the forward pass across model architectures.
trait ModelForward: Send {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor>;
}

// ── LLaMA family ───────────────────────────────────────────────────────────

struct LlamaModel {
    inner: candle_transformers::models::quantized_llama::ModelWeights,
}

impl ModelForward for LlamaModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.inner.forward(input, index_pos)?)
    }
}

impl CandleBackend {
    /// Create a new Candle backend that reads models from the given store.
    pub fn new(store: ModelStore) -> Result<Self> {
        let device = Device::Cpu;
        Ok(Self { store, device })
    }

    /// Find the GGUF blob file for a given model name+tag.
    fn find_model_blob(&self, name: &str, tag: &str) -> Result<Option<PathBuf>> {
        // First, try the saved OCI manifest
        if let Some(path) = self
            .store
            .find_gguf_blob(name, tag)
            .context("looking up GGUF blob")?
        {
            return Ok(Some(path));
        }

        // Fallback: scan blobs directory for any file that looks like a GGUF
        let blobs_dir = self.store.blobs_dir();
        if !blobs_dir.exists() {
            return Ok(None);
        }

        for entry in std::fs::read_dir(&blobs_dir)
            .map_err(|e| anyhow!("reading blobs dir: {e}"))?
        {
            let entry = entry.map_err(|e| anyhow!("reading blob entry: {e}"))?;
            let path = entry.path();
            if path.is_file() {
                // Check for GGUF magic bytes
                if let Ok(mut f) = std::fs::File::open(&path) {
                    use std::io::Read;
                    let mut magic = [0u8; 4];
                    if f.read_exact(&mut magic).is_ok() && &magic == b"GGUF" {
                        return Ok(Some(path));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Try to find or download a tokenizer for the given model.
    fn resolve_tokenizer(
        &self,
        model_path: &std::path::Path,
        ct: &gguf_file::Content,
        model_name: &str,
    ) -> Option<tokenizers::Tokenizer> {
        // Strategy 1: tokenizer.json next to the model file
        if let Some(dir) = model_path.parent() {
            let tok_path = dir.join("tokenizer.json");
            if tok_path.exists() {
                if let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path) {
                    tracing::info!("Loaded tokenizer from {:?}", tok_path);
                    return Some(tok);
                }
            }
        }

        // Strategy 2: try to build from GGUF tokenizer metadata
        // (BPE-based models like LLaMA)
        if let Some(tok) = build_tokenizer_from_gguf(ct) {
            tracing::info!("Built tokenizer from GGUF metadata");
            return Some(tok);
        }

        // Strategy 3: try downloading from HuggingFace Hub
        if let Some(tok) = download_tokenizer_hf(model_name) {
            tracing::info!("Downloaded tokenizer from HuggingFace for {model_name}");
            return Some(tok);
        }

        tracing::warn!("No tokenizer found for {model_name}, using word-level fallback");
        None
    }

    /// Load a GGUF model from disk, returning the loaded model.
    fn load_gguf(&self, model_path: &std::path::Path, model_name: &str) -> Result<LoadedModel> {
        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("opening GGUF file {:?}", model_path))?;

        let ct = gguf_file::Content::read(&mut file)
            .with_context(|| format!("reading GGUF content from {:?}", model_path))?;

        // Determine architecture from metadata
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();

        tracing::info!("Loading model, architecture: {arch}");

        let max_seq_len = ct
            .metadata
            .get("llama.context_length")
            .or_else(|| ct.metadata.get("llama.max_position_embeddings"))
            .or_else(|| ct.metadata.get("mistral.context_length"))
            .or_else(|| ct.metadata.get("phi3.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;

        let tokenizer = self.resolve_tokenizer(model_path, &ct, model_name);

        // Load the appropriate architecture
        let model: Box<dyn ModelForward> = match arch.as_str() {
            "llama" | "llama2" | "llama3" | "yi" | "deepseek2" | "codellama" => {
                let m = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    ct, &mut file, &self.device,
                )
                .context("loading quantized LLaMA model")?;
                Box::new(LlamaModel { inner: m })
            }
            // Future architectures will be added here
            other => {
                anyhow::bail!(
                    "Unsupported model architecture '{other}'. \
                     Currently supported: llama, llama2, llama3, yi, deepseek2, codellama"
                );
            }
        };

        Ok(LoadedModel {
            model,
            tokenizer,
            max_seq_len,
        })
    }

    /// Generate tokens from a prompt using the loaded model.
    fn generate_tokens_impl(
        &self,
        model: &mut LoadedModel,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f64>,
    ) -> Result<Vec<String>> {
        let token_ids = match &model.tokenizer {
            Some(tokenizer) => {
                let encoding = tokenizer
                    .encode(prompt, true)
                    .map_err(|e| anyhow!("tokenization error: {e}"))?;
                encoding.get_ids().to_vec()
            }
            None => {
                // Fallback: character-level tokenization using byte values
                // This is very basic and won't produce good results, but
                // lets the system run without a proper tokenizer
                prompt.bytes().map(|b| b as u32).collect()
            }
        };

        if token_ids.is_empty() {
            return Err(anyhow!("empty tokenization for prompt"));
        }

        let max_tokens = max_tokens.min(model.max_seq_len.saturating_sub(token_ids.len()));
        let eos_token = match &model.tokenizer {
            Some(tok) => tok
                .token_to_id("<|end_of_text|>")
                .or_else(|| tok.token_to_id("</s>"))
                .or_else(|| tok.token_to_id("<|eot_id|>"))
                .unwrap_or(2),
            None => 2,
        };

        let mut logits_processor = LogitsProcessor::new(42, temperature, None);
        let mut all_tokens = token_ids.clone();
        let mut output_tokens: Vec<String> = Vec::new();
        let mut prev_index = 0usize;

        for _ in 0..max_tokens {
            let context_size = all_tokens.len();
            let pos = context_size - 1;

            let input = Tensor::new(&[all_tokens[pos]], &self.device)?.unsqueeze(0)?;
            let logits = model.model.forward(&input, pos)?;

            // Get next token
            let next_token = logits_processor.sample(&logits.squeeze(0)?)?;

            if next_token == eos_token {
                break;
            }

            all_tokens.push(next_token);

            // Decode incrementally for streaming
            if let Some(tokenizer) = &model.tokenizer {
                let new_tokens = &all_tokens[prev_index..];
                if let Ok(text) = tokenizer.decode(new_tokens, true) {
                    if text.len() > 0 {
                        // Send token string (space-delimited for compatibility with SSE stream)
                        output_tokens.push(format!("{} ", text));
                        prev_index = all_tokens.len();
                    } else {
                        // Accumulate until we get printable text
                    }
                }
            } else {
                // Fallback: use raw token IDs as strings
                output_tokens.push(format!("[{}] ", next_token));
            }
        }

        if output_tokens.is_empty() {
            // Ensure at least one token
            output_tokens.push("".to_string());
        }

        Ok(output_tokens)
    }
}

/// Try to download a tokenizer.json from HuggingFace Hub.
fn download_tokenizer_hf(model_name: &str) -> Option<tokenizers::Tokenizer> {
    // Map common Ollama model names to HuggingFace repo IDs
    let hf_repos: Vec<(&str, &str)> = vec![
        ("llama3.2", "meta-llama/Llama-3.2-1B"),
        ("llama3.1", "meta-llama/Llama-3.1-8B"),
        ("llama3", "meta-llama/Meta-Llama-3-8B"),
        ("llama2", "meta-llama/Llama-2-7b-hf"),
        ("mistral", "mistralai/Mistral-7B-v0.1"),
        ("mixtral", "mistralai/Mixtral-8x7B-v0.1"),
        ("phi3", "microsoft/Phi-3-mini-4k-instruct"),
        ("phi", "microsoft/phi-2"),
        ("qwen2", "Qwen/Qwen2-7B-Instruct"),
        ("gemma2", "google/gemma-2-2b"),
        ("deepseek", "deepseek-ai/deepseek-llm-7b-chat"),
    ];

    let model_lower = model_name.to_lowercase();
    let repo_id = hf_repos
        .iter()
        .find(|(key, _)| model_lower.contains(*key))
        .map(|(_, repo)| *repo)?;

    match (|| -> Result<tokenizers::Tokenizer> {
        let api = hf_hub::api::sync::Api::new()?;
        let path = api
            .repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get("tokenizer.json")?;
        let tok = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        Ok(tok)
    })() {
        Ok(tok) => Some(tok),
        Err(e) => {
            tracing::warn!("Failed to download tokenizer for {model_name}: {e}");
            None
        }
    }
}

/// Extract a vector of strings from a GGUF metadata value.
fn gguf_vec_string(val: &candle::quantized::gguf_file::Value) -> Option<Vec<String>> {
    let vec = val.to_vec().ok()?;
    vec.iter().map(|v| v.to_string().ok().cloned()).collect()
}

/// Attempt to build a tokenizer from GGUF metadata.
fn build_tokenizer_from_gguf(ct: &gguf_file::Content) -> Option<tokenizers::Tokenizer> {
    let tokenizer_model = ct
        .metadata
        .get("tokenizer.ggml.model")?
        .to_string()
        .ok()?;
    let tokenizer_model = tokenizer_model.as_str();

    let tokens: Vec<String> = match ct.metadata.get("tokenizer.ggml.tokens") {
        Some(v) => gguf_vec_string(v)?,
        None => return None,
    };

    let merges: Vec<String> = ct
        .metadata
        .get("tokenizer.ggml.merges")
        .and_then(|v| gguf_vec_string(v))
        .unwrap_or_default();

    // Only support BPE tokenizers (used by LLaMA/Mistral/etc.)
    match tokenizer_model {
        "gpt2" | "bpe" => {
            let vocab: ahash::AHashMap<String, u32> = tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i as u32))
                .collect();

            let merge_pairs: Vec<(String, String)> = merges
                .iter()
                .filter_map(|m| {
                    let mut parts = m.splitn(2, ' ');
                    Some((parts.next()?.to_string(), parts.next()?.to_string()))
                })
                .collect();

            let bpe = tokenizers::models::bpe::BpeBuilder::new()
                .vocab_and_merges(vocab, merge_pairs)
                .build()
                .ok()?;

            let tokenizer = tokenizers::Tokenizer::new(bpe);

            Some(tokenizer)
        }
        _ => {
            tracing::warn!("Unsupported tokenizer model: {tokenizer_model}");
            None
        }
    }
}

// ── InferenceBackend trait implementation ──────────────────────────────────

#[async_trait::async_trait]
impl InferenceBackend for CandleBackend {
    async fn generate(&self, req: &GenerateRequest) -> Result<Vec<String>> {
        // Parse model name:tag
        let (name, tag) = crate::model::registry::split_name(&req.model);
        let tag = if tag.is_empty() { "latest" } else { &tag };

        // Find the GGUF model file
        let blob_path = self
            .find_model_blob(&name, tag)?
            .ok_or_else(|| anyhow!("Model '{name}:{tag}' has no GGUF blob. Pull it first."))?;

        // Load model
        let mut loaded = self.load_gguf(&blob_path, &req.model)?;

        // Determine max tokens from request options
        let max_tokens = req
            .options
            .as_ref()
            .and_then(|o| o.num_predict)
            .filter(|n| *n > 0)
            .map(|n| n as usize)
            .unwrap_or(512);

        let temperature = req.options.as_ref().and_then(|o| o.temperature).map(|t| t as f64);

        // Generate
        let tokens = self.generate_tokens_impl(
            &mut loaded,
            &req.prompt,
            max_tokens,
            temperature,
        )?;

        Ok(tokens)
    }

    async fn chat(&self, req: &ChatRequest) -> Result<Vec<String>> {
        // Build a prompt from the chat messages
        let last_user = req
            .messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // Simple prompt format — future versions will use proper chat templates
        let prompt = format!("<|im_start|>user\n{last_user}\n<|im_end|>\n<|im_start|>assistant\n");

        // Parse model name:tag
        let (name, tag) = crate::model::registry::split_name(&req.model);
        let tag = if tag.is_empty() { "latest" } else { &tag };

        let blob_path = self
            .find_model_blob(&name, tag)?
            .ok_or_else(|| anyhow!("Model '{name}:{tag}' has no GGUF blob. Pull it first."))?;

        let mut loaded = self.load_gguf(&blob_path, &req.model)?;

        let max_tokens = req
            .options
            .as_ref()
            .and_then(|o| o.num_predict)
            .filter(|n| *n > 0)
            .map(|n| n as usize)
            .unwrap_or(512);

        let temperature = req.options.as_ref().and_then(|o| o.temperature).map(|t| t as f64);

        let tokens = self.generate_tokens_impl(
            &mut loaded,
            &prompt,
            max_tokens,
            temperature,
        )?;

        Ok(tokens)
    }

    fn embed(&self, prompt: &str, dim: usize) -> Vec<f32> {
        // For now, use the same stub embedding until we implement
        // a proper embedding model backend
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vec: Vec<f32> = (0..dim)
            .map(|i| {
                let mut h = DefaultHasher::new();
                format!("{}{}", prompt, i).hash(&mut h);
                (h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }
        vec
    }
}
