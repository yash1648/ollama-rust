//! Real inference via [Candle](https://github.com/huggingface/candle).
//!
//! Loads GGUF model files from the Ollama blob store and runs
//! inference using HuggingFace's pure-Rust ML framework.
//!
//! ## Supported architectures
//! - LLaMA (llama, llama2, llama3, codellama, deepseek-llm, yi) — `from_gguf`
//! - Mistral / Mixtral — `quantized_mistral::Model` via `Config` + `VarBuilder`
//! - Phi-3 — `from_gguf`
//! - Qwen2 — `from_gguf`
//! - Gemma 2/3 — `from_gguf`

use super::InferenceBackend;
use crate::model::types::*;
use crate::storage::ModelStore;
use anyhow::{anyhow, Context, Result};
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::sync::Mutex as AsyncMutex;

/// Candle-based inference backend.
///
/// Loads GGUF models on demand and runs generation using
/// quantized transformer implementations from `candle-transformers`.
///
/// Loaded models are cached in RAM so subsequent requests to the
/// same model skip the expensive GGUF file loading step.
pub struct CandleBackend {
    store: ModelStore,
    device: Device,
    /// Cache of loaded models keyed by `name:tag`.
    /// Each model is behind an `AsyncMutex` so concurrent requests to
    /// different models run in parallel, while requests to the same
    /// model are serialized (one generation at a time).
    model_cache: RwLock<HashMap<String, Arc<AsyncMutex<LoadedModel>>>>,
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

// ── Architecture wrappers ──────────────────────────────────────────────────
// Each wraps a specific candle-transformers model and delegates `forward`.

struct LlamaModel {
    inner: candle_transformers::models::quantized_llama::ModelWeights,
}

impl ModelForward for LlamaModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.inner.forward(input, index_pos)?)
    }
}

struct Phi3Model {
    inner: candle_transformers::models::quantized_phi3::ModelWeights,
}

impl ModelForward for Phi3Model {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.inner.forward(input, index_pos)?)
    }
}

struct Qwen2Model {
    inner: candle_transformers::models::quantized_qwen2::ModelWeights,
}

impl ModelForward for Qwen2Model {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.inner.forward(input, index_pos)?)
    }
}

struct Gemma3Model {
    inner: candle_transformers::models::quantized_gemma3::ModelWeights,
}

impl ModelForward for Gemma3Model {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.inner.forward(input, index_pos)?)
    }
}

/// Mistral uses `Config` + `VarBuilder` instead of `from_gguf`.
struct MistralModel {
    inner: candle_transformers::models::quantized_mistral::Model,
}

impl ModelForward for MistralModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(self.inner.forward(input, index_pos)?)
    }
}

/// Build a `quantized_mistral::Config` from GGUF metadata keys.
fn build_mistral_config(ct: &gguf_file::Content) -> candle::Result<candle_transformers::models::mistral::Config> {
    use candle_nn::Activation;

    let md_get = |s: &str| match ct.metadata.get(s) {
        None => candle::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };

    let head_count = md_get("mistral.attention.head_count")?.to_u32()? as usize;
    let head_count_kv = md_get("mistral.attention.head_count_kv")?.to_u32()? as usize;
    let block_count = md_get("mistral.block_count")?.to_u32()? as usize;
    let embedding_length = md_get("mistral.embedding_length")?.to_u32()? as usize;
    let context_length = md_get("mistral.context_length")?.to_u32()? as usize;
    let feed_forward_length = md_get("mistral.feed_forward_length")?.to_u32()? as usize;
    let rms_eps = md_get("mistral.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let rope_theta = md_get("mistral.rope.freq_base")
        .and_then(|m| m.to_f32())
        .unwrap_or(10_000f32) as f64;

    // Check for optional sliding window
    let sliding_window = ct.metadata
        .get("mistral.attention.sliding_window")
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize);

    // Activation function — default to Silu for Mistral
    let hidden_act = ct.metadata
        .get("mistral.feed_forward.activation")
        .and_then(|v| v.to_string().ok())
        .and_then(|s| {
            match s.as_str() {
                "silu" | "SiLU" | "swish" => Some(Activation::Silu),
                "gelu" => Some(Activation::Gelu),
                _ => None,
            }
        })
        .unwrap_or(Activation::Silu);

    Ok(candle_transformers::models::mistral::Config {
        vocab_size: 0,               // populated from tensor shape at load time
        hidden_size: embedding_length,
        intermediate_size: feed_forward_length,
        num_hidden_layers: block_count,
        num_attention_heads: head_count,
        head_dim: None,              // inferred from hidden_size / num_attention_heads
        num_key_value_heads: head_count_kv,
        hidden_act,
        max_position_embeddings: context_length,
        rms_norm_eps: rms_eps,
        rope_theta,
        sliding_window,
        use_flash_attn: false,
    })
}

impl CandleBackend {
    /// Create a new Candle backend that reads models from the given store.
    pub fn new(store: ModelStore) -> Result<Self> {
        let device = Device::Cpu;
        Ok(Self {
            store,
            device,
            model_cache: RwLock::new(HashMap::new()),
        })
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

        // Determine max_seq_len from the correct architecture prefix
        let ctx_keys: &[&str] = &[
            "llama.context_length",
            "llama.max_position_embeddings",
            "mistral.context_length",
            "phi3.context_length",
            "qwen2.context_length",
            "gemma3.context_length",
        ];
        let max_seq_len = ctx_keys
            .iter()
            .find_map(|k| ct.metadata.get(*k).and_then(|v| v.to_u32().ok()))
            .unwrap_or(4096) as usize;

        let tokenizer = self.resolve_tokenizer(model_path, &ct, model_name);

        // Load the appropriate architecture
        let model: Box<dyn ModelForward> = match arch.as_str() {
            // ── LLaMA family (from_gguf) ──
            "llama" | "llama2" | "llama3" | "yi" | "deepseek2" | "codellama" => {
                let m = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    ct, &mut file, &self.device,
                )
                .context("loading quantized LLaMA model")?;
                Box::new(LlamaModel { inner: m })
            }

            // ── Phi-3 (from_gguf) ──
            "phi3" | "phi-3" | "phi-3-mini" => {
                let m = candle_transformers::models::quantized_phi3::ModelWeights::from_gguf(
                    false, // use_flash_attn
                    ct, &mut file, &self.device,
                )
                .context("loading quantized Phi-3 model")?;
                Box::new(Phi3Model { inner: m })
            }

            // ── Qwen2 (from_gguf) ──
            "qwen2" => {
                let m = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                    ct, &mut file, &self.device,
                )
                .context("loading quantized Qwen2 model")?;
                Box::new(Qwen2Model { inner: m })
            }

            // ── Gemma 2/3 (from_gguf) ──
            "gemma3" | "gemma2" | "gemma" => {
                let m = candle_transformers::models::quantized_gemma3::ModelWeights::from_gguf(
                    ct, &mut file, &self.device,
                )
                .context("loading quantized Gemma model")?;
                Box::new(Gemma3Model { inner: m })
            }

            // ── Mistral / Mixtral (Config + VarBuilder) ──
            "mistral" | "mixtral" => {
                // Build config from GGUF metadata
                let config = build_mistral_config(&ct)
                    .context("building Mistral config from GGUF metadata")?;

                // Release the file handle before creating VarBuilder (it re-opens)
                drop(file);

                // VarBuilder loads quantized tensors directly from the GGUF file
                let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                    model_path, &self.device,
                )
                .context("creating quantized VarBuilder from GGUF")?;

                let m = candle_transformers::models::quantized_mistral::Model::new(
                    &config, vb,
                )
                .context("loading quantized Mistral model")?;

                Box::new(MistralModel { inner: m })
            }

            other => {
                anyhow::bail!(
                    "Unsupported model architecture '{other}'. \
                     Currently supported: llama, llama2, llama3, yi, deepseek2, codellama, \
                     mistral, mixtral, phi3, qwen2, gemma2, gemma3"
                );
            }
        };

        Ok(LoadedModel {
            model,
            tokenizer,
            max_seq_len,
        })
    }

    /// Get or load a model from the cache.
    ///
    /// On cache hit, returns the existing in-memory model (instant).
    /// On cache miss, loads the GGUF file from disk, inserts it into
    /// the cache, and returns it.
    async fn get_or_load_model(
        &self,
        name: &str,
        tag: &str,
    ) -> Result<Arc<AsyncMutex<LoadedModel>>> {
        let cache_key = format!("{name}:{tag}");

        // Fast path: check cache under read lock
        {
            let cache = self
                .model_cache
                .read()
                .map_err(|e| anyhow!("model cache read lock poisoned: {e}"))?;
            if let Some(model_arc) = cache.get(&cache_key) {
                tracing::debug!("Model cache hit: {cache_key}");
                return Ok(model_arc.clone());
            }
        }

        // Cache miss: find blob and load from disk
        tracing::info!("Model cache miss: {cache_key} — loading from disk");
        let blob_path = self
            .find_model_blob(name, tag)?
            .ok_or_else(|| anyhow!("Model '{name}:{tag}' has no GGUF blob. Pull it first."))?;

        let model_full_name = format!("{name}:{tag}");
        let loaded = self.load_gguf(&blob_path, &model_full_name)?;
        let model_arc = Arc::new(AsyncMutex::new(loaded));

        // Store in cache under write lock
        let mut cache = self
            .model_cache
            .write()
            .map_err(|e| anyhow!("model cache write lock poisoned: {e}"))?;
        cache.insert(cache_key, model_arc.clone());

        Ok(model_arc)
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
                    if !text.is_empty() {
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
        .and_then(gguf_vec_string)
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

        // Get or load model from cache
        let model_arc = self.get_or_load_model(&name, tag).await?;
        let mut model = model_arc.lock().await;

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
        self.generate_tokens_impl(&mut model, &req.prompt, max_tokens, temperature)
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

        // Get or load model from cache
        let model_arc = self.get_or_load_model(&name, tag).await?;
        let mut model = model_arc.lock().await;

        let max_tokens = req
            .options
            .as_ref()
            .and_then(|o| o.num_predict)
            .filter(|n| *n > 0)
            .map(|n| n as usize)
            .unwrap_or(512);

        let temperature = req.options.as_ref().and_then(|o| o.temperature).map(|t| t as f64);

        self.generate_tokens_impl(&mut model, &prompt, max_tokens, temperature)
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
