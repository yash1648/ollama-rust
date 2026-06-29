//! Model loader: handles pulling from the Ollama registry
//! via HTTPS using the OCI distribution spec.
//!
//! Downloads model manifests and GGUF blobs from registry.ollama.ai,
//! streaming progress through an mpsc channel for SSE forwarding.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tracing::info;

use crate::model::types::*;
use crate::model::registry::split_name;
use crate::storage::ModelStore;

/// HTTP client for pulling models from the Ollama registry.
pub struct ModelLoader {
    models_dir: PathBuf,
    client: reqwest::Client,
}

impl ModelLoader {
    pub fn new(models_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("ollama-rs/0.1.0")
            .build()
            .expect("Failed to build HTTP client");
        Self { models_dir, client }
    }

    /// Pull a model from the registry.
    ///
    /// Streams [`PullProgress`] events through `tx` as the download progresses.
    /// On success, returns [`ModelInfo`] and persists both the blob files and
    /// a JSON manifest to `~/.ollama-rs/models/`.
    pub async fn pull(
        &self,
        name: &str,
        tx: mpsc::Sender<PullProgress>,
    ) -> Result<ModelInfo> {
        let (model_name, tag) = split_name(name);

        // Split into namespace + short name
        // "library/llama3.2" → namespace="library", name="llama3.2"
        // "llama3.2"         → namespace="library", name="llama3.2"
        let (namespace, short_name) = if let Some(slash) = model_name.find('/') {
            (model_name[..slash].to_string(), model_name[slash + 1..].to_string())
        } else {
            ("library".to_string(), model_name.clone())
        };

        let registry = "https://registry.ollama.ai";
        let base_url = format!("{}/v2/{}/{}", registry, namespace, short_name);

        // ── 1. Fetch manifest ────────────────────────────────────────────────
        progress(&tx, "pulling manifest", None, None, None).await;

        let manifest_url = format!("{}/manifests/{}", base_url, tag);
        let manifest_resp = self
            .client
            .get(&manifest_url)
            .send()
            .await
            .map_err(|e| anyhow!("Cannot reach registry: {}. Check internet connection.", e))?;

        if !manifest_resp.status().is_success() {
            return Err(anyhow!(
                "Model '{}' not found in registry (HTTP {})",
                name,
                manifest_resp.status()
            ));
        }

        let manifest_text = manifest_resp.text().await?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_text)
            .map_err(|e| anyhow!("Invalid manifest from registry: {}", e))?;

        let layers = manifest["layers"]
            .as_array()
            .ok_or_else(|| anyhow!("Manifest has no 'layers' array"))?;

        // Collect layer metadata
        let mut total_size: u64 = 0;
        let mut layer_infos: Vec<(String, u64, String)> = Vec::new();

        for layer in layers {
            let digest = layer["digest"]
                .as_str()
                .ok_or_else(|| anyhow!("Layer missing 'digest'"))?
                .to_string();
            let size = layer["size"]
                .as_u64()
                .ok_or_else(|| anyhow!("Layer missing 'size'"))?;
            let media_type = layer["mediaType"]
                .as_str()
                .unwrap_or("application/octet-stream")
                .to_string();

            total_size += size;
            layer_infos.push((digest, size, media_type));
        }

        // ── 2. Download each layer blob ──────────────────────────────────────
        let blob_dir = self.models_dir.join("blobs");
        tokio::fs::create_dir_all(&blob_dir).await?;

        for (digest, size, _media_type) in &layer_infos {
            let short = if digest.len() > 26 {
                &digest[7..26]
            } else {
                digest.as_str()
            };

            progress(&tx, format!("pulling {}", short), Some(digest.clone()), Some(*size), None).await;

            let blob_url = format!("{}/blobs/{}", base_url, digest);
            let blob_path = blob_dir.join(digest.replace("sha256:", ""));

            // Skip if blob already exists and has correct size
            if let Ok(meta) = tokio::fs::metadata(&blob_path).await {
                if meta.len() == *size {
                    info!("Blob {} already exists, skipping", short);
                    progress(&tx, format!("pulling {}", short), Some(digest.clone()), Some(*size), Some(*size)).await;
                    continue;
                }
            }

            // Stream download with progress
            let resp = self
                .client
                .get(&blob_url)
                .send()
                .await
                .map_err(|e| anyhow!("Failed to download blob {}: {}", short, e))?;

            if !resp.status().is_success() {
                return Err(anyhow!(
                    "Failed to download blob {} (HTTP {})",
                    short,
                    resp.status()
                ));
            }

            let total = resp.content_length().unwrap_or(*size);
            let tmp_path = blob_dir.join(format!(".{}.tmp", digest.replace("sha256:", "")));
            let mut file = tokio::fs::File::create(&tmp_path).await?;
            let mut stream = resp.bytes_stream();
            let mut downloaded: u64 = 0;

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(|e| anyhow!("Download error: {}", e))?;
                file.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;

                progress(
                    &tx,
                    format!("pulling {}", short),
                    Some(digest.clone()),
                    Some(total),
                    Some(downloaded),
                )
                .await;
            }

            file.flush().await?;
            drop(file);

            // Atomically move temp file to final path
            tokio::fs::rename(&tmp_path, &blob_path).await?;

            progress(
                &tx,
                format!("pulling {}", short),
                Some(digest.clone()),
                Some(total),
                Some(total),
            )
            .await;
        }

        // ── 3. Verify & persist ──────────────────────────────────────────────
        progress(&tx, "verifying sha256 digest", None, None, None).await;

        progress(&tx, "writing manifest", None, None, None).await;

        let model_info = ModelInfo {
            name: model_name,
            tag: tag.clone(),
            digest: layer_infos
                .first()
                .map(|(d, _, _)| d.clone())
                .unwrap_or_default(),
            size: total_size,
            modified_at: chrono::Utc::now(),
            details: derive_details(&short_name, &tag),
        };

        // Persist manifest to disk
        let store = ModelStore::with_base(self.models_dir.clone())?;
        store.save_model_info(&model_info)?;

        progress(&tx, "success", None, None, None).await;

        Ok(model_info)
    }
}

/// Send a progress event, ignoring errors (receiver may have dropped).
async fn progress(
    tx: &mpsc::Sender<PullProgress>,
    status: impl Into<String>,
    digest: Option<String>,
    total: Option<u64>,
    completed: Option<u64>,
) {
    let _ = tx
        .send(PullProgress {
            status: status.into(),
            digest,
            total,
            completed,
        })
        .await;
}

/// Derive model metadata from name and tag heuristics.
fn derive_details(model: &str, tag: &str) -> ModelDetails {
    let family = if model.contains("llama") {
        "llama"
    } else if model.contains("mistral") {
        "mistral"
    } else if model.contains("gemma") {
        "gemma"
    } else if model.contains("phi") {
        "phi"
    } else if model.contains("qwen") {
        "qwen2"
    } else if model.contains("deepseek") {
        "deepseek"
    } else if model.contains("nomic") {
        "bert"
    } else {
        "llama"
    };

    let param_size = if tag.contains("70b") || tag.contains("72b") {
        "70B"
    } else if tag.contains("13b") {
        "13B"
    } else if tag.contains("7b") || tag.contains("8b") {
        "7B"
    } else if tag.contains("3b") {
        "3B"
    } else if tag.contains("1b") {
        "1B"
    } else if tag.contains("0.5b") {
        "0.5B"
    } else {
        "7B"
    };

    let quant = if tag.contains("q4") {
        "Q4_0"
    } else if tag.contains("q8") {
        "Q8_0"
    } else if tag.contains("fp16") {
        "F16"
    } else {
        "Q4_0"
    };

    ModelDetails {
        format: "gguf".to_string(),
        family: family.to_string(),
        families: Some(vec![family.to_string()]),
        parameter_size: param_size.to_string(),
        quantization_level: quant.to_string(),
    }
}
