use anyhow::{anyhow, Result};
use std::fs;
use std::path::PathBuf;
use tracing::info;

use crate::model::types::ModelInfo;

/// Manages persistent model metadata on disk, mirroring ~/.ollama/models/
#[derive(Clone)]
pub struct ModelStore {
    base: PathBuf,
}

impl ModelStore {
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Cannot find home dir"))?;
        let base = home.join(".ollama-rs").join("models");
        fs::create_dir_all(&base)?;
        fs::create_dir_all(base.join("manifests"))?;
        fs::create_dir_all(base.join("blobs"))?;
        info!("ModelStore at: {:?}", base);
        Ok(Self { base })
    }

    /// Create a ModelStore rooted at an arbitrary path.
    /// Useful for tests or when a non-default root is needed.
    #[allow(dead_code)]
    pub fn with_base(base: PathBuf) -> Result<Self> {
        fs::create_dir_all(&base)?;
        fs::create_dir_all(base.join("manifests"))?;
        fs::create_dir_all(base.join("blobs"))?;
        Ok(Self { base })
    }

    pub fn models_dir(&self) -> PathBuf {
        self.base.clone()
    }

    pub fn manifests_dir(&self) -> PathBuf {
        self.base.join("manifests")
    }

    #[allow(dead_code)]
    pub fn blobs_dir(&self) -> PathBuf {
        self.base.join("blobs")
    }

    pub fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let manifests_dir = self.manifests_dir();
        let mut models = Vec::new();

        for entry in fs::read_dir(&manifests_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                match fs::read_to_string(&path) {
                    Ok(content) => {
                        if let Ok(info) = serde_json::from_str::<ModelInfo>(&content) {
                            models.push(info);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read manifest {:?}: {}", path, e);
                    }
                }
            }
        }
        Ok(models)
    }

    pub fn save_model_info(&self, info: &ModelInfo) -> Result<()> {
        let filename = format!("{}_{}.json", info.name.replace('/', "_"), info.tag);
        let path = self.manifests_dir().join(filename);
        let content = serde_json::to_string_pretty(info)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn delete_model(&self, full_name: &str) -> Result<()> {
        let sanitized = full_name.replace([':', '/'], "_");
        let filename = format!("{}.json", sanitized);
        let path = self.manifests_dir().join(&filename);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn blob_path(&self, digest: &str) -> PathBuf {
        let filename = digest.replace("sha256:", "");
        self.blobs_dir().join(filename)
    }

    /// Save the full OCI manifest JSON for a model (from registry.ollama.ai).
    /// Used by the inference backend to locate GGUF model blobs.
    pub fn save_manifest(&self, name: &str, tag: &str, manifest: &str) -> Result<()> {
        let sanitized = format!("{}_{}_manifest.json", name.replace('/', "_"), tag);
        let path = self.manifests_dir().join(sanitized);
        fs::write(path, manifest)?;
        Ok(())
    }

    /// Load the saved OCI manifest for a model.
    pub fn load_manifest(&self, name: &str, tag: &str) -> Result<Option<String>> {
        let sanitized = format!("{}_{}_manifest.json", name.replace('/', "_"), tag);
        let path = self.manifests_dir().join(sanitized);
        if path.exists() {
            Ok(Some(fs::read_to_string(&path)?))
        } else {
            Ok(None)
        }
    }

    /// Find the path to the GGUF model blob for a given model.
    /// Reads the saved OCI manifest and locates the layer with the model media type.
    pub fn find_gguf_blob(&self, name: &str, tag: &str) -> Result<Option<PathBuf>> {
        let manifest_json = match self.load_manifest(name, tag)? {
            Some(m) => m,
            None => return Ok(None),
        };

        let manifest: serde_json::Value =
            serde_json::from_str(&manifest_json).map_err(|e| anyhow!("Invalid manifest: {}", e))?;

        let layers = match manifest["layers"].as_array() {
            Some(l) => l,
            None => return Ok(None),
        };

        // Find the model layer (GGUF format)
        for layer in layers {
            let media_type = layer["mediaType"].as_str().unwrap_or("");
            // Ollama model weights use media types like:
            // application/vnd.ollama.image.model
            // Also accept application/octet-stream as fallback for GGUF blobs
            if media_type.contains("image.model") || media_type.contains("gguf") {
                if let Some(digest) = layer["digest"].as_str() {
                    return Ok(Some(self.blob_path(digest)));
                }
            }
        }
        Ok(None)
    }
}
