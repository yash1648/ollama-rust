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
}
