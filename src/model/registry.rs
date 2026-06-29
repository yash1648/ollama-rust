use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use tracing::info;
use crate::model::types::*;
use crate::storage::ModelStore;

pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    store: ModelStore,
}

impl ModelRegistry {
    pub fn new(store: ModelStore) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            store,
        }
    }

    pub async fn load_from_disk(&self) -> Result<()> {
        let models = self.store.list_models()?;
        let mut map = self.models.write().await;
        for model in models {
            let key = model.full_name();
            info!("Loaded model from disk: {}", key);
            map.insert(key, model);
        }
        Ok(())
    }

    pub async fn list(&self) -> Vec<ModelInfo> {
        self.models.read().await.values().cloned().collect()
    }

    pub async fn get(&self, name: &str) -> Option<ModelInfo> {
        let key = normalize_name(name);
        self.models.read().await.get(&key).cloned()
    }

    pub async fn exists(&self, name: &str) -> bool {
        let key = normalize_name(name);
        self.models.read().await.contains_key(&key)
    }

    pub async fn register(&self, info: ModelInfo) -> Result<()> {
        let key = info.full_name();
        self.store.save_model_info(&info)?;
        self.models.write().await.insert(key, info);
        Ok(())
    }

    pub async fn remove(&self, name: &str) -> Result<()> {
        let key = normalize_name(name);
        {
            let mut map = self.models.write().await;
            if map.remove(&key).is_none() {
                return Err(anyhow!("model '{}' not found", name));
            }
        }
        self.store.delete_model(&key)?;
        Ok(())
    }

    pub async fn copy(&self, source: &str, dest: &str) -> Result<()> {
        let src_key = normalize_name(source);
        let mut info = {
            let map = self.models.read().await;
            map.get(&src_key)
                .cloned()
                .ok_or_else(|| anyhow!("model '{}' not found", source))?
        };
        let (name, tag) = split_name(dest);
        info.name = name;
        info.tag = tag;
        self.register(info).await
    }
}

pub fn normalize_name(name: &str) -> String {
    if name.contains(':') {
        name.to_string()
    } else {
        format!("{}:latest", name)
    }
}

pub fn split_name(name: &str) -> (String, String) {
    if let Some(pos) = name.find(':') {
        (name[..pos].to_string(), name[pos + 1..].to_string())
    } else {
        (name.to_string(), "latest".to_string())
    }
}
