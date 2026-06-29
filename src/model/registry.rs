use crate::model::types::*;
use crate::storage::ModelStore;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ModelStore;

    fn temp_store() -> ModelStore {
        let dir = std::env::temp_dir().join(format!("ollama_rs_test_{}", uuid::Uuid::new_v4()));
        ModelStore::with_base(dir).unwrap()
    }

    fn sample_model(name: &str, tag: &str) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            tag: tag.to_string(),
            digest: format!("sha256:{}", hex::encode([0u8; 32])),
            size: 1024,
            modified_at: chrono::Utc::now(),
            details: ModelDetails {
                format: "gguf".to_string(),
                family: "llama".to_string(),
                families: Some(vec!["llama".to_string()]),
                parameter_size: "7B".to_string(),
                quantization_level: "Q4_0".to_string(),
            },
        }
    }

    // ── normalize_name ──────────────────────────────────────────────────────

    #[test]
    fn normalize_name_with_tag() {
        assert_eq!(normalize_name("llama3:8b"), "llama3:8b");
    }

    #[test]
    fn normalize_name_without_tag_appends_latest() {
        assert_eq!(normalize_name("llama3"), "llama3:latest");
    }

    #[test]
    fn normalize_name_with_colon_only_returns_as_is() {
        // A bare ":" passes the contains(':') check and is returned verbatim.
        assert_eq!(normalize_name(":"), ":");
    }

    // ── split_name ──────────────────────────────────────────────────────────

    #[test]
    fn split_name_with_tag() {
        assert_eq!(split_name("llama3:8b"), ("llama3".into(), "8b".into()));
    }

    #[test]
    fn split_name_without_tag_uses_latest() {
        assert_eq!(split_name("llama3"), ("llama3".into(), "latest".into()));
    }

    #[test]
    fn split_name_multiple_colons() {
        assert_eq!(split_name("a:b:c"), ("a".into(), "b:c".into()));
    }

    // ── ModelRegistry ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn register_and_list() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        let model = sample_model("llama3", "8b");

        registry.register(model.clone()).await.unwrap();
        let models = registry.list().await;

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].full_name(), "llama3:8b");
    }

    #[tokio::test]
    async fn register_and_get() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        let model = sample_model("mistral", "7b");

        registry.register(model.clone()).await.unwrap();
        let found = registry.get("mistral:7b").await;

        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "mistral");
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);

        let found = registry.get("nonexistent:latest").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn exists_returns_true_for_registered() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        registry
            .register(sample_model("gemma", "2b"))
            .await
            .unwrap();

        assert!(registry.exists("gemma:2b").await);
    }

    #[tokio::test]
    async fn exists_returns_false_for_missing() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);

        assert!(!registry.exists("missing:latest").await);
    }

    #[tokio::test]
    async fn remove_existing_model() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        registry.register(sample_model("phi3", "3b")).await.unwrap();

        registry.remove("phi3:3b").await.unwrap();
        assert!(!registry.exists("phi3:3b").await);
    }

    #[tokio::test]
    async fn remove_nonexistent_returns_error() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);

        let result = registry.remove("nope:1b").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn copy_model() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        registry
            .register(sample_model("llama3", "8b"))
            .await
            .unwrap();

        registry.copy("llama3:8b", "my-llama:latest").await.unwrap();
        assert!(registry.exists("my-llama:latest").await);

        let original = registry.get("llama3:8b").await.unwrap();
        let copied = registry.get("my-llama:latest").await.unwrap();
        assert_eq!(original.digest, copied.digest);
    }

    #[tokio::test]
    async fn copy_nonexistent_returns_error() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);

        let result = registry.copy("ghost:1b", "copy:1b").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn list_empty_registry() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);

        let models = registry.list().await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn register_multiple_models() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);

        registry
            .register(sample_model("llama3", "8b"))
            .await
            .unwrap();
        registry
            .register(sample_model("mistral", "7b"))
            .await
            .unwrap();
        registry
            .register(sample_model("gemma", "2b"))
            .await
            .unwrap();

        let models = registry.list().await;
        assert_eq!(models.len(), 3);
    }

    #[tokio::test]
    async fn exists_by_normalized_name() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        registry.register(sample_model("test", "v1")).await.unwrap();

        // Should find by full name
        assert!(registry.exists("test:v1").await);
        // "test" normalizes to "test:latest" which doesn't match "test:v1"
        assert!(!registry.exists("test").await);
    }

    // Additional test for exists with implicit normalization
    #[tokio::test]
    async fn exists_normalizes_name() {
        let store = temp_store();
        let registry = ModelRegistry::new(store);
        registry
            .register(sample_model("demo", "latest"))
            .await
            .unwrap();

        // "demo" should normalize to "demo:latest"
        assert!(registry.exists("demo").await);
    }
}
