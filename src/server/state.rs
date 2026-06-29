use std::sync::Arc;
use crate::model::ModelRegistry;
use crate::model::loader::ModelLoader;
use crate::storage::ModelStore;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub loader: Arc<ModelLoader>,
    #[allow(dead_code)]
    pub store: ModelStore,
}

impl AppState {
    pub fn new(store: ModelStore) -> Self {
        let loader = ModelLoader::new(store.models_dir());
        Self {
            registry: Arc::new(ModelRegistry::new(store.clone())),
            loader: Arc::new(loader),
            store,
        }
    }
}
