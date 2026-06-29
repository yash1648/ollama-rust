use crate::model::loader::ModelLoader;
use crate::model::ModelRegistry;
use crate::server::backend::{BackendKind, InferenceBackend};
use crate::server::backend::{candle::CandleBackend, stub::StubBackend};
use crate::storage::ModelStore;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub loader: Arc<ModelLoader>,
    #[allow(dead_code)]
    pub store: ModelStore,
    pub backend: Arc<dyn InferenceBackend>,
}

impl AppState {
    /// Create a new AppState with a stub (simulated) inference backend.
    pub fn new(store: ModelStore) -> Self {
        Self::with_backend(store, BackendKind::Stub).expect("StubBackend never fails")
    }

    /// Create a new AppState with the specified inference backend.
    pub fn with_backend(store: ModelStore, kind: BackendKind) -> Result<Self, anyhow::Error> {
        let loader = ModelLoader::new(store.models_dir());
        let backend: Arc<dyn InferenceBackend> = match kind {
            BackendKind::Stub => Arc::new(StubBackend),
            BackendKind::Candle => Arc::new(CandleBackend::new(store.clone())?),
        };
        Ok(Self {
            registry: Arc::new(ModelRegistry::new(store.clone())),
            loader: Arc::new(loader),
            store,
            backend,
        })
    }
}
