# Architecture Map: ollama-rs

## System Context (C4 — Level 1)

```
┌─────────────┐     HTTP/1.1     ┌───────────────────┐     ┌──────────────────────────┐
│  Ollama CLI  │ ──────────────> │    ollama-rs       │ ──> │  Candle Inference        │
│  curl/httpx  │ <────────────── │    (this server)   │ <── │  (pure Rust, GGUF)      │
│  OpenAI SDK  │     SSE JSON    │    :11434          │     │  5 architectures         │
└─────────────┘                  └────────┬──────────┘     └──────────────────────────┘
                                           │
                                           ▼
                                   ┌──────────────────┐
                                   │  ~/.ollama-rs/    │
                                   │  models/          │
                                   │  ├─ manifests/    │
                                   │  └─ blobs/        │
                                   └──────────────────┘
```

## Container Architecture (C4 — Level 2)

```
                          ┌──────────────────────────────────────────────┐
                          │           Axum HTTP Server                   │
                          │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
                          │  │ Routes   │  │  SSE     │  │  CORS +   │  │
                          │  │ Router   │  │  Stream  │  │  Tracing  │  │
                          │  └────┬─────┘  └──────────┘  └───────────┘  │
                          └───────┼──────────────────────────────────────┘
                                  │
          ┌───────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
┌──────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
│  Model Registry   │   │  Inference Backend       │   │  Model Loader        │
│  (in-memory       │   │  (Candle or Stub)        │   │  (pull/download)    │
│   RwLock<HashMap>)│   │                          │   │                      │
│  ┌──────────────┐ │   │  CandleBackend:          │   │  OCI distribution    │
│  │ list/get     │ │   │  ├─ LLaMA/Mistral/Qwen.. │   │  spec pull with      │
│  │ register/    │ │   │  ├─ Model cache (RAM)    │   │  SSE progress        │
│  │ remove/copy  │ │   │  └─ HF tokenizer         │   │  stream              │
│  └──────────────┘ │   │                          │   │                      │
│                   │   │  StubBackend:            │   │  pull() → stream     │
│                   │   │  └─ Simulated tokens     │   │  progress via         │
│                   │   │                          │   │  mpsc channel        │
│                   │   └──────────────────────────┘   └──────────────────────┘
└─────────┬─────────┘                                    │
          │                                              │
          ▼                                              ▼
┌──────────────────┐                          ┌──────────────────────┐
│  ModelStore       │                          │  Blob Storage        │
│  (filesystem)     │                          │  (filesystem)        │
│  ┌──────────────┐ │                          │  ~/.ollama-rs/models/│
│  │ manifests/*  │ │                          │  /blobs/{sha256}     │
│  │ .json        │ │                          └──────────────────────┘
│  └──────────────┘ │
└──────────────────┘
```

## Component Design

### 1. HTTP Server Layer (`src/server/`)
```
server/
├── mod.rs       # Axum server setup, address binding, middleware stacking
├── routes.rs    # All route handlers (20 endpoints)
├── state.rs     # AppState: shared Arc-wrapped registry + loader + store
├── error.rs     # ApiError type → HTTP JSON error responses
└── inference.rs # Simulated inference (stub, FFI hook point)
```

**Routes** (20 endpoints):
| Method | Path | Handler | Status |
|--------|------|---------|--------|
| GET | `/` | `root` | ✅ |
| GET | `/api/version` | `version` | ✅ |
| GET | `/api/tags` | `list_models` | ✅ |
| POST | `/api/show` | `show_model` | ✅ |
| POST | `/api/pull` | `pull_model` | ✅ |
| POST | `/api/push` | `push_model` | ✅ |
| POST | `/api/create` | `create_model` | ✅ |
| POST | `/api/copy` | `copy_model` | ✅ |
| DELETE | `/api/delete` | `delete_model` | ✅ |
| POST | `/api/generate` | `generate` | ✅ |
| POST | `/api/chat` | `chat` | ✅ |
| POST | `/api/embeddings` | `embeddings` | ✅ |
| GET | `/api/ps` | `ps` | ✅ |
| GET | `/v1/models` | `openai_list_models` | ✅ |
| POST | `/v1/chat/completions` | `openai_chat` | ✅ |

### 2. Model Layer (`src/model/`)
```
model/
├── mod.rs    # Re-exports
├── types.rs  # All request/response types (serde Serialize/Deserialize)
├── registry.rs # In-memory HashMap<RwLock> + disk persistence
└── loader.rs # Model pulling with simulated download progress
```

**Type definitions**: 20+ structs covering the full Ollama API surface.

### 3. Storage Layer (`src/storage/`)
```
storage/
└── mod.rs  # ModelStore: filesystem CRUD for manifests + blobs
```

**Disk layout** (`~/.ollama-rs/models/`):
```
~/.ollama-rs/
└── models/
    ├── manifests/           # JSON model metadata
    │   ├── llama3_8b.json   # {name, tag, digest, size, details}
    │   └── ...
    └── blobs/               # GGUF weight files (future)
        └── {sha256_hex}
```

### 4. Public API (`src/api/`)
Re-exports types + registry for library consumers.

## Data Flow

### Generate (streaming)
```
Client → POST /api/generate
  → routes::generate()
  → registry.exists(model) — 404 if missing
  → inference::generate_tokens() — stub response
  → SSE stream of token chunks
  → Final "done: true" event with timing stats
```

### Pull Model
```
Client → POST /api/pull
  → routes::pull_model()
  → creates mpsc channel
  → tokio::spawn(loader::pull())
  → streams PullProgress events via SSE
  → on complete: registry.register()
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **In-memory registry** | Simple, fast, no DB dependency. Models reload on restart from disk. |
| **Arc<RwLock<HashMap>>** | Standard Rust concurrent shared state pattern |
| **mpsc channels for progress** | Decouples download progress from SSE stream |
| **Word-level tokenization** | Stub only — real impl uses Candle tokenizer |
| **Candle over llama.cpp** | Pure Rust, no FFI, single-binary deployment, active development |
| **Model cache with per-model Mutex** | `RwLock<HashMap>` for fast reads, `AsyncMutex` per model so requests to different models run in parallel but same-model requests serialize safely |
| **Stub inference** | Allows API/UX development before inference backend |
| **Axum 0.6** | Stable, well-documented, tower ecosystem |
| **No TLS in server** | Delegate to reverse proxy (nginx/caddy) |
| **Ollama disk layout** | Migration compatibility with existing Ollama models |

## Evolution History

### ✅ Phase 2: Real Inference (Complete)
```
backend/:
  candle.rs → Candle (pure Rust), 5 architectures, model cache
  stub.rs → Fallback simulated inference
  InferenceBackend trait → generate(), chat(), embed()
```

### ✅ Phase 3: HTTPS Registry (Complete)
```
loader.rs:
  reqwest::Client + TLS
  registry.ollama.ai → real HTTPS downloads with blob caching
  OCI manifest persistence → find_gguf_blob for CandleBackend
  SSE progress stream
```

### 🔜 Phase 4: Embeddable Library
```
ollama-rs as a library dependency:
  use ollama_rs::{Ollama, GenerateRequest};
  let client = Ollama::new("http://localhost:11434");
  let resp = client.generate(GenerateRequest { ... }).await?;
```

---

*Generated: 2026-06-29*
