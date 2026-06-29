# Product Brief: ollama-rs

## Overview
A complete **Ollama-compatible API server** written in Rust. Drop-in replacement for the Ollama REST API — any client that talks to Ollama works with this server unchanged.

## Core Problem
Ollama's Go-based server requires the full Go toolchain and has limited embeddability. A Rust-native implementation provides:
- Single-binary deployment (static linkage)
- Embeddability as a library (`ollama-rs` as a crate)
- Lower resource footprint
- Easier cross-compilation
- Async Rust ecosystem (tokio, axum) for high concurrency

## Target Users
| Persona | Need |
|---------|------|
| **Rust developers** | Embed Ollama-compatible inference in Rust apps |
| **Self-hosters** | Lightweight server for Edge/ARM/limited-resource environments |
| **Ollama users** | Drop-in replacement with identical API behavior |
| **ML pipeline devs** | Programmatic model management (pull, push, create) |

## Core User Journeys

### 1. Run a model (happy path)
1. Start `ollama-rs` server
2. Pull a model (`POST /api/pull`)
3. Generate text (`POST /api/generate`) or chat (`POST /api/chat`)
4. Get streaming SSE responses
5. Stop server

### 2. Model management
1. List models (`GET /api/tags`)
2. Show model details (`POST /api/show`)
3. Create from Modelfile (`POST /api/create`)
4. Copy (`POST /api/copy`)
5. Delete (`DELETE /api/delete`)

### 3. OpenAI-compatible usage
1. List models (`GET /v1/models`)
2. Chat completions (`POST /v1/chat/completions`)
3. Any OpenAI SDK client works transparently

## MVP Scope (v0.1.0)

### In Scope
- ✅ Full Ollama REST API (v1 compatible)
- ✅ SSE streaming for `/api/generate` and `/api/chat`
- ✅ `/v1/models` and `/v1/chat/completions` OpenAI compat
- ✅ Model management: list, pull, push, delete, copy, create, show, ps
- ✅ Embeddings endpoint
- ✅ Model persistence to `~/.ollama-rs/models/`
- ✅ In-memory model registry (RwLock)
- ✅ CORS + tracing middleware
- ✅ Stub inference engine (simulated tokens)

### MVP Scope — To Complete
- [ ] **Real inference backend** — llama.cpp FFI, candle, or subprocess
- [x] **Real model pulling** — HTTPS download from registry.ollama.ai
- [ ] **Configurable host/port** — `OLLAMA_HOST` env var support
- [ ] **Graceful shutdown** — signal handling
- [ ] **Health check endpoint** — `/api/health`
- [ ] **Test suite** — unit + integration tests
- [ ] **CI/CD** — GitHub Actions
- [ ] **Docker image** — multi-arch

### Non-Goals (v0.1.0)
- Model training / fine-tuning
- Web UI
- Authentication / multi-user
- Cluster / distributed inference
- GPU scheduling

## Technical Constraints
| Constraint | Detail |
|------------|--------|
| Language | Rust (edition 2021) |
| HTTP Framework | Axum 0.6 |
| Async Runtime | Tokio (full features) |
| Serialization | Serde JSON |
| Port | 11434 (Ollama default) |
| Domain | `0.0.0.0` |
| TLS | Not in MVP (add via reverse proxy) |

## Success Metrics
- **API parity**: Pass Ollama's client integration tests
- **Latency**: < 50ms overhead vs Ollama server (on same model)
- **Memory**: < 10MB idle (vs ~30MB for Ollama)
- **Binary size**: < 15MB stripped

## Risks
| Risk | Mitigation |
|------|------------|
| llama.cpp FFI is complex | Start with subprocess, graduate to FFI |
| No TLS in MVP | Document nginx/caddy reverse proxy setup |
| Ollama API drift | Pin to known working version, integration tests |
| Storage format changes | Compatible with `~/.ollama/models/` layout |

---

*Generated: 2026-06-29*
