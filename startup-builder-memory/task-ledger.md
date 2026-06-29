# Task Ledger: ollama-rs

## Progress

```
████▓░░░░░░░░░░░░░  [ 1/5 milestones ]  Milestone 2: 9/10 tasks
```

## Milestones

### 🏁 Milestone 1: MVP Core — API Server Foundation ✅
> *Status: Complete*

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Scaffold Rust project structure | ✅ | Cargo.toml, modules |
| 1.2 | Implement API request/response types | ✅ | All 20+ structs in types.rs |
| 1.3 | Build Axum HTTP server with middleware | ✅ | CORS, tracing, 0.0.0.0:11434 |
| 1.4 | Implement model registry (in-memory + disk) | ✅ | RwLock<HashMap> + JSON manifests |
| 1.5 | Implement all route handlers | ✅ | 15 endpoints, SSE streaming |
| 1.6 | Build model storage (manifests + blobs) | ✅ | ~/.ollama-rs/models/ layout |
| 1.7 | Implement stub inference engine | ✅ | Simulated token generation |
| 1.8 | Add OpenAI-compatible endpoints | ✅ | /v1/models, /v1/chat/completions |
| 1.9 | Implement model pulling with progress | ✅ | Simulated pull, mpsc progress stream |
| 1.10 | Create README with usage docs | ✅ | |

### 🏁 Milestone 2: Hardening & Real Infrastructure
> *Status: In Progress (7/10 tasks)*

| # | Task | Status | Effort | Notes |
|---|------|--------|--------|-------|
| 2.1 | Remove `#![allow(dead_code, ...)]` from main.rs | ✅ | Small | Fixed 24 warnings across 8 files |
| 2.2 | Implement real model pulling via HTTPS | ✅ | Medium | reqwest + OCI distribution spec; downloads blobs with SSE progress |
| 2.3 | Add configurable host/port (OLLAMA_HOST) | ✅ | Small | Env var, default 0.0.0.0:11434 |
| 2.4 | Add graceful shutdown (signal handling) | ✅ | Small | SIGINT + SIGTERM handler |
| 2.5 | Add health check endpoint (/api/health) | ✅ | Small | Returns `{"status": "ok"}` |
| 2.6 | Add unit tests for registry | ✅ | Medium | 19 unit tests (normalize, split, CRUD) |
| 2.7 | Add integration tests for API endpoints | ✅ | Medium | 17 integration tests (all endpoints) |
| 2.8 | Set up CI/CD (GitHub Actions) | ✅ | Medium | Build + test + clippy + fmt on push/PR |
| 2.9 | Add clippy linting + fix warnings | ✅ | Small | Fixed all clippy warnings, zero-clippy verified with `-D clippy::all` |
| 2.10 | Create Dockerfile + multi-arch build | ⬜ | Medium | Alpine-based |

### 🏁 Milestone 3: Real Inference Engine
> *Status: Not Started*

| # | Task | Est. Effort | Deps | Notes |
|---|------|-------------|------|-------|
| 3.1 | Research inference backends (FFI vs subprocess vs candle) | Medium | — | Trade-off analysis |
| 3.2 | Implement subprocess-based llama.cpp integration | Medium | — | Spawn llama-cli, parse output |
| 3.3 | Implement candle-native inference (pure Rust) | Large | — | Alternative backend |
| 3.4 | Add model loading/unloading to RAM | Medium | 3.2 | Cache loaded models |
| 3.5 | Implement proper tokenization (tiktoken-rs?) | Medium | — | Replace word-level stub |
| 3.6 | Implement context window management | Medium | 3.5 | kv-cache, sliding window |
| 3.7 | Add GPU detection + device selection | Small | — | Metal/CUDA/RocM |

### 🏁 Milestone 4: Tooling & Developer Experience
> *Status: Not Started*

| # | Task | Est. Effort | Deps | Notes |
|---|------|-------------|------|-------|
| 4.1 | Create CLI with clap or similar | Medium | — | `ollama-rs serve`, `ollama-rs pull` |
| 4.2 | Add progress bars for CLI pull | Small | 4.1 | indicatif integration |
| 4.3 | Implement `ollama-rs` as a library crate | Medium | — | Separate lib.rs from main.rs |
| 4.4 | Publish to crates.io | Small | 4.3 | |
| 4.5 | Add comprehensive benchmarks | Medium | — | latency, throughput, memory |
| 4.6 | Add logging for inference events | Small | — | per-request timing |

### 🏁 Milestone 5: Production Readiness
> *Status: Not Started*

| # | Task | Est. Effort | Deps | Notes |
|---|------|-------------|------|-------|
| 5.1 | Add TLS support (rustls via axum) | Medium | — | Optional, configurable |
| 5.2 | Add rate limiting middleware | Medium | — | Token bucket / leaky bucket |
| 5.3 | Add metrics endpoint (Prometheus) | Medium | — | Request count, latency histograms |
| 5.4 | Add structured error messages matching Ollama | Small | — | Error format parity |
| 5.5 | Implement concurrent request queue | Medium | — | Fair scheduling across models |
| 5.6 | Add multi-model support (load multiple) | Medium | — | RAM management |
| 5.7 | Fuzz testing for API parsing | Medium | — | cargo-fuzz |
| 5.8 | Security audit | Medium | — | Path traversal, injection |

## Current Focus

### Batch 1 Complete: Core Hardening (feat/harden-core)

| Task | Result |
|------|--------|
| 2.1 — Remove lint suppressions | ✅ Zero warnings on `cargo build` |
| 2.3 — OLLAMA_HOST env var | ✅ Configurable bind address |
| 2.4 — Graceful shutdown | ✅ SIGINT + SIGTERM handling |
| 2.5 — Health check endpoint | ✅ `GET /api/health` → `{"status": "ok"}` |

### Batch 2 Complete: Test Suite (feat/test-suite)

| Task | Result |
|------|--------|
| 2.6 — Unit tests for registry | ✅ 19 tests — normalize_name, split_name, register, get, exists, remove, copy, edge cases |
| 2.7 — Integration tests for API | ✅ 17 tests — health, version, CRUD models, inference errors, OpenAI compat, push/ps |
| Library target added | ✅ `src/lib.rs` with re-exports for external consumers |

### Batch 3 Complete: Real Model Pulling (feat/real-pull)

| Task | Result |
|------|--------|
| 2.2 — Real model pulling via HTTPS | ✅ reqwest + OCI dist spec, streaming blob download with SSE progress, blob caching, manifest persistence |

### Batch 4 Complete: CI/CD + Clippy + Fmt (feat/ci-cd)

| Task | Result |
|------|--------|
| 2.8 — CI/CD (GitHub Actions) | ✅ Build, test, clippy, fmt — runs on push/PR to master |
| 2.9 — Clippy linting | ✅ Zero warnings across all targets with `-D clippy::all` |
| cargo fmt | ✅ All source files formatted, `cargo fmt --check` passes |

**Next recommended feature**: Milestone 2 task 2.10 — Dockerfile + multi-arch build

## Task Status Key
- ✅ Complete
- 🔄 In Progress
- ⏳ Blocked
- ❌ Failed
- ⬜ Not Started

---

*Generated: 2026-06-29*
