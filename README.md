# ollama-rs

A complete **Ollama-compatible API server** written in Rust. Drop-in replacement for the Ollama REST API — any client that talks to Ollama works with this server unchanged.

## Features

| Category | Details |
|----------|---------|
| **API** | Full Ollama REST API (v1 compatible) |
| **Streaming** | SSE streaming for `/api/generate` and `/api/chat` |
| **OpenAI compat** | `/v1/models` and `/v1/chat/completions` |
| **Model mgmt** | list, pull, push, delete, copy, create, show, ps |
| **Embeddings** | `/api/embeddings` endpoint |
| **Persistence** | Models stored in `~/.ollama-rs/models/` |
| **Port** | `11434` (same as Ollama) |

## Architecture

```
src/
├── main.rs              # Entry point, tracing setup
├── api/mod.rs           # Public re-exports
├── model/
│   ├── types.rs         # All request/response types (serde)
│   ├── registry.rs      # In-memory model registry (RwLock)
│   └── loader.rs        # Pull flow + progress streaming
├── server/
│   ├── mod.rs           # Axum server setup, CORS, tracing
│   ├── state.rs         # Shared AppState (Arc)
│   ├── routes.rs        # All API route handlers
│   ├── inference.rs     # Inference engine (stub / FFI hook)
│   └── error.rs         # ApiError → HTTP response
└── storage/mod.rs       # Disk persistence (~/.ollama-rs/)
```

## Build & Run

```bash
cargo build --release
./target/release/ollama-rs
```

Server starts on `http://0.0.0.0:11434`.

## API Endpoints

### Model Management

```bash
# List models
curl http://localhost:11434/api/tags

# Pull a model (streaming progress)
curl -X POST http://localhost:11434/api/pull \
  -d '{"name": "llama3:8b"}'

# Show model info
curl -X POST http://localhost:11434/api/show \
  -d '{"name": "llama3:8b"}'

# Copy model
curl -X POST http://localhost:11434/api/copy \
  -d '{"source": "llama3:8b", "destination": "my-llama:latest"}'

# Delete model
curl -X DELETE http://localhost:11434/api/delete \
  -d '{"name": "llama3:8b"}'

# Create from Modelfile
curl -X POST http://localhost:11434/api/create \
  -d '{"name": "mymodel", "modelfile": "FROM llama3:8b\nSYSTEM You are a pirate."}'
```

### Inference

```bash
# Generate (streaming)
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3:8b", "prompt": "Why is the sky blue?"}'

# Generate (non-streaming)
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3:8b", "prompt": "Hello", "stream": false}'

# Chat
curl -X POST http://localhost:11434/api/chat \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Embeddings
curl -X POST http://localhost:11434/api/embeddings \
  -d '{"model": "llama3:8b", "prompt": "Hello world"}'
```

### OpenAI-Compatible

```bash
# List models
curl http://localhost:11434/v1/models

# Chat completions
curl -X POST http://localhost:11434/v1/chat/completions \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Plugging in Real Inference

The `src/server/inference.rs` module is the hook point. Replace `generate_tokens()` and `chat_tokens()` with:

- **llama.cpp via subprocess**: spawn `./llama-cli -m model.gguf -p prompt`
- **llama.cpp via FFI**: use `llama_sys` crate, call `llama_eval()`
- **candle**: pure-Rust inference with `candle-core` + `candle-transformers`
- **burn**: training/inference framework

Example FFI hook location in `inference.rs`:

```rust
pub async fn generate_tokens(req: &GenerateRequest) -> Vec<String> {
    // Replace this with: llama_cpp_ffi::generate(&req.model, &req.prompt)
    todo!("wire up llama.cpp FFI here")
}
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `RUST_LOG` | `ollama_rs=info` | Log level |
| `OLLAMA_HOST` | `0.0.0.0:11434` | Bind address (future) |

## Model Storage

Models are stored under `~/.ollama-rs/models/`:

```
~/.ollama-rs/
└── models/
    ├── manifests/          # JSON metadata per model
    │   └── llama3_8b.json
    └── blobs/              # GGUF weight files (sha256 named)
        └── a1b2c3d4...
```

Compatible with `~/.ollama/models/` layout for easy migration.

## License

MIT
