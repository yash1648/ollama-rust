# Multi-stage Docker build for ollama-rs
#
# Build:
#   docker build -t ollama-rs .
#
# Run:
#   docker run -d -p 11434:11434 --name ollama-rs ollama-rs
#
# Multi-arch (arm64, amd64):
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t ollama-rs:latest --push .
#
# Persistent model storage:
#   docker run -d -p 11434:11434 -v ollama-models:/home/rust/.ollama-rs/models ollama-rs

# ── Builder stage ────────────────────────────────────────────────────────────
FROM rust:alpine AS builder

# Install system dependencies for reqwest (native-tls via OpenSSL)
RUN apk add --no-cache \
    musl-dev \
    pkgconfig \
    openssl-dev \
    openssl-libs-static

WORKDIR /app

# Cache: copy dependency manifests first
COPY Cargo.toml Cargo.lock ./

# Create a dummy src/main.rs to build dependencies (cached layer)
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release 2>/dev/null || true

# Copy actual source and build for real
COPY src/ src/
COPY tests/ tests/

# Static linking for OpenSSL on musl
ENV OPENSSL_STATIC=1
ENV OPENSSL_LIBS="ssl:crypto"
RUN cargo build --release --bin ollama-rs

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM alpine:latest

# Install OpenSSL runtime libraries (needed by reqwest native-tls)
RUN apk add --no-cache \
    openssl \
    ca-certificates

# Create non-root user
RUN addgroup -S ollama && \
    adduser -S ollama -G ollama

# Copy the compiled binary from builder
COPY --from=builder /app/target/release/ollama-rs /usr/local/bin/ollama-rs

# Create model storage directory
RUN mkdir -p /home/ollama/.ollama-rs/models && \
    chown -R ollama:ollama /home/ollama/.ollama-rs

# Switch to non-root user
USER ollama

# Environment
ENV OLLAMA_HOST=0.0.0.0:11434
ENV RUST_LOG=ollama_rs=info,tower_http=debug

# Port (matching Ollama default)
EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:11434/api/health || exit 1

ENTRYPOINT ["ollama-rs"]
