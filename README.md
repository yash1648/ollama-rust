# rust-ollama

A minimal starter Rust application named `rust-ollama`.

> Current state: scaffolded crate with a `hello world` entrypoint in `src/main.rs`.

## Project overview

- **Name:** rust-ollama
- **Version:** 0.1.0
- **Edition:** 2024
- **Purpose:** Base template for building a Rust command-line app, potentially a client for Ollama (future expansion).

## Status

This repository is currently a barebones Rust project created via `cargo new`.

- `src/main.rs` prints `Hello, world!`
- `Cargo.toml` has no external dependencies

## Getting started

1. Clone repo:

```bash
git clone https://github.com/yash1648/ollama-rust.git
cd rust-ollama
```

2. Build:

```bash
cargo build
```

3. Run:

```bash
cargo run
```

Expected output:

```
Hello, world!
```

## Project structure

- `Cargo.toml` - Cargo project metadata
- `src/main.rs` - Main application entrypoint

## Contribution

1. Fork the repository
2. Create a feature branch `feature/<name>`
3. Commit with clean messages
4. Open a pull request

## Future extensions

- Add Ollama API client integration
- Add command-line interface via `clap`
- Add tests under `tests` and `src`
- Add docs and examples in `docs/`

## License

Currently not issued any license
