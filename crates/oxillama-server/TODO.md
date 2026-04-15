# oxillama-server — TODO

## 1. Overview

`oxillama-server` is the OpenAI-compatible HTTP API surface for the
OxiLLaMa stack. It is designed as a drop-in replacement for upstream
`llama-server`, speaking the same wire contract (JSON requests, SSE
streaming, OpenAI-shaped responses) while delegating every inference
call to `oxillama-runtime`.

The crate follows a strict queue + worker architecture: route handlers
never touch the `InferenceEngine` directly. Each request is converted
into a `BatchRequest` variant, dispatched through a bounded
`tokio::sync::mpsc` channel, and picked up by a single blocking worker
thread that owns the engine exclusively. This removes all mutex
contention between concurrent connections.

The server is built entirely on Pure Rust foundations (axum, tower,
tokio, tower-http, serde_json, thiserror, tracing). It ships behind
the workspace-level `server` feature (default on) and carries zero
C/C++/Fortran dependencies — fully compliant with the COOLJAPAN Pure
Rust Policy.

## 2. Status Snapshot

| Item | Value |
|---|---|
| Workspace version | 0.1.0 |
| Completion | ~98% toward v0.1.0 API parity |
| Source files | 18 (`src/*.rs` + `src/routes/*.rs`) |
| Total SLoC | ~2,600 lines |
| Framework | axum + tower + tower-http (tokio runtime) |
| Feature flag | `server` (default — enabled at workspace level) |
| Route handlers | 5 shipped (chat, completions, embeddings, models, health) |
| Error type | `ServerError` → HTTP status via `IntoResponse` |
| Queue backend | `tokio::sync::mpsc` + per-request `oneshot` reply |
| Worker model | Single blocking thread owning `InferenceEngine` |

## 3. Module Map

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports: `build_app`, `ServerConfig`, `AppState`, `spawn_inference_worker`, `BatchRequest`, `VocabBytes`, `ServerError`, `ServerResult` |
| `src/app.rs` | axum `Router` wiring — mounts all five routes onto `AppState` |
| `src/config.rs` | `ServerConfig` DTO (host, port, max_concurrent, timeout_secs, cors_enabled) |
| `src/state.rs` | `AppState` — queue sender, model id, cached sampler, vocab bytes, hidden size |
| `src/queue.rs` | `BatchRequest` enum (`Generate` / `GenerateStream` / `Embed`), `VocabBytes`, `StreamCallback`, `ModelMeta` |
| `src/worker.rs` | `spawn_inference_worker` on `tokio::task::spawn_blocking`, drains queue, resets KV between requests |
| `src/sse.rs` | `SseEvent` helper — `data:` line formatting + `[DONE]` sentinel |
| `src/error.rs` | `ServerError` (thiserror) + OpenAI-shaped JSON error body mapped to HTTP status |
| `src/routes/mod.rs` | Route module index |
| `src/routes/chat.rs` | `POST /v1/chat/completions` — non-streaming + SSE (~630 lines) |
| `src/routes/completions.rs` | `POST /v1/completions` — prompt-based, non-streaming and streaming |
| `src/routes/embeddings.rs` | `POST /v1/embeddings` — single-string or batch input, L2-normalised vectors |
| `src/routes/models.rs` | `GET /v1/models` — single-model listing |
| `src/routes/health.rs` | `GET /health` — liveness probe, version string |
| `src/auth.rs` | Bearer-token authentication middleware (`ApiKeys`, `auth_middleware`) |
| `src/rate_limit.rs` | Token-bucket rate limiter (`RateLimiter`, `TokenBucket`, `rate_limit_middleware`) |
| `src/shutdown.rs` | Graceful shutdown (`shutdown_signal`, `ShutdownTrigger`, `ShutdownSignal`) |
| `src/test_helpers.rs` | In-memory app builder + live-worker fixture for integration tests |

## 4. Shipped in v0.1.0

OpenAI route coverage (v0.1.0):

| Route | Method | Status | Notes |
|---|:-:|:-:|---|
| `/v1/chat/completions` | POST | OK | SSE streaming |
| `/v1/completions` | POST | OK | SSE streaming |
| `/v1/embeddings` | POST | OK | single + batch input |
| `/v1/models` | GET | OK | single-model list |
| `/health` | GET | OK | liveness probe |
| `/v1/chat/completions` (tools) | POST | OK | function/tool calling |
| `/metrics` | GET | OK | lock-free AtomicU64 counters |
| `/v1/batches` | POST | pending | v2.0 — batch API |
| `/v1/audio/transcriptions` | POST | pending | v2.0 — requires whisper arch |

Implementation highlights:

- `POST /v1/chat/completions` with full SSE streaming: initial role
  chunk, per-token `delta.content` chunks, trailing `finish_reason`
  chunk, terminating `[DONE]` sentinel.
- `POST /v1/completions` with both non-streaming JSON and SSE modes.
- `POST /v1/embeddings` with `EmbeddingInput::{Single, Batch}` untagged
  enum, L2-normalised vectors of length `hidden_size`.
- `GET /v1/models` — OpenAI-shaped `{ "object": "list", "data": [...] }`.
- `GET /health` — returns `{ "status": "ok", "version": <pkg ver> }`.
- SSE framing: `data: <json>\n\n` lines plus `data: [DONE]\n\n` sentinel.
- llama.cpp CLI flag aliases surfaced through the `oxillama-cli` crate:
  `-n/--n-predict`, `--temperature`, `-c/--n-ctx`, `--seed`,
  `--repeat-penalty`, `--min-p` — all map onto `SamplerConfig` and
  `EngineConfig` fields consumed by the server at startup.
- Queue + worker backing every route — continuous batching via the
  `oxillama-runtime` scheduler, single worker owning the engine.
- CORS middleware (tower-http) — configurable via `ServerConfig`.
- Live-worker integration tests: 7 tests covering the success path for
  embeddings (single + batch) and `/v1/models` / `/health`.
- SSE streaming chat tests: 3 tests validating chunk order and
  `[DONE]` termination.
- Structured error mapping — `ServerError` → OpenAI JSON error envelope
  with `type` field: `invalid_request_error` (400),
  `service_unavailable` (503), `rate_limit_error` (429),
  `internal_error` (500).
- GBNF grammar on `/v1/chat/completions` via the optional `grammar`
  field (`Grammar::parse` in the runtime).
- Bearer-token authentication middleware — `ApiKeys` type,
  `auth_middleware` function, configurable key rotation via
  `ServerConfig.api_keys`.
- Token-bucket rate limiter — global request-rate limiter with
  configurable burst and refill rate, 429 with `Retry-After` header.
- Graceful shutdown — `shutdown_signal()` for SIGTERM/Ctrl-C +
  programmatic `ShutdownTrigger`/`ShutdownSignal` for testing.
- Usage accounting — `UsageStats { prompt_tokens, completion_tokens,
  total_tokens }` populated from actual engine tokenization in
  `Generate`, `GenerateStream`, and route responses.
- `build_app_with_config()` — wires auth + rate limiting layers based
  on `ServerConfig`.
- Request body-size limits (configurable, default 10 MiB).
- Prometheus `/metrics` endpoint (lock-free AtomicU64 counters).
- Structured tracing middleware (method, path, status, latency_ms,
  request_id).
- Function / tool calling: `Tool`, `FunctionDef`, `ToolChoice`, `ToolCall`
  types with JSON Schema → GBNF grammar conversion (`tools_to_gbnf`),
  tool call parsing (`parse_tool_call_output`), streaming tool call deltas
  (`ToolCallDelta`, `FunctionCallDelta`), and updated `ChatMessage` for
  tool roles.

## 5. Known Gaps / Incomplete

- ~~**Function / tool calling.**~~ ✅ Shipped. The chat schema accepts
  `tools` and `tool_choice` fields; `tool_calls` are emitted in both
  streaming and non-streaming responses.
- **Batch API.** No `/v1/batches` (OpenAI long-running job API).
- **WebSocket.** No WebSocket transport alternative to SSE; SSE is the
  only streaming option (sufficient for OpenAI parity, but limits
  bidirectional tool streaming).

## 6. v1.1 Roadmap

- ~~**OpenAI function / tool calling.**~~ ✅ Shipped.
  `tools` and `tool_choice` on `ChatCompletionRequest` accepted. JSON
  Schema to GBNF grammar conversion, tool-call boundary detection in
  stream, `tool_calls` arrays in both streaming and non-streaming
  responses.

## 7. v2.0+ Vision

- **Multi-model router.**
  Host multiple models side-by-side. `AppState` becomes a registry
  keyed by model id. Requests dispatch to the matching worker based
  on the `model` field of the request. Admin API to hot-load and
  hot-unload models at runtime (`POST /admin/models`,
  `DELETE /admin/models/{id}`).

- **Batch API.**
  `/v1/batches` endpoint for long-running offline jobs. File upload,
  async execution over hours, downloadable result file — mirroring
  the OpenAI batch API contract. Backed by a disk-spool queue rather
  than the in-memory mpsc.

- **Assistants API subset.**
  `POST /v1/threads`, `POST /v1/threads/{id}/messages`,
  `POST /v1/threads/{id}/runs` with persistent thread storage. Target
  parity with the OpenAI assistants v2 specification.

- **WebSocket streaming.**
  Full-duplex streaming alongside SSE for bidirectional tool
  invocation — the client can stream partial tool outputs back into
  the generation mid-response.

- **Audio transcriptions.**
  `POST /v1/audio/transcriptions` once a Whisper architecture lands
  in `oxillama-arch`. Multipart upload, text + verbose-json response
  formats.

- **gRPC alternative transport.**
  Mirror the HTTP surface behind a `tonic` server for lower-overhead
  internal deployments. Same `AppState` and queue — new framing only.

- **JWT auth with scopes.**
  Move beyond static bearer keys: signed JWTs carrying scopes
  (`chat:read`, `embed:read`, `admin:write`) enforced per-route.

- **Admin API.**
  `/admin/models/{id}` for runtime model swap, `/admin/loras/{id}`
  for multi-LoRA slot management once runtime-level hot-swap lands,
  `/admin/config` for live tuning of sampler defaults.

*Last updated: 2026-04-15 (v0.1.0 release)*
