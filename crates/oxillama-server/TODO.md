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
| Workspace version | 0.1.1 |
| Tests | 115 passing |
| Completion | ~98% toward v0.1.1 API parity |
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
| `/v1/batches` | POST/GET | OK | in-memory batch store; create, list, retrieve, cancel |
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
- ~~**Batch API.**~~ ✅ `/v1/batches` implemented (in-memory store, create/list/retrieve/cancel).
- ~~**WebSocket.** No WebSocket transport alternative to SSE; SSE is the
  only streaming option (sufficient for OpenAI parity, but limits
  bidirectional tool streaming).~~ ✅ `GET /v1/chat/ws` implemented.

## 6. v1.1 Roadmap

- ~~**OpenAI function / tool calling.**~~ ✅ Shipped.
  `tools` and `tool_choice` on `ChatCompletionRequest` accepted. JSON
  Schema to GBNF grammar conversion, tool-call boundary detection in
  stream, `tool_calls` arrays in both streaming and non-streaming
  responses.

- ~~**Server-side prefix-KV cache wiring.**~~ ✅ Shipped in v0.1.3.
  `AppState::prefix_cache: Arc<Mutex<PrefixKvCache>>`, per-request
  `cache_prompt: bool` flag (default `true`), worker-side hit/miss/store
  logic calling `engine.prime_with_prefix` + `engine.generate_with_logits`
  on cache hit; full-prefill fallback on miss; `store_kv_in_prefix_cache`
  stores post-generation KV state.

- ~~**Multi-LoRA per-request registry + admin CRUD.**~~ ✅ Shipped in v0.1.3.
  `AppState::loras: Arc<RwLock<HashMap<String, Arc<LoadedLora>>>>`;
  `lora_selection: Vec<(String, f32)>` on `BatchRequest`; chat route
  parses `"lora": "name"` or `"lora": [{"name": "...", "scale": 0.8}]`,
  resolves → 400 on unknown name; `POST /admin/loras` (load GGUF + register),
  `DELETE /admin/loras/{name}`, `GET /admin/loras`; `engine.unapply_all_loras()`
  restores base weights after generation.

## 7. v2.0+ Vision

- [x] **Multi-model router (LRU warm-pool) (done 2026-04-20)**
  - **Goal:** Single server binary holds K loaded models in a warm pool; routes incoming requests by `model` field; evicts least-recently-used model under memory pressure; pre-loads N models on startup.
  - **Design:**
    - New module `crates/oxillama-server/src/router/{mod,pool,eviction}.rs`.
    - `ModelPool { models: HashMap<ModelId, Arc<RwLock<LoadedModel>>>, lru: Mutex<VecDeque<ModelId>>, capacity: usize, total_memory_budget: usize }`.
    - `LoadedModel { engine: Engine, last_used: Instant, mem_bytes: usize }`.
    - `pool.acquire(model_id)` — if loaded, mark MRU and return Arc; else load (evicting LRU until under budget).
    - Eviction: pop LRU, drop its `Engine` (Rust drops state pool, KV pool, weights).
    - Concurrent: `RwLock<LoadedModel>` so multiple requests to same model share the engine (continuous-batching handles them).
    - Config: `[router] capacity = 4, mem_budget_mb = 16384, preload = [...]` in server.toml.
    - `--model <path>` CLI flag still works (1-slot router).
    - Memory formula: `mem_bytes ≈ weights_size + max_batch * (kv_size_per_seq + state_size_per_seq)`.
  - **Files:** `crates/oxillama-server/src/router/{mod,pool,eviction}.rs` (new, ~700 LoC); `crates/oxillama-server/src/state.rs` (replace single-engine field with `ModelPool`); `crates/oxillama-server/src/openai_chat.rs` and other endpoints (acquire from pool).
  - **Prerequisites:** none.
  - **Tests:** (a) `router_single_model_routes_correctly`. (b) `router_evicts_lru_under_pressure`. (c) `router_preload_works`. (d) `router_concurrent_requests_share_engine`. (e) `router_unknown_model_404`.
  - **Risk:** Memory underestimation causes OOM. Expose formula via admin API.

- [x] **Batch API disk-spool backend (done 2026-04-20)**
  - **Goal:** OpenAI-compatible `/v1/batches` endpoint backed by a disk-spooled job queue. Batches persist across server restarts; processed in background; results downloadable.
  - **Design:**
    - New module `crates/oxillama-server/src/batch/{mod,queue,worker,store}.rs`.
    - Job storage: `<batch_dir>/<job_id>/{input.jsonl, status.json, output.jsonl, errors.jsonl}`. Atomic writes via `tempfile::NamedTempFile::persist`.
    - Worker pool: configurable N workers; each processes one job at a time, line-by-line; writes outputs incrementally.
    - `POST /v1/batches` `{ input_file_id, endpoint, completion_window }` → 200 with batch object (id, status: `validating`).
    - `GET /v1/batches/:id` → status + counts. `GET /v1/batches/:id/output` → stream output JSONL. `POST /v1/batches/:id/cancel`. `GET /v1/batches` → paginated list.
    - Reuse existing `/v1/files` endpoint to accept input.jsonl.
    - Persistence on restart: scan `<batch_dir>/*` for `in_progress` and resume.
    - Limits: `max_batch_size_lines` (50000), `max_total_pending_bytes` (1 GB); reject with 413.
  - **Files:** `crates/oxillama-server/src/batch/{mod,queue,worker,store}.rs` (new, ~1000 LoC); `crates/oxillama-server/src/openai_files.rs` (extend if not present); `crates/oxillama-server/src/config.rs` (batch dir, worker count).
  - **Prerequisites:** C1 (worker uses model pool).
  - **Tests:** (a) `batch_submit_process_complete`. (b) `batch_persistence_across_restart`. (c) `batch_cancel_mid_flight`. (d) `batch_concurrent_jobs_dont_interleave_outputs`.
  - **Risk:** Disk fills under unbounded submission; enforce limits.

- [x] **Assistants API subset (done 2026-05-05)**
  `POST /v1/threads`, `GET /v1/threads/{id}`,
  `POST /v1/threads/{id}/messages`, `GET /v1/threads/{id}/messages`,
  `POST /v1/threads/{id}/runs`, `GET /v1/threads/{id}/runs/{run_id}`,
  `POST /v1/threads/{id}/runs/{run_id}/cancel`.
  Persistent thread/message/run storage with atomic disk writes
  (tempfile + rename), append-only JSONL message log, background run
  worker reusing chat-template prompt formatting.  199 tests all pass.

- ~~**WebSocket streaming.**~~
  ~~Full-duplex streaming alongside SSE for bidirectional tool~~
  ~~invocation — the client can stream partial tool outputs back into~~
  ~~the generation mid-response.~~
  ✅ Done: `GET /v1/chat/ws` endpoint added in `ws.rs`; stub token
  stream with proper `WsEvent` JSON framing (token / done / error).
  Real inference integration pending full engine hookup.

- **Audio transcriptions.**
  `POST /v1/audio/transcriptions` once a Whisper architecture lands
  in `oxillama-arch`. Multipart upload, text + verbose-json response
  formats.

- **gRPC alternative transport.**
  Mirror the HTTP surface behind a `tonic` server for lower-overhead
  internal deployments. Same `AppState` and queue — new framing only.

- [x] **JWT auth with scopes (done 2026-04-24)**
  Move beyond static bearer keys: signed JWTs carrying scopes
  (`chat:read`, `embed:read`, `admin:write`) enforced per-route.
  - **Feature flag:** `jwt` (opt-in, default off).
  - **Algorithms:** HS256 (constant-time HMAC-SHA256), RS256 (RSA PKCS1v15 + SHA-256, DER public key).
  - **Security:** `alg: "none"` always rejected. Only configured algorithms accepted.
  - **Module:** `crates/oxillama-server/src/jwt_auth/{mod,scopes,verifier,middleware}.rs`.
  - **Tests:** 15 unit tests (HS256 sign/verify, expired, nbf, wrong aud/iss, malformed, alg:none, scopes); RS256 `#[ignore]`d (slow key gen).

- [x] **Admin API (load/unload/status) (done 2026-04-20)**
  - **Goal:** HTTP endpoints under `/admin/*` for fleet management. Bound to `127.0.0.1` by default; optional bearer-token auth.
  - **Design:**
    - New module `crates/oxillama-server/src/admin/{mod,routes,auth}.rs`.
    - `POST /admin/models/load` `{ "id": "...", "path": "...", "quant": "..." }` → 202 Accepted, returns load token.
    - `POST /admin/models/unload` `{ "id": "..." }` → 200 OK.
    - `GET /admin/models` → list with `{id, mem_bytes, last_used, inflight_requests}`.
    - `GET /admin/stats` → requests/sec, p50/p95/p99 latency, queue depths.
    - `GET /admin/health` → extend existing `/health` with model pool readiness.
    - Auth: `[admin] bearer_token = "..."` in server.toml. If set: all `/admin/*` require `Authorization: Bearer <token>`. If unset: `/admin/*` only listens on loopback.
    - Atomic load: 202 immediately; load in background task; poll `GET /admin/models` for `loading | ready | failed`.
    - Hard error at startup if `admin_listen` is non-loopback AND no token configured.
  - **Files:** `crates/oxillama-server/src/admin/{mod,routes,auth}.rs` (new, ~500 LoC); server main/router setup (mount admin router); `crates/oxillama-server/src/config.rs` (admin config block).
  - **Prerequisites:** C1 (router exists to administer).
  - **Tests:** (a) `admin_load_unload_cycle`. (b) `admin_bearer_auth_rejects_missing_token`. (c) `admin_loopback_only_when_no_auth`. (d) `admin_stats_returns_metrics`.
  - **Risk:** Non-auth + public interface = full fleet control to anyone. Mitigate with hard startup error.

*Last updated: 2026-05-03 (v0.1.3 — ~190 tests, prefix-KV + multi-LoRA server wiring shipped)*
