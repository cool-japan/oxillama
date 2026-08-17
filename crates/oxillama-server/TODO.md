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
| Workspace version | 0.1.4 |
| Tests | 357 passing, 1 skipped (`--all-features`); 341 passing, 0 skipped (default features) |
| Completion | ~99% complete (v0.1.4) — remaining gaps: client-disconnect mid-decode cancellation (§5), audio transcriptions, gRPC transport (§7) |
| Source files | 58 (`src/**/*.rs`) — 21 top-level + `routes/` (11), `admin/` (6), `batch_spool/` (5), `threads/` (8), `jwt_auth/` (4), `router/` (3) |
| Total SLoC | ~13,372 lines (tokei) |
| Framework | axum + tower + tower-http (tokio runtime) |
| Feature flag | `server` (default — enabled at workspace level); `jwt` (opt-in, default off) |
| Route handlers | chat/completions/embeddings/models/health/ready, WebSocket, batches (legacy in-memory `/v1/batches` + disk-spooled `/v1/batch_jobs`), files, threads (Assistants), responses, admin (incl. LoRA registry), metrics |
| Error type | `ServerError` → HTTP status via `IntoResponse` |
| Queue backend | `tokio::sync::mpsc` + per-request `oneshot` reply |
| Worker model | Single blocking thread owning `InferenceEngine` |

## 3. Module Map

| File | Role |
|---|---|
| `src/lib.rs` | Public re-exports: `build_app`, `ServerConfig`, `AppState`, `spawn_inference_worker`, `BatchRequest`, `VocabBytes`, `ServerError`, `ServerResult` |
| `src/app.rs` | axum `Router` wiring — mounts the full inference + admin route table (25+ endpoints) onto `AppState` |
| `src/config.rs` | `ServerConfig` DTO (host, port, max_concurrent, timeout_secs, cors_enabled) |
| `src/state.rs` | `AppState` — queue sender, model id, cached sampler, vocab bytes, hidden size |
| `src/queue.rs` | `BatchRequest` enum (`Generate` / `GenerateStream` / `Embed`), `VocabBytes`, `StreamCallback`, `ModelMeta` |
| `src/worker.rs` | `spawn_inference_worker` on `tokio::task::spawn_blocking`, drains queue, resets KV between requests |
| `src/sse.rs` | `SseEvent` helper — `data:` line formatting + `[DONE]` sentinel |
| `src/error.rs` | `ServerError` (thiserror) + OpenAI-shaped JSON error body mapped to HTTP status |
| `src/routes/mod.rs` | Route module index |
| `src/routes/chat.rs` | `POST /v1/chat/completions` — non-streaming + SSE (~1,390 lines) |
| `src/routes/completions.rs` | `POST /v1/completions` — prompt-based, non-streaming and streaming |
| `src/routes/embeddings.rs` | `POST /v1/embeddings` — single-string or batch input, L2-normalised vectors |
| `src/routes/models.rs` | `GET /v1/models` — single-model listing |
| `src/routes/health.rs` | `GET /health` / `GET /ready` — liveness + readiness probes |
| `src/routes/files.rs` | `/v1/files` — create/list/get/delete/content handlers (Files API) |
| `src/routes/metrics.rs` | `GET /metrics` — Prometheus text exposition |
| `src/routes/responses.rs` | `/v1/responses` — create/list/get, `previous_response_id` chaining, SSE |
| `src/routes/tools.rs` | OpenAI tool/function-calling types + JSON-Schema-to-GBNF grammar generation |
| `src/routes/tool_dispatcher.rs` | Server-side `ToolDispatcher` — dispatches `tools: [...]` from chat requests |
| `src/auth.rs` | Bearer-token authentication middleware (`ApiKeys`, `auth_middleware`) |
| `src/rate_limit.rs` | Token-bucket rate limiter (`RateLimiter`, `TokenBucket`, `rate_limit_middleware`) — global + per-key |
| `src/shutdown.rs` | Graceful shutdown (`shutdown_signal`, `ShutdownTrigger`, `ShutdownSignal`) |
| `src/test_helpers.rs` | In-memory app builder + live-worker fixture for integration tests |
| `src/batch.rs` | `/v1/batches` — legacy in-memory batch store (`HashMap` + `RwLock`); does **not** survive a restart |
| `src/metrics.rs` | Lock-free `AtomicU64` Prometheus-compatible counters/gauges — no external `prometheus` crate |
| `src/files_store.rs` | Disk-backed persistent store backing the Files API |
| `src/responses_store.rs` | In-memory store for Responses API objects (`ResponseStore`, `ResponseRecord`) |
| `src/prefix_registry.rs` | Namespaced, bounded registry of prefix-KV caches, keyed by `(model_id, lora_selection)` |
| `src/resource_id.rs` | Shared validation for request-supplied resource IDs used as filesystem path components (path-traversal guard) |
| `src/body_limit.rs` | `tower_http::limit::RequestBodyLimitLayer` wrapper, configurable max body size |
| `src/tracing_layer.rs` | Structured request-tracing middleware (method, path, status, latency_ms, request_id) |
| `src/ws.rs` | `GET /v1/chat/ws` — WebSocket streaming wired to the **same** `BatchRequest::GenerateStream` worker queue/engine as SSE (not a stub) |
| `src/admin/*.rs` (6 files) | `/admin/*` fleet management: load/unload/list/stats/health (`routes.rs`), LoRA registry (`loras.rs`), bearer/loopback auth (`auth.rs`), model-path allow-list guard (`path_guard.rs`) |
| `src/batch_spool/*.rs` (5 files) | `/v1/batch_jobs` — disk-spooled batch queue, the **persistent** batch backend; atomic tempfile writes, background worker, resumes `in_progress` jobs across restarts |
| `src/jwt_auth/*.rs` (4 files) | `jwt` feature (opt-in, default off) — Pure-Rust HS256/RS256 JWT verification + scope-based route authorization |
| `src/router/*.rs` (3 files) | Multi-model LRU warm-pool (`ModelPool`) — `pool.rs` acquire/evict, `eviction.rs` |
| `src/threads/*.rs` (8 files) | `/v1/threads` — OpenAI Assistants v2 subset: threads/messages/runs/steps, disk-persisted store, background run worker |

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

- ~~**`/v1/chat/completions` does not use the loaded model's actual chat
  template — it hardcodes one generic, model-agnostic template for every
  model.**~~ ✅ **Fixed.** `ChatTemplate`/`Turn`/`detect_from_jinja`/
  `detect_from_vocab`/`render_*` moved from `oxillama-cli/src/chat_template.rs`
  into `oxillama-runtime/src/chat_template.rs` (the CLI now re-exports them,
  keeping only its `SessionSnapshot`-coupled helpers); `AppState` gained a
  `chat_template` field resolved once at load time from
  `InferenceEngine::chat_template()`; and **all five** hand-rolled copies of
  the fabricated `<|system|>…<|end|>` skeleton — `routes/chat.rs`, `ws.rs`,
  `threads/worker.rs`, `routes/responses.rs`, `batch_spool/worker.rs` — now
  render through that one template. `BatchRequest` carries `add_special` so
  templates that emit their own literal BOS (`Llama3`, `Mistral`) do not get a
  duplicate one, and the prefix-cache lookup tokenizes with the same flag so a
  hit can never restore KV state for a differently-tokenized prompt.
  - **Repro verified fixed 2026-08-05** on the exact command from the original
    report (`oxillama serve --model Qwen3-4B-Instruct-2507-Q4_K_M.gguf
    --port 18099 --ctx-size 512`, then `POST /v1/chat/completions` with
    `"Say hello in one short sentence."`, `max_tokens:64`, `temperature:0`):
    `"content": "Hello! 😊"` — previously `"Hello! <|end|#>"`.
    Meta-Llama-3-8B-Instruct on the same request returns `"content": "Hello!"`.
  - **Coverage:** per-family render tests in `routes/chat.rs` and `ws.rs`
    (Llama3/ChatML/Mistral/Alpaca, plus a "no renderer emits the fabricated
    markers" assertion); end-to-end tests that prove the *dispatched* prompt
    (not just the pure render function) uses the served model's markers and
    the right `add_special`; and env-gated real-model tests in
    `tests/real_model_chat.rs` (`OXILLAMA_SERVER_QWEN3_GGUF` /
    `OXILLAMA_SERVER_LLAMA3_GGUF`) asserting a real reply contains no `<|`
    control-token text at all.

- ~~**`finish_reason` on `POST /v1/chat/completions` is hardcoded to
  `"stop"`.**~~ ✅ **Fixed.** The worker's reply channels are now typed
  `GenerateReply` / `GenerateStreamReply` (`src/queue.rs`), carrying the
  `FinishReason` the decode loop already computed but which
  `run_generate` used to discard by calling the bare-`String` runtime
  wrappers. `routes/chat.rs` (non-streaming and the trailing SSE chunk),
  `routes/completions.rs`, `ws.rs`, and the batch-spool output records all
  report it, so a generation truncated at `max_tokens` or by context
  exhaustion now yields `"length"`. `"tool_calls"` still overrides it when a
  tool call parsed, and `"cancelled"` still wins for an abandoned stream.
  Verified on both real models: `max_tokens:4` yields
  `{"content":"The sea is an","finish_reason":"length"}` (Qwen3) and
  `{"content":"The sea is a","finish_reason":"length"}` (Llama-3).

- **Client disconnect does not preempt an in-flight decode.** Now cheap to
  close, but deliberately left out of the chat-template/`finish_reason` fix so
  the two land separately. `GenerationConfig::cancel_flag`
  (`oxillama-runtime`) is an `Option<Arc<AtomicBool>>` checked once per decode
  iteration, and `BatchRequest::GenerateStream` already carries a
  `tokio_util::sync::CancellationToken` that the SSE/WS producers cancel the
  moment the client stops draining. Bridging them is a small, local change:
  spawn nothing, just allocate an `Arc<AtomicBool>` per streaming request,
  hand the clone to `GenerationConfig::cancel_flag` in `run_generate`, and
  raise it from a `tokio::spawn(async move { token.cancelled().await; flag.store(true) })`
  (or check `token.is_cancelled()` from the existing per-token callback, which
  already runs on the worker thread — cheaper still, no task). Today an
  in-flight decode for an abandoned client runs to completion and the result
  is discarded; after this it would stop at the next token boundary with
  `FinishReason::Cancelled`. Worth doing: on a 2048-token budget it is the
  difference between seconds and minutes of wasted worker time per abandoned
  request.
  - **Decide the wire string deliberately when this lands.**
    `FinishReason::Cancelled.as_openai_str()` returns `"cancelled"`, which is
    *not* in OpenAI's `finish_reason` vocabulary (`stop` / `length` /
    `tool_calls` / `content_filter`). Today that string is unreachable from
    any HTTP response, because nothing sets `cancel_flag` — the streaming
    routes special-case their own `"cancelled"` before consulting the runtime
    reason. The moment the bridge above exists, `Cancelled` starts flowing
    into `choices[0].finish_reason` on the **non-streaming** path too, where
    no such special case exists. Either keep `"cancelled"` (a deliberate
    OxiLLaMa extension, consistent with what the SSE/WS paths already emit and
    what a client actually needs to know) or map it to `"stop"` at the route
    boundary — but pick one on purpose rather than inheriting it.

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
  `AppState::prefix_cache_registry: Arc<PrefixCacheRegistry>` (since
  refactored from a single global `Arc<Mutex<PrefixKvCache>>` into a
  namespaced, bounded registry keyed by `(model_id, lora_selection)` —
  the D6 fix described above that prevents LoRA-contaminated cache
  hits), per-request `cache_prompt: bool` flag (default `true`),
  worker-side hit/miss/store logic calling `engine.prime_with_prefix` +
  `engine.generate_with_logits_detailed` on cache hit; full-prefill
  fallback on miss; `store_kv_in_prefix_cache` stores post-generation
  KV state.

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
  ✅ Done: `GET /v1/chat/ws` in `ws.rs` is wired to the **same**
  `BatchRequest::GenerateStream` worker queue/engine as the SSE chat
  route — real chat-template rendering, sampler config, usage stats,
  and `finish_reason`, plus `CancellationToken`-based shed-on-disconnect
  (no longer a stub token stream). `WsEvent` JSON framing (token / done
  / error). What remains open from the original goal is specifically
  bidirectional mid-response tool-output injection — the client cannot
  stream a partial tool result back into an in-flight generation.

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
  - **Tests:** 16 unit tests (13 in `verifier.rs`: HS256 sign/verify, expired, nbf, wrong aud/iss, malformed, alg:none; 3 in `scopes.rs`); RS256 sign/verify `#[ignore]`d (slow key gen).

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

## 8. v0.1.3 — Responses API + Per-Key Rate Limiting (done 2026-05-05)

- [x] **Responses API (`/v1/responses`)**
  - `POST /v1/responses` — create a response (non-streaming or SSE when `stream: true`)
  - `GET /v1/responses` — list all responses (descending `created_at`)
  - `GET /v1/responses/:id` — retrieve one response
  - `previous_response_id` chaining: prior input + output prepended to context
  - SSE event names: `response.created`, `response.output_text.delta`, `response.completed`, `[DONE]`
  - New module: `src/responses_store.rs` (`ResponseStore`, `ResponseRecord`, `ResponseStatus`)
  - New module: `src/routes/responses.rs` (all three route handlers)
  - `AppState::responses_store: Option<Arc<ResponseStore>>` + `with_responses_store()` builder
  - `ServerError::ResponseNotFound` + `ServerError::PreviousResponseNotFound` → HTTP 404

- [x] **Per-API-key rate limiting (`PerKeyRateLimiter`)**
  - `PerKeyRateLimiter` with lazy-insert bucket map (`Arc<RwLock<HashMap<String, Mutex<TokenBucket>>>>`)
  - `with_overrides(HashMap<String, (f64, f64)>)` builder for per-key capacity/rate
  - `check_key(&str) -> bool` — fast read-lock path + slow write-lock insertion
  - `per_key_rate_limit_middleware` reads `Authorization: Bearer` or `X-Api-Key`
  - Anonymous requests (no key header) pass through
  - `ServerConfig::per_key_rate_limits: Option<HashMap<String, (f64, f64)>>`
  - `AppState::per_key_rate_limiter: Option<Arc<PerKeyRateLimiter>>` + `with_per_key_rate_limiter()` builder
  - Layered in `build_app_with_config` when `state.per_key_rate_limiter.is_some()`
  - Re-exported from `lib.rs`: `PerKeyRateLimiter`, `ResponseStore`
  - 15 new tests (5 store unit + 5 route integration + 5 per-key unit)
  - Total test count: 236 (all passing)

*Last updated: 2026-08-17 (v0.1.4 — security hardening: `/admin/*` fail-closed loopback detection
(the guard previously existed but was only exercised inside test code), path-traversal fix across
all three disk stores, `serve` rewired to `build_app_with_config` (auth/rate-limiting/CORS/metrics
were previously all dead in the deployed binary), constant-time API-key comparison; `GET
/admin/health` now reports GPU backend status; 357 tests, 1 skipped)*
