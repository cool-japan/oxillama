# oxillama-server

OpenAI-compatible HTTP API server for OxiLLaMa — drop-in replacement for `llama-server`.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.5 — **Tests:** 357 passing, 1 skipped — **Status:** Alpha (~99% complete)

## What It Provides

### Inference Endpoints

- **`POST /v1/chat/completions`** — OpenAI chat completions (streaming via SSE + non-streaming)
- **`GET /v1/chat/ws`** — WebSocket streaming transport (alternative to SSE), wired to the same inference worker queue as the SSE path
- **`POST /v1/completions`** — Legacy text completions
- **`POST /v1/embeddings`** — Text embedding extraction
- **`GET /v1/models`** — List available loaded models
- **`GET /health`** — Liveness probe
- **`GET /ready`** — Readiness probe

### Files API

- **`POST /v1/files`** / **`GET /v1/files`** — Upload / list files (disk-backed store)
- **`GET /v1/files/:id`** / **`DELETE /v1/files/:id`** — Retrieve / delete a file
- **`GET /v1/files/:id/content`** — Download raw file content

### Assistants API

- **`POST /v1/threads`** / **`GET /v1/threads/:id`** — Create / retrieve a thread
- **`POST /v1/threads/:id/messages`** / **`GET /v1/threads/:id/messages`** — Add / list messages
- **`POST /v1/threads/:id/runs`** / **`GET /v1/threads/:id/runs/:run_id`** — Create / retrieve a run
- **`POST /v1/threads/:id/runs/:run_id/cancel`** — Cancel a run
- **`GET /v1/threads/:id/runs/:run_id/steps`** / **`GET .../steps/:step_id`** — List / retrieve run steps

### Responses API

- **`POST /v1/responses`** — Create a response (non-streaming or SSE via `"stream": true`); supports `previous_response_id` chaining
- **`GET /v1/responses`** / **`GET /v1/responses/:id`** — List / retrieve responses

### Batch API

- **`POST /v1/batches`** / **`GET /v1/batches`** / **`GET /v1/batches/:id`** / **`POST /v1/batches/:id/cancel`** — Legacy in-memory batch store (does **not** survive a server restart)
- **`POST /v1/batch_jobs`** / **`GET /v1/batch_jobs`** — Disk-spooled batch queue — this is the **persistent** batch backend
- **`GET /v1/batch_jobs/:id`** / **`GET /v1/batch_jobs/:id/output`** / **`POST /v1/batch_jobs/:id/cancel`** — Status / stream output / cancel

### Admin API (loopback-bound, bearer auth)

- **`POST /admin/models/load`** — Load a model into the warm pool
- **`POST /admin/models/unload`** — Unload a model from the warm pool
- **`GET /admin/models`** — List currently loaded models and pool state
- **`GET /admin/stats`** — Runtime statistics and memory usage
- **`GET /admin/health`** — Liveness + model pool readiness; since v0.1.4 also reports a `"backend"` object (GPU device/backend name, resident tensor/byte counts) when the CLI populated one via `AppState::with_backend_info()`, or the CPU shape (`gpu_enabled: false`) otherwise
- **`POST /admin/loras`** / **`GET /admin/loras`** — Register / list LoRA adapters
- **`DELETE /admin/loras/:name`** — Unregister a LoRA adapter

### Features

- Server-Sent Events (SSE) streaming with `delta` chunked responses
- WebSocket streaming as an alternative low-latency transport
- JSON request/response fully compatible with OpenAI SDK clients
- Tool/function calling — JSON Schema to GBNF grammar conversion, `tool_calls` in streaming and non-streaming responses
- Multi-model LRU warm-pool router (`router/pool.rs`) — supports K simultaneously loaded models with LRU eviction
- Batch disk-spool backend (`batch_spool/`, `/v1/batch_jobs`) — batch jobs persist across server restarts (the older `/v1/batches` store is in-memory only, see Batch API above)
- Optional JWT bearer authentication (`jwt` feature, opt-in — off by default) — Pure-Rust HS256/RS256 verification, scope-based route authorization (`chat:read`, `embed:read`, `admin:write`); `alg: "none"` always rejected
- Prometheus-compatible `/metrics` endpoint (lock-free atomic counters, no external `prometheus` crate), enabled via `ServerConfig::metrics_enabled`

## Security Fixes in v0.1.4 (2026-08-17)

- **`/admin/*` was unauthenticated in both router builders.** The loopback/bearer-token guard existed but was only ever exercised inside `#[cfg(test)]` code. Loopback detection now fails closed when `ConnectInfo` is absent, and model paths passed to `/admin/models/load` are checked against a configured allow-list.
- **Path traversal in all three disk stores** (files, batches, threads). Request IDs went straight into `PathBuf::join`; axum percent-decodes path parameters after routing, so `..%2F` traversed and `%2Ftmp%2Fx` (an absolute component) discarded the base entirely. `FilesStore::delete` then called `remove_dir_all` on the result.
- **`serve` called the wrong app-builder function** (`build_app()` instead of `build_app_with_config()`), so authentication, rate limiting, body-size limits, CORS, `/metrics`, structured tracing and graceful shutdown were all dead in the deployed server. Now wired to `build_app_with_config`, plus request cancellation on client disconnect, load shedding (429 instead of parking on a full queue), and a `/ready` endpoint.
- API-key comparison is now constant-time; the per-key rate-limit map is bounded.

## Usage

Start the server from the CLI:

```bash
# Via the oxillama binary
oxillama serve --model ./llama-3.2-3b.Q4_K_M.gguf --port 8080

# Or with extra options
oxillama serve \
  --model ./model.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --threads 8
```

Query it with curl:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }' | jq .
```

Or use the official OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
resp = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "Explain RoPE embeddings."}],
)
print(resp.choices[0].message.content)
```

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
