# oxillama-server

OpenAI-compatible HTTP API server for OxiLLaMa — drop-in replacement for `llama-server`.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## Status

**Version:** 0.1.2 — **Tests:** 165 passing, 1 skipped — **Status:** Alpha (~98% complete)

## What It Provides

### Inference Endpoints

- **`POST /v1/chat/completions`** — OpenAI chat completions (streaming via SSE + non-streaming)
- **`GET /v1/chat/ws`** — WebSocket streaming transport (alternative to SSE)
- **`POST /v1/completions`** — Legacy text completions
- **`POST /v1/embeddings`** — Text embedding extraction
- **`GET /v1/models`** — List available loaded models
- **`GET /health`** — Liveness probe

### Batch API

- **`POST /v1/batches`** — Submit a batch of inference requests
- **`GET /v1/batches`** — List all batches
- **`GET /v1/batches/:id`** — Get status and results for a batch
- **`POST /v1/batches/:id/cancel`** — Cancel a pending or in-progress batch

### Admin API (loopback-bound, bearer auth)

- **`POST /admin/models/load`** — Load a model into the warm pool
- **`POST /admin/models/unload`** — Unload a model from the warm pool
- **`GET /admin/models`** — List currently loaded models and pool state
- **`GET /admin/stats`** — Runtime statistics and memory usage

### Features

- Server-Sent Events (SSE) streaming with `delta` chunked responses
- WebSocket streaming as an alternative low-latency transport
- JSON request/response fully compatible with OpenAI SDK clients
- Tool/function calling — JSON Schema to GBNF grammar conversion, `tool_calls` in streaming and non-streaming responses
- Multi-model LRU warm-pool router (`router/pool.rs`) — supports K simultaneously loaded models with LRU eviction
- Batch disk-spool backend (`batch_spool/`) — batch jobs persist across server restarts

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
