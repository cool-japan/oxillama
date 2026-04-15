# oxillama-server

OpenAI-compatible HTTP API server for OxiLLaMa — drop-in replacement for `llama-server`.

Part of the [OxiLLaMa](https://github.com/cool-japan/oxillama) workspace — a Pure Rust LLM inference engine.

## What It Provides

- **`POST /v1/chat/completions`** — OpenAI chat completions (streaming via SSE + non-streaming)
- **`POST /v1/completions`** — Legacy text completions
- **`POST /v1/embeddings`** — Text embedding extraction
- **`GET /v1/models`** — List available loaded models
- **`GET /health`** — Liveness probe
- Server-Sent Events (SSE) streaming with `delta` chunked responses
- JSON request/response fully compatible with OpenAI SDK clients

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
