# OxiLLaMa Recipes

Task-oriented code recipes for common OxiLLaMa use-cases.
Each recipe is a self-contained, compilable snippet (not runnable without a real GGUF file).

---

## Recipe 1 — Load a GGUF and generate 100 tokens

Stream 100 tokens from a locally-stored GGUF model to stdout.

```rust,no_run
use oxillama::runtime::{EngineConfig, InferenceEngine, SamplerConfig};

fn main() -> anyhow::Result<()> {
    let config = EngineConfig {
        model_path: "llama-3-8b-instruct.Q4_K_M.gguf".to_string(),
        num_threads: 4,
        ..EngineConfig::default()
    };

    let mut engine = InferenceEngine::new(config);
    engine.load_model()?;

    let sampler = SamplerConfig {
        temperature: 0.8,
        top_k: 40,
        ..SamplerConfig::default()
    };

    let prompt = "The Rust programming language is great because";

    engine.generate_with_config(prompt, 100, sampler, |tok| {
        use std::io::Write as _;
        print!("{tok}");
        let _ = std::io::stdout().flush();
    })?;
    println!();

    Ok(())
}
```

---

## Recipe 2 — Serve an OpenAI-compatible API on port 8080

Spin up an HTTP server that accepts `/v1/completions` and `/v1/chat/completions`
requests from any OpenAI client (requires the `server` feature, enabled by default).

```rust,no_run
#[cfg(feature = "server")]
fn main() -> anyhow::Result<()> {
    use std::sync::Arc;
    use oxillama::runtime::{EngineConfig, InferenceEngine};
    use oxillama::server::{
        build_app, spawn_inference_worker, AppState, ServerConfig,
    };

    // 1. Load the model.
    let engine_config = EngineConfig {
        model_path: "llama-3-8b-instruct.Q4_K_M.gguf".to_string(),
        num_threads: 8,
        ..EngineConfig::default()
    };
    let mut engine = InferenceEngine::new(engine_config);
    engine.load_model()?;

    // 2. Cache read-only metadata before the engine moves into the worker.
    let cached_sampler = engine.config().sampler.clone();
    let hidden_size = engine.hidden_size().unwrap_or(0);
    let vocab_bytes = engine.vocab_bytes().map(Arc::new);

    // 3. Spawn the background inference worker.
    //    Route handlers communicate with it via an mpsc channel.
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let prefix_cache = Arc::new(std::sync::Mutex::new(
        oxillama::runtime::PrefixKvCache::new(oxillama::runtime::PrefixCacheConfig::default()),
    ));
    let loras: Arc<std::sync::RwLock<std::collections::HashMap<
        String,
        Arc<oxillama::arch::lora::LoadedLora>,
    >>> = Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));
    spawn_inference_worker(engine, rx, prefix_cache, loras);

    // 4. Build shared state + axum router.
    let state = Arc::new(AppState::new(
        tx,
        "llama-3-8b-instruct".to_string(),
        cached_sampler,
        vocab_bytes,
        hidden_size,
    ));
    let app = build_app(Arc::clone(&state));

    // 5. Block on the Tokio runtime.
    let server_config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        ..ServerConfig::default()
    };
    let bind_addr = format!("{}:{}", server_config.host, server_config.port);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(oxillama::server::shutdown_signal())
            .await
            .map_err(anyhow::Error::from)
    })
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("Enable the `server` feature: cargo run --features server");
}
```

---

## Recipe 3 — Load a LoRA adapter at runtime

Load a LoRA adapter from a GGUF file, push it onto the engine's LoRA stack,
and generate text with the adapted model. Swap to a second adapter without
reloading the base model.

```rust,no_run
use std::sync::Arc;
use oxillama::runtime::{EngineConfig, InferenceEngine};
use oxillama::arch::lora::LoadedLora;

fn main() -> anyhow::Result<()> {
    // 1. Load the base model.
    let config = EngineConfig {
        model_path: "llama-3-8b.gguf".to_string(),
        ..EngineConfig::default()
    };
    let mut engine = InferenceEngine::new(config);
    engine.load_model()?;

    // 2. Load and push the first LoRA adapter.
    let lora_a = Arc::new(LoadedLora::load("adapter-style-a.gguf")?);
    engine.push_lora(Arc::clone(&lora_a), 1.0); // scale = 1.0
    engine.apply_lora_stack()?;

    engine.generate("Describe Paris in one sentence", 60, |tok| print!("{tok}"))?;
    println!();

    // 3. Hot-swap to a second adapter — no model reload needed.
    let _popped = engine.pop_lora();
    let lora_b = Arc::new(LoadedLora::load("adapter-style-b.gguf")?);
    engine.push_lora(lora_b, 0.8); // lower scale for subtle influence
    engine.apply_lora_stack()?;

    engine.generate("Describe Tokyo in one sentence", 60, |tok| print!("{tok}"))?;
    println!();

    // 4. Return to the bare base model.
    engine.clear_loras();

    Ok(())
}
```

---

## Recipe 4 — Run speculative decoding with a smaller draft model

Use a small, fast draft model to propose candidate tokens that are verified by
the large target model, achieving higher throughput with identical output distribution.

```rust,no_run
use oxillama::runtime::{EngineConfig, SpeculativeConfig, SpeculativeEngine};

fn main() -> anyhow::Result<()> {
    let target_config = EngineConfig {
        model_path: "llama-3-70b.Q4_K_M.gguf".to_string(),
        num_threads: 8,
        ..EngineConfig::default()
    };

    let draft_config = EngineConfig {
        model_path: "llama-3-1b.Q4_K_M.gguf".to_string(),
        num_threads: 2,
        ..EngineConfig::default()
    };

    // The draft model proposes 5 tokens per round; the target verifies them.
    let mut spec_config = SpeculativeConfig::new(target_config, draft_config);
    spec_config.num_speculative = 5;

    let mut engine = SpeculativeEngine::new(spec_config)?;

    engine.generate(
        "Explain the Rust borrow checker in simple terms:",
        256,
        |tok| {
            use std::io::Write as _;
            print!("{tok}");
            let _ = std::io::stdout().flush();
        },
    )?;
    println!();

    Ok(())
}
```

---

## Recipe 5 — Snapshot a session and resume it in a new process

Capture the full KV cache + sampler state after a prefill pass, persist it to
disk, then resume from that exact position in a separate process.

```rust,no_run
use std::path::Path;
use oxillama::runtime::{EngineConfig, InferenceEngine};

// ── Process A: generate some tokens then snapshot ────────────────────────────

fn snapshot_session(model_path: &str, snapshot_path: &str) -> anyhow::Result<()> {
    let config = EngineConfig {
        model_path: model_path.to_string(),
        ..EngineConfig::default()
    };
    let mut engine = InferenceEngine::new(config);
    engine.load_model()?;

    // Prefill the prompt into the KV cache.
    let prompt_tokens = engine.tokenize("The capital of France is")?;
    engine.prefill(&prompt_tokens)?;

    // Capture the session snapshot as a byte blob.
    let snapshot_bytes = engine.snapshot()?;
    std::fs::write(snapshot_path, &snapshot_bytes)?;
    println!("Snapshot written to {snapshot_path} ({} bytes)", snapshot_bytes.len());

    Ok(())
}

// ── Process B: resume from the snapshot and continue generation ───────────────

fn resume_session(model_path: &str, snapshot_path: &str) -> anyhow::Result<()> {
    let snapshot_bytes = std::fs::read(snapshot_path)?;

    // resume() validates the model fingerprint against the on-disk file
    // so it rejects mismatched or corrupted model files.
    let mut engine = InferenceEngine::resume(&snapshot_bytes, Path::new(model_path))?;

    // Continue generating from the restored KV position.
    engine.generate("", 50, |tok| print!("{tok}"))?;
    println!();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let model = "llama-3-8b.gguf";
    let snap = "/tmp/session.oxisnap";
    snapshot_session(model, snap)?;
    resume_session(model, snap)?;
    Ok(())
}
```

---

## Recipe 6 — Build a browser chat app with oxillama-wasm

Run OxiLLaMa entirely in the browser via WebAssembly. Load the model bytes with
`fetch`, initialise the engine from the in-memory buffer, then drive a simple
streaming chat loop from JavaScript.

```js,no_run
// index.js (browser, ES module)
import init, { WasmEngine } from "./oxillama_wasm.js";

async function main() {
  // Initialise the WASM module.
  await init();

  // Fetch the GGUF model into an ArrayBuffer.
  const resp = await fetch("/models/qwen3-0.6b.Q4_K_M.gguf");
  const modelBytes = new Uint8Array(await resp.arrayBuffer());

  // Fetch the tokenizer.json.
  const tokResp = await fetch("/models/tokenizer.json");
  const tokJson = await tokResp.text();

  // Create the engine and load the model from bytes (no filesystem access needed).
  const engine = new WasmEngine();
  await engine.loadModelFromBytes(modelBytes, tokJson);

  const output = document.getElementById("output");
  const prompt = document.getElementById("prompt").value;

  // Stream tokens into the DOM.
  await engine.generate(prompt, 256, (tok) => {
    output.textContent += tok;
  });
}

main().catch(console.error);
```

The Rust side (`oxillama-wasm`) exposes `WasmEngine` via `wasm_bindgen`.
See `crates/oxillama-wasm/src/lib.rs` for the full binding.

---

## Recipe 7 — Resume an interrupted HuggingFace pull

When a large GGUF download is interrupted, use `GgufModel::resume` to validate
the partial file against its checkpoint sidecar (`.oxiresume`) and finish loading
once the download completes.

```rust,no_run
use oxillama::gguf::GgufModel;

fn main() -> anyhow::Result<()> {
    let gguf_path = "/models/llama-3-70b.Q4_K_M.gguf";

    // GgufModel::resume reads the adjacent `.oxiresume` sidecar.
    // Returns Ok(None) if no checkpoint exists (nothing to resume).
    match GgufModel::resume(gguf_path)? {
        None => {
            println!("No resume checkpoint found — starting a fresh download.");
            // ... trigger your download logic here ...
        }
        Some(handle) => {
            println!(
                "Checkpoint validated. Last valid offset: {}",
                handle.checkpoint.last_valid_offset
            );

            if handle.checkpoint.tensors_fully_loaded {
                // The download was already complete; load the model directly.
                // Remove the sidecar before calling finish() because finish()
                // consumes the handle (takes ownership of `self`).
                handle.remove_checkpoint()?;
                let model = handle.finish()?;
                println!(
                    "Model loaded: {} tensors",
                    model.file.header.tensor_count
                );
            } else {
                println!(
                    "Download incomplete (expected {} bytes). \
                     Resume your download from offset {}.",
                    handle.checkpoint.file_size_expected,
                    handle.checkpoint.last_valid_offset
                );
                // ... resume download, then call handle.finish() when done ...
            }
        }
    }

    Ok(())
}
```

---

## Recipe 8 — Load a sharded 70B model

Load a model split across multiple GGUF shards (HuggingFace multi-part format)
as a single unified logical model. Provide the path to any one shard — all
siblings are auto-discovered.

```rust,no_run
use oxillama::gguf::ShardedGgufModel;

fn main() -> anyhow::Result<()> {
    // Shards follow HuggingFace naming: <base>-NNNNN-of-MMMMM.gguf
    // Pass the path to shard 1; the rest are auto-discovered in the same dir.
    let sharded = ShardedGgufModel::load_sharded(
        "/models/llama-3-70b-00001-of-00008.gguf",
    )?;

    println!(
        "Loaded sharded model: {:?}",
        sharded
    );

    // The sharded model exposes the same architecture/tensor API as a single
    // GgufModel. Pass it to your architecture builder as usual.
    // (Use the Runtime's InferenceEngine with a model_path pointing at shard 1
    // when the runtime gains native shard support.)

    Ok(())
}
```
