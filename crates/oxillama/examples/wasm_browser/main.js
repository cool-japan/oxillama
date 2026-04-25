// OxiLLaMa Browser Demo — main.js
//
// Calls the actual wasm-bindgen exports from oxillama-wasm/src/lib.rs:
//
//   init()                                              — #[wasm_bindgen(start)], auto-called
//   parseGgufMetadata(bytes)                            — returns typed metadata object
//   loadModelFromBytesWithProgress(bytes, json, cb)    — returns WasmEngine handle
//   WasmEngine.generate(prompt, maxTokens, onToken)    — streaming generation on the handle
//
// NOTE: the top-level `generate(modelBytes, tokenizerJson, prompt, maxTokens, onToken)`
// also exists, but it reloads the model on every call. We use WasmEngine for efficiency.

import init, {
  parseGgufMetadata,
  loadModelFromBytesWithProgress,
} from './pkg/oxillama_wasm.js';

// ── State ─────────────────────────────────────────────────────────────────────

/** @type {import('./pkg/oxillama_wasm.js').WasmEngine | null} */
let engine = null;
let modelLoaded = false;
let generating = false;
let stopRequested = false;

// Raw file buffers held until the user clicks "Load Model"
let pendingModelBytes = null;   // Uint8Array
let pendingTokenizerJson = null; // string

// ── DOM refs ──────────────────────────────────────────────────────────────────

const modelFileInput    = document.getElementById('model-file');
const tokenizerFileInput = document.getElementById('tokenizer-file');
const modelInfoDiv      = document.getElementById('model-info');
const modelArchSpan     = document.getElementById('model-arch');
const modelCtxSpan      = document.getElementById('model-ctx');
const modelTensorsSpan  = document.getElementById('model-tensors');
const loadBtn           = document.getElementById('load-btn');
const loadProgressBar   = document.getElementById('load-progress');
const loadProgressInner = document.getElementById('load-progress-inner');
const loadStatus        = document.getElementById('load-status');
const promptInput       = document.getElementById('prompt-input');
const maxTokensInput    = document.getElementById('max-tokens');
const generateBtn       = document.getElementById('generate-btn');
const stopBtn           = document.getElementById('stop-btn');
const outputPre         = document.getElementById('output');
const statsP            = document.getElementById('stats');

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Enable the Load button only when both files are selected. */
function updateLoadBtn() {
  loadBtn.disabled = !(pendingModelBytes && pendingTokenizerJson);
}

/** Format a byte count as "X.Y MB" or "X KB". */
function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }
  return `${Math.round(bytes / 1024)} KB`;
}

// ── Initialization ────────────────────────────────────────────────────────────

async function main() {
  // `init()` is marked #[wasm_bindgen(start)] so it runs automatically when the
  // WASM module is instantiated.  We still call the generated ES-module `init()`
  // to await module instantiation before touching any exports.
  await init();

  // ── File pickers ──────────────────────────────────────────────────────────

  modelFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    loadStatus.textContent = `Reading ${file.name} (${fmtSize(file.size)})…`;
    modelInfoDiv.hidden = true;
    pendingModelBytes = null;
    updateLoadBtn();

    const buffer = await file.arrayBuffer();
    pendingModelBytes = new Uint8Array(buffer);

    // Show GGUF metadata immediately (header parse is cheap).
    try {
      const meta = parseGgufMetadata(pendingModelBytes);
      const archLabel = meta.arch ? meta.arch : 'unknown arch';
      const ctxLabel  = meta.context_length ? `ctx ${meta.context_length}` : '';
      const tLabel    = meta.tensor_count ? `${meta.tensor_count} tensors` : '';
      modelArchSpan.textContent   = archLabel;
      modelCtxSpan.textContent    = ctxLabel;
      modelTensorsSpan.textContent = tLabel;
      modelInfoDiv.hidden = false;
    } catch (_) {
      // Non-fatal: metadata preview is best-effort.
    }

    loadStatus.textContent = `GGUF ready (${fmtSize(file.size)}). Now pick tokenizer.json.`;
    updateLoadBtn();
  });

  tokenizerFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    loadStatus.textContent = `Reading tokenizer (${fmtSize(file.size)})…`;
    pendingTokenizerJson = null;
    updateLoadBtn();

    pendingTokenizerJson = await file.text();
    loadStatus.textContent = 'Both files ready. Click "Load Model" to initialize the engine.';
    updateLoadBtn();
  });

  // ── Load button ───────────────────────────────────────────────────────────

  loadBtn.addEventListener('click', handleLoadModel);

  // ── Generate / Stop ───────────────────────────────────────────────────────

  generateBtn.addEventListener('click', handleGenerate);
  stopBtn.addEventListener('click', () => { stopRequested = true; });
}

// ── Model loading ─────────────────────────────────────────────────────────────

async function handleLoadModel() {
  if (!pendingModelBytes || !pendingTokenizerJson) return;

  // Reset any previously loaded engine.
  engine = null;
  modelLoaded = false;
  generateBtn.disabled = true;
  promptInput.disabled = true;
  maxTokensInput.disabled = true;
  loadBtn.disabled = true;

  loadProgressBar.hidden = false;
  loadProgressInner.style.width = '0%';
  loadStatus.textContent = 'Initializing inference engine…';

  try {
    // `loadModelFromBytesWithProgress` signature from lib.rs:
    //   pub fn load_model_from_bytes_with_progress(
    //       model_bytes: &[u8],
    //       tokenizer_json: &str,
    //       on_progress: Option<js_sys::Function>,
    //   ) -> Result<WasmEngine, JsValue>
    //
    // Progress callback receives 0, 25, or 100 as a number.
    const onProgress = (pct) => {
      loadProgressInner.style.width = `${pct}%`;
      loadStatus.textContent = `Loading engine… ${pct}%`;
    };

    engine = await loadModelFromBytesWithProgress(
      pendingModelBytes,
      pendingTokenizerJson,
      onProgress,
    );

    loadProgressInner.style.width = '100%';
    loadStatus.textContent = `Engine ready! Model loaded (${fmtSize(pendingModelBytes.byteLength)}).`;

    modelLoaded = true;
    promptInput.disabled = false;
    maxTokensInput.disabled = false;
    generateBtn.disabled = false;
  } catch (err) {
    loadStatus.textContent = `Load error: ${err}`;
    console.error('Model load failed:', err);
    loadBtn.disabled = false; // Allow retry
  }
}

// ── Text generation ───────────────────────────────────────────────────────────

async function handleGenerate() {
  if (!modelLoaded || !engine || generating) return;

  const prompt = promptInput.value.trim();
  if (!prompt) return;

  generating = true;
  stopRequested = false;
  outputPre.textContent = '';
  statsP.textContent = '';
  generateBtn.disabled = true;
  stopBtn.disabled = false;

  const maxTokens = Math.max(1, parseInt(maxTokensInput.value, 10) || 200);
  let tokenCount = 0;
  const startTime = Date.now();

  try {
    // `WasmEngine.generate` signature from lib.rs:
    //   pub fn generate(
    //       &mut self,
    //       prompt: &str,
    //       max_tokens: usize,
    //       on_token: Option<js_sys::Function>,
    //   ) -> Result<String, JsValue>
    //
    // The on_token callback receives each generated token string as it is decoded.
    const onToken = (token) => {
      if (stopRequested) return;
      outputPre.textContent += token;
      outputPre.scrollTop = outputPre.scrollHeight;
      tokenCount++;
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed > 0) {
        statsP.textContent = `${tokenCount} tokens · ${(tokenCount / elapsed).toFixed(1)} tok/s`;
      }
    };

    // engine.generate is synchronous on the Rust side (WASM has no async within
    // the runtime yet), so we wrap it in a minimal promise to keep the UI
    // responsive between the progress ticks delivered by onToken.
    await new Promise((resolve, reject) => {
      try {
        engine.generate(prompt, maxTokens, onToken);
        resolve();
      } catch (err) {
        reject(err);
      }
    });

    if (stopRequested) {
      outputPre.textContent += '\n\n[Stopped by user]';
    }

    const elapsed = (Date.now() - startTime) / 1000;
    if (elapsed > 0 && tokenCount > 0) {
      statsP.textContent =
        `${tokenCount} tokens in ${elapsed.toFixed(2)}s · ${(tokenCount / elapsed).toFixed(1)} tok/s`;
    }
  } catch (err) {
    outputPre.textContent += `\n\n[Generation error: ${err}]`;
    console.error('Generation failed:', err);
  } finally {
    generating = false;
    generateBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

// ── Entry point ───────────────────────────────────────────────────────────────

main().catch((err) => {
  console.error('Fatal WASM initialization error:', err);
  const body = document.querySelector('main');
  if (body) {
    const errDiv = document.createElement('p');
    errDiv.style.color = '#ef4444';
    errDiv.style.marginTop = '1rem';
    errDiv.textContent = `Failed to initialize OxiLLaMa WASM: ${err}`;
    body.prepend(errDiv);
  }
});
