//! Throughput harness for the LLaMA forward pass on a **real** checkpoint.
//!
//! The CLI cannot run a stock Llama-3 GGUF yet (no GGUF-embedded tokenizer), so
//! `oxillama-cli` gives no evidence at all about the llama decode/prefill path.
//! This harness sidesteps the tokenizer entirely: it feeds raw token ids
//! straight into [`LlamaModel`] and times the same two loops the engine would.
//!
//! It is `#[ignore]`d — it needs a multi-GB checkpoint and takes tens of
//! seconds — and it *skips* rather than fails when the file is absent, so a
//! machine without the model still gets a green `cargo nextest run`.
//!
//! ```text
//! cargo nextest run -p oxillama-arch --release --run-ignored all -E 'test(llama_real)' --no-capture
//! ```
//!
//! Override the checkpoint with `OXILLAMA_LLAMA_GGUF=/path/to/model.gguf`.

use std::time::Instant;

use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::ArchResult;
use oxillama_arch::llama::{load_llama_from_gguf, LlamaModel};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::GgufModel;

/// Flat per-layer KV cache, big enough for the harness's fixed budget.
struct FlatKv {
    kv_dim: usize,
    max_seq: usize,
    position: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl FlatKv {
    fn new(n_layers: usize, kv_dim: usize, max_seq: usize) -> Self {
        Self {
            kv_dim,
            max_seq,
            position: 0,
            keys: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
            values: vec![vec![0.0f32; max_seq * kv_dim]; n_layers],
        }
    }
}

impl KvCacheAccess for FlatKv {
    fn seq_len(&self) -> usize {
        self.position
    }

    fn store_kv(&mut self, layer: usize, key: &[f32], value: &[f32]) -> ArchResult<()> {
        let off = self.position * self.kv_dim;
        let n = key.len().min(self.kv_dim);
        self.keys[layer][off..off + n].copy_from_slice(&key[..n]);
        self.values[layer][off..off + n].copy_from_slice(&value[..n]);
        Ok(())
    }

    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.keys[layer][..(self.position + 1) * self.kv_dim])
    }

    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(&self.values[layer][..(self.position + 1) * self.kv_dim])
    }

    fn advance(&mut self) {
        self.position = (self.position + 1).min(self.max_seq - 1);
    }
}

fn checkpoint_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("OXILLAMA_LLAMA_GGUF") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let p = std::path::PathBuf::from(home)
        .join("jan/models/llama3-8b-instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf");
    p.exists().then_some(p)
}

/// Prompt tokens: arbitrary but in-vocabulary and deterministic.
fn synthetic_prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7919 + 11) % vocab) as u32).collect()
}

/// Print which projections take the fused-Q8 route, and whether the tiled
/// prefill can engage at all.
///
/// Both the fused decode GEMV and the batched prefill are gated on the weight
/// kernel advertising `q8_fused_acts_blocks`.  Without this readout a null
/// measurement is indistinguishable from a measurement of the old path.
fn report_fused_coverage(model: &LlamaModel) {
    use oxillama_arch::llama::FfnVariant;

    let fused = |linear: &oxillama_arch::common::linear::QuantLinear| {
        model
            .dispatcher
            .get_kernel(linear.weight.tensor_type)
            .map(|k| {
                (
                    k.name().to_string(),
                    linear.q8_fused_blocks(&*k).is_some() && linear.lora.is_none(),
                )
            })
            .unwrap_or_else(|_| ("<none>".to_string(), false))
    };

    let mut total = 0usize;
    let mut fused_count = 0usize;
    let mut all_dense = true;
    let mut kernels: Vec<String> = Vec::new();
    for layer in &model.layers {
        let mut projections: Vec<&oxillama_arch::common::linear::QuantLinear> = vec![
            &layer.attn_q,
            &layer.attn_k,
            &layer.attn_v,
            &layer.attn_output,
        ];
        match &layer.ffn {
            FfnVariant::Dense(dense) => projections.extend([&dense.gate, &dense.up, &dense.down]),
            FfnVariant::Moe(_) => all_dense = false,
        }
        for p in projections {
            let (name, ok) = fused(p);
            total += 1;
            fused_count += usize::from(ok);
            if !kernels.contains(&name) {
                kernels.push(name);
            }
        }
    }
    let (head_kernel, head_fused) = fused(&model.output);
    eprintln!(
        "fused-Q8 coverage: {fused_count}/{total} projections, kernels={kernels:?}, \
         lm_head={head_kernel}(fused={head_fused}), all_dense={all_dense}, \
         batched_prefill={}",
        all_dense && fused_count == total
    );
}

#[test]
#[ignore = "needs a multi-GB checkpoint on disk"]
fn llama_real_checkpoint_throughput() {
    let Some(path) = checkpoint_path() else {
        eprintln!("SKIP: no llama checkpoint (set OXILLAMA_LLAMA_GGUF)");
        return;
    };
    eprintln!("checkpoint: {}", path.display());

    let t_load = Instant::now();
    let gguf = GgufModel::load(&path).expect("open the checkpoint");
    let config = ModelConfig::from_metadata(&gguf.file.metadata).expect("parse metadata");
    eprintln!(
        "arch={} layers={} hidden={} heads={} kv_heads={} head_dim={} vocab={} ctx={}",
        config.architecture,
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
        config.max_context_length,
    );
    let mut model = load_llama_from_gguf(&gguf, &config).expect("load llama weights");
    eprintln!("load: {:.3} s", t_load.elapsed().as_secs_f64());
    report_fused_coverage(&model);

    let prompt_len: usize = std::env::var("OXILLAMA_PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let decode_len: usize = std::env::var("OXILLAMA_DECODE_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);

    let kv_dim = config.num_kv_heads * config.head_dim;
    let mut kv = FlatKv::new(config.num_layers, kv_dim, prompt_len + decode_len + 8);

    // ── Prefill ───────────────────────────────────────────────────────────
    let prompt = synthetic_prompt(prompt_len, config.vocab_size);
    let t = Instant::now();
    let logits = model.forward(&prompt, &mut kv).expect("prefill");
    let prefill_s = t.elapsed().as_secs_f64();
    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "prefill logits finite"
    );
    eprintln!(
        "PREFILL {prompt_len} tok in {prefill_s:.3} s = {:.2} tok/s",
        prompt_len as f64 / prefill_s
    );

    // ── Decode ────────────────────────────────────────────────────────────
    let mut next = 1u32;
    let t = Instant::now();
    for _ in 0..decode_len {
        let logits = model.forward(&[next], &mut kv).expect("decode");
        next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(1);
    }
    let decode_s = t.elapsed().as_secs_f64();
    eprintln!(
        "DECODE  {decode_len} tok in {decode_s:.3} s = {:.2} tok/s",
        decode_len as f64 / decode_s
    );
}
