//! Regression tests for the Mamba-2 architecture.
//!
//! Each test here pins a bug that shipped in the Mamba-2 block:
//!
//! * **MB1** — the depthwise conv1d had no state.  `mamba2_block` called the
//!   stateless `conv1d_depthwise` with `seq_len = 1`, so taps
//!   `0 ..= d_conv-2` always read left-zero-padding and the convolution
//!   degenerated into `silu(bias + w[d_conv-1] * x)`.
//! * **MB2** — the block was Mamba-**1** shaped: no `n_group`, no per-head
//!   `A`/`dt`, `B`/`C` projected from the post-conv activation instead of
//!   flowing through the conv, and no gated `ssm_norm`.
//! * **SHARED** — out-of-range embedding lookups, a missing context-length
//!   guard, `build_from_gguf` returning `NotSupported`, and a no-op
//!   `reset_sequence()`.
//!
//! # Evidence provenance
//!
//! Each test below that quotes a literal "failed with" message observed it
//! against the *unfixed* tree.  Where the quoted text was produced by the
//! pre-rewrite `Mamba2Config` / `Mamba2LayerWeights`, the doc comment says so:
//! the assertion is unchanged, but the model construction had to be ported to
//! the Mamba-2 weight layout, so the literal numbers are not reproducible from
//! today's body.  Tests with no quoted message are new capabilities that could
//! not be expressed at all before the rewrite.

#![cfg(feature = "mamba2")]

use oxillama_arch::common::rms_norm::RmsNorm;
use oxillama_arch::error::ArchResult;
use oxillama_arch::mamba2::model::{
    build_mamba2_model, make_zero_mamba2_layer, Mamba2LayerWeights, Mamba2Model,
};
use oxillama_arch::mamba2::Mamba2Config;
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};

/// Null KV cache — Mamba-2 is recurrent and never touches a KV cache.
struct NullKv;
impl KvCacheAccess for NullKv {
    fn seq_len(&self) -> usize {
        0
    }
    fn store_kv(&mut self, _: usize, _: &[f32], _: &[f32]) -> ArchResult<()> {
        Ok(())
    }
    fn get_keys(&self, _: usize) -> ArchResult<&[f32]> {
        Ok(&[])
    }
    fn get_values(&self, _: usize) -> ArchResult<&[f32]> {
        Ok(&[])
    }
    fn advance(&mut self) {}
}

// d_inner = 4, n_head = 2 -> head_dim = 2; n_group = 1 -> bc_width = d_state = 2.
// conv_dim = 4 + 2*1*2 = 8; d_in_proj = 2*4 + 2*1*2 + 2 = 14.
const D_MODEL: usize = 2;
const D_INNER: usize = 4;
const D_STATE: usize = 2;
const D_CONV: usize = 3;
const N_HEAD: usize = 2;
const N_GROUP: usize = 1;
const VOCAB: usize = 4;
const MAX_SEQ: usize = 16;

fn test_config() -> Mamba2Config {
    Mamba2Config {
        d_model: D_MODEL,
        n_layer: 1,
        d_inner: D_INNER,
        d_state: D_STATE,
        d_conv: D_CONV,
        n_head: N_HEAD,
        n_group: N_GROUP,
        vocab_size: VOCAB,
        max_seq_len: MAX_SEQ,
        rms_norm_eps: 1e-5,
    }
}

/// Build a 1-layer model whose only variable is the conv kernel.
///
/// `taps` is the `[d_conv]` kernel shared by every conv channel.  Every other
/// weight is fixed and non-degenerate so that a change in the conv output
/// propagates all the way to the logits.
fn conv_probe_model(taps: [f32; D_CONV]) -> Mamba2Model {
    let cfg = test_config();
    let conv_dim = cfg.conv_dim();
    let d_in_proj = cfg.d_in_proj();

    // token_embd: token t -> [t+1, t+2] so consecutive tokens differ.
    let mut token_embd = vec![0.0f32; VOCAB * D_MODEL];
    for t in 0..VOCAB {
        token_embd[t * D_MODEL] = t as f32 + 1.0;
        token_embd[t * D_MODEL + 1] = t as f32 + 2.0;
    }

    // Fused zxBCdt projection: a deterministic non-degenerate pattern.
    let mut w_in = vec![0.0f32; d_in_proj * D_MODEL];
    for (i, w) in w_in.iter_mut().enumerate() {
        *w = ((i % 5) as f32 - 2.0) * 0.2;
    }

    let mut w_conv = vec![0.0f32; conv_dim * D_CONV];
    for ch in 0..conv_dim {
        w_conv[ch * D_CONV..(ch + 1) * D_CONV].copy_from_slice(&taps);
    }

    // w_out: mixes both halves of d_inner into d_model.
    let mut w_out = vec![0.0f32; D_MODEL * D_INNER];
    for (i, w) in w_out.iter_mut().enumerate() {
        *w = ((i % 3) as f32 - 1.0) * 0.5;
    }

    let layer = Mamba2LayerWeights {
        norm: RmsNorm::new(vec![1.0f32; D_MODEL], 1e-5),
        w_in,
        w_conv,
        b_conv: vec![0.05f32; conv_dim],
        dt_bias: vec![0.1f32; N_HEAD],
        // A is stored already negative in GGUF; keep it negative here too.
        a: vec![-0.5f32, -1.5],
        d_skip: vec![0.25f32; N_HEAD],
        ssm_norm: vec![1.0f32; D_INNER],
        w_out,
    };

    // lm_head: a distinct row per token so logits expose the hidden state.
    let mut lm_head = vec![0.0f32; VOCAB * D_MODEL];
    for v in 0..VOCAB {
        lm_head[v * D_MODEL] = (v as f32) + 1.0;
        lm_head[v * D_MODEL + 1] = -(v as f32) - 1.0;
    }

    build_mamba2_model(
        cfg,
        token_embd,
        vec![layer],
        RmsNorm::new(vec![1.0f32; D_MODEL], 1e-5),
        lm_head,
    )
    .expect("probe model must build")
}

// ─── MB1: the conv must remember ─────────────────────────────────────────────

/// **MB1** — conv taps `0 ..= d_conv-2` must influence tokens after the first.
///
/// Two models differ *only* in taps 0 and 1; tap 2 (the current-position tap)
/// is zero in both.  With a working conv state the two must diverge as soon as
/// a second token arrives.
///
/// Against the unfixed block this assertion failed with:
///
/// ```text
/// conv taps 0..2 are dead: kernels [5,7,0] and [0,0,0] gave identical logits
/// (max |diff| = 0). The depthwise conv has no state, so position t never sees
/// positions t-1 .. t-(d_conv-1).
///   with_history = [-0.22086304, -0.4417261, -0.6625891, -0.8834522]
///   no_history   = [-0.22086304, -0.4417261, -0.6625891, -0.8834522]
/// ```
///
/// **Evidence provenance:** those numbers were observed against the *unfixed*
/// code using the pre-rewrite `Mamba2LayerWeights` (`w_in_z`/`w_b`/`w_c`/
/// `w_delta`, `expand: 1`).  The assertion is unchanged, but the model
/// construction had to be ported to the Mamba-2 weight layout, so re-running
/// today's body against the old code is not possible and the literal values
/// will not reproduce.
#[test]
fn mb1_conv_history_taps_affect_later_tokens() {
    let mut with_history = conv_probe_model([5.0, 7.0, 0.0]);
    let mut no_history = conv_probe_model([0.0, 0.0, 0.0]);

    let tokens = [1u32, 2, 3];
    let mut kv_a = NullKv;
    let mut kv_b = NullKv;

    let a = with_history
        .forward(&tokens, &mut kv_a)
        .expect("forward (history taps set)");
    let b = no_history
        .forward(&tokens, &mut kv_b)
        .expect("forward (history taps zeroed)");

    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-6,
        "conv taps 0..{} are dead: kernels [5,7,0] and [0,0,0] gave identical \
         logits (max |diff| = {max_diff}). The depthwise conv has no state, so \
         position t never sees positions t-1 .. t-(d_conv-1).\n  with_history = \
         {a:?}\n  no_history   = {b:?}",
        D_CONV - 1
    );
}

/// **MB1** — one `forward()` over N tokens must equal N single-token calls.
///
/// This is the property a recurrent decoder actually needs: the conv ring and
/// the SSM state must carry over between calls exactly as they do inside one.
///
/// **Not a red→green test.** The unfixed block processed one token per step in
/// both modes and was broken identically either way, so this would have
/// *passed* before the fix.  It guards against a future batched path that
/// forgets the ring; `mb1_conv_history_taps_affect_later_tokens` is the test
/// that actually discriminates.
#[test]
fn mb1_incremental_decoding_matches_batch() {
    let tokens = [0u32, 1, 2, 3, 1];

    let mut batch_model = conv_probe_model([0.7, -0.4, 0.9]);
    let mut kv = NullKv;
    let batch = batch_model
        .forward(&tokens, &mut kv)
        .expect("batch forward");

    let mut step_model = conv_probe_model([0.7, -0.4, 0.9]);
    let mut last = Vec::new();
    for &t in &tokens {
        last = step_model.forward(&[t], &mut kv).expect("stepped forward");
    }

    assert_eq!(batch.len(), last.len());
    for (i, (a, b)) in batch.iter().zip(last.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit[{i}]: batch {a} != incremental {b}"
        );
    }
}

// ─── MB2: the block must be Mamba-2, not Mamba-1 ─────────────────────────────

/// **MB2** — `tensor_names()` must describe llama.cpp's MAMBA2 tensor set.
///
/// llama.cpp (`src/llama-arch.cpp`, `LLM_ARCH_MAMBA2` at line 1388) lists
/// exactly: TOKEN_EMBD, OUTPUT_NORM, OUTPUT, ATTN_NORM, SSM_IN, SSM_CONV1D,
/// SSM_DT, SSM_A, SSM_D, SSM_NORM, SSM_OUT — with the lowercase names
/// `blk.%d.ssm_a` / `blk.%d.ssm_d` / `blk.%d.ssm_norm`.  There is **no**
/// SSM_X and no separate B/C projection.
///
/// Against the unfixed code this failed with:
///
/// ```text
/// tensor_names() is missing 'blk.*.attn_norm.weight'; got ["token_embd.weight",
/// "output_norm.weight", "output.weight", "blk.*.ssm_in.weight",
/// "blk.*.ssm_conv1d.weight", "blk.*.ssm_conv1d.bias", "blk.*.ssm_x.weight",
/// "blk.*.ssm_dt.weight", "blk.*.ssm_dt.bias", "blk.*.ssm_A", "blk.*.ssm_D",
/// "blk.*.ssm_out.weight"]
/// ```
#[test]
fn mb2_tensor_names_match_llama_cpp_mamba2_set() {
    use oxillama_arch::registry::ArchitectureRegistry;

    let reg = ArchitectureRegistry::with_builtins();
    let arch = reg.get("mamba2").expect("mamba2 must be registered");
    let names: Vec<String> = arch.tensor_names().into_iter().map(|p| p.pattern).collect();

    for required in [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.*.attn_norm.weight",
        "blk.*.ssm_in.weight",
        "blk.*.ssm_conv1d.weight",
        "blk.*.ssm_conv1d.bias",
        "blk.*.ssm_dt.bias",
        "blk.*.ssm_a",
        "blk.*.ssm_d",
        "blk.*.ssm_norm.weight",
        "blk.*.ssm_out.weight",
    ] {
        assert!(
            names.iter().any(|n| n == required),
            "tensor_names() is missing '{required}'; got {names:?}"
        );
    }

    for forbidden in ["blk.*.ssm_x.weight", "blk.*.ssm_dt.weight"] {
        assert!(
            !names.iter().any(|n| n == forbidden),
            "tensor_names() advertises '{forbidden}', which LLM_ARCH_MAMBA2 does \
             not have; got {names:?}"
        );
    }
}

/// **MB2** — `mamba2.ssm.group_count` must reach the block.
///
/// `n_group` controls the conv width (`d_inner + 2*n_group*d_state`), the
/// `B`/`C` slice width and the gated `ssm_norm` grouping.  Before the rewrite
/// `Mamba2Config` had no such field at all, so this could not be expressed.
#[test]
fn mb2_group_count_drives_the_block_geometry() {
    let bytes = oxillama_gguf::test_utils::build_minimal_mamba2_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture must parse");
    let model = oxillama_arch::mamba2::load_mamba2_from_gguf(&gguf).expect("load");

    assert_eq!(model.config.n_group, 2, "mamba2.ssm.group_count");
    assert_eq!(
        model.config.conv_dim(),
        32 + 2 * 2 * 8,
        "conv width must include 2*n_group*d_state for B and C"
    );
    assert_eq!(
        model.config.d_in_proj(),
        2 * 32 + 2 * 2 * 8 + 4,
        "ssm_in must emit z ‖ x ‖ B ‖ C ‖ dt"
    );
    assert_eq!(
        model.layers[0].w_conv.len(),
        model.config.conv_dim() * model.config.d_conv,
        "ssm_conv1d covers x, B and C"
    );
    assert_eq!(
        model.layers[0].a.len(),
        model.config.n_head,
        "ssm_a is one scalar per head"
    );
    assert_eq!(
        model.layers[0].d_skip.len(),
        model.config.n_head,
        "ssm_d is one scalar per head"
    );
    assert_eq!(
        model.layers[0].dt_bias.len(),
        model.config.n_head,
        "ssm_dt.bias is one scalar per head"
    );
    assert_eq!(
        model.layers[0].ssm_norm.len(),
        model.config.d_inner,
        "ssm_norm is {{d_inner/n_group, n_group}}"
    );
}

/// **MB2** — `ssm_norm` must actually gate the output.
///
/// The tensor was absent from the block, the loader and `tensor_names()`, so
/// scaling it had no effect at all.
#[test]
fn mb2_ssm_norm_scale_changes_the_output() {
    let mut baseline = conv_probe_model([0.3, 0.6, 0.9]);
    let mut scaled = conv_probe_model([0.3, 0.6, 0.9]);
    // Give the second group a different scale.
    for (i, w) in scaled.layers[0].ssm_norm.iter_mut().enumerate() {
        *w = if i < D_INNER / 2 { 1.0 } else { 3.0 };
    }

    let mut kv = NullKv;
    let a = baseline.forward(&[1u32, 2], &mut kv).expect("baseline");
    let b = scaled.forward(&[1u32, 2], &mut kv).expect("scaled");

    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "ssm_norm.weight has no effect on the block output (max |diff| = {max_diff})"
    );
}

// ─── SHARED items ─────────────────────────────────────────────────────────────

/// **SHARED** — the registry must be able to build Mamba-2 from a GGUF file.
///
/// `ModelArchitecture::build_from_gguf` defaults to `NotSupported`, so the
/// engine's registry routing could never reach the Mamba-2 loader.  Against
/// the unfixed code:
///
/// ```text
/// build_from_gguf must route to load_mamba2_from_gguf, got
/// Some(NotSupported { detail: "architecture 'mamba2' has not implemented
/// build_from_gguf()" })
/// ```
///
/// **Evidence provenance:** the quoted failure is verbatim from the unfixed
/// code.  The body has since gained the trailing `forward()` round trip, which
/// the old fixture could not have satisfied.
#[test]
fn shared_build_from_gguf_routes_to_mamba2_loader() {
    use oxillama_arch::config::ModelConfig;
    use oxillama_arch::registry::ArchitectureRegistry;

    let bytes = oxillama_gguf::test_utils::build_minimal_mamba2_gguf();
    let gguf = oxillama_gguf::GgufModel::from_bytes(bytes).expect("fixture must parse");
    let cfg = ModelConfig::from_metadata(&gguf.file.metadata).expect("ModelConfig::from_metadata");

    let reg = ArchitectureRegistry::with_builtins();
    let arch = reg.get("mamba2").expect("mamba2 must be registered");

    let mut built = arch
        .build_from_gguf(&gguf, &cfg)
        .expect("build_from_gguf must route to load_mamba2_from_gguf");

    let mut kv = NullKv;
    let logits = built
        .forward(&[1u32], &mut kv)
        .expect("forward via registry");
    assert_eq!(logits.len(), 256, "logits must be vocab_size long");
}

/// **SHARED** — a token id inside `vocab_size` but outside the real embedding
/// table must be an error, not a panicking slice.
///
/// Against the unfixed code this panicked inside the block:
///
/// ```text
/// thread 'shared_forward_bounds_checks_embedding_lookup' panicked at
/// crates/oxillama-arch/src/mamba2/model.rs:335:55:
/// range start index 6 out of range for slice of length 4
/// ```
///
/// **Evidence provenance:** observed against the unfixed code with a model
/// built directly from a short `token_embd`.  `build_mamba2_model` is fallible
/// now and rejects that at construction (see
/// `shared_load_validates_token_embd_against_vocab_size`), so this body
/// truncates the table afterwards to exercise the runtime bound check.  The
/// panic site (`model.rs`, the raw `token_embd[off..off + d_model]` slice) is
/// the same defect.
#[test]
fn shared_forward_bounds_checks_embedding_lookup() {
    let mut model = conv_probe_model([0.1, 0.2, 0.3]);
    // Truncate the embedding table behind the config's back.
    model.token_embd.truncate(2 * D_MODEL);

    let mut kv = NullKv;
    let result = model.forward(&[3u32], &mut kv);
    assert!(
        result.is_err(),
        "token id 3 has no embedding row (table holds 2 rows); \
         forward() must return an error instead of indexing out of bounds"
    );
}

/// **SHARED** — an over-estimated `vocab_size` is rejected at load time.
#[test]
fn shared_load_validates_token_embd_against_vocab_size() {
    let mut cfg = test_config();
    cfg.vocab_size = VOCAB + 8; // claim more rows than the table has
    let layers = vec![make_zero_mamba2_layer(&cfg)];

    let err = build_mamba2_model(
        cfg,
        vec![0.0f32; VOCAB * D_MODEL],
        layers,
        RmsNorm::new(vec![1.0f32; D_MODEL], 1e-5),
        vec![0.0f32; (VOCAB + 8) * D_MODEL],
    )
    .err()
    .expect("token_embd shorter than vocab_size × d_model must be rejected");
    assert!(
        format!("{err}").contains("token_embd.weight"),
        "unexpected error: {err}"
    );
}

/// **SHARED** — `forward()` must refuse more tokens than the context allows.
///
/// Against the unfixed code:
///
/// ```text
/// 20 tokens exceed max_seq_len=16; forward() must return a context-bound error
/// ```
///
/// **Evidence provenance:** observed against the unfixed code (which had no
/// guard at all and returned `Ok`).  The assertion is unchanged; only the
/// model construction was ported to the Mamba-2 weight layout.
#[test]
fn shared_forward_enforces_context_bounds() {
    let mut model = conv_probe_model([0.1, 0.2, 0.3]);
    let mut kv = NullKv;

    let tokens: Vec<u32> = (0..(MAX_SEQ as u32 + 4))
        .map(|i| i % VOCAB as u32)
        .collect();
    let result = model.forward(&tokens, &mut kv);
    assert!(
        result.is_err(),
        "{} tokens exceed max_seq_len={MAX_SEQ}; forward() must return a \
         context-bound error",
        tokens.len()
    );
}

/// **SHARED** — `embed()` gets the same guards as `forward()`.
#[test]
fn shared_embed_enforces_the_same_guards() {
    let mut model = conv_probe_model([0.1, 0.2, 0.3]);
    let mut kv = NullKv;

    let tokens: Vec<u32> = (0..(MAX_SEQ as u32 + 4))
        .map(|i| i % VOCAB as u32)
        .collect();
    assert!(
        model.embed(&tokens, &mut kv).is_err(),
        "embed() must enforce the context bound"
    );
    assert!(
        model.embed(&[VOCAB as u32], &mut kv).is_err(),
        "embed() must reject an out-of-vocabulary token id"
    );
}

/// **SHARED** — `ForwardPass::reset_sequence()` must clear all recurrent state.
///
/// The trait default is a no-op, so the runtime's per-request reset left the
/// SSM hidden state *and* the conv ring from the previous request in place.
/// Against the unfixed code:
///
/// ```text
/// layer 0: reset_sequence() must zero the SSM hidden state,
/// got [0.009270016, 0.0123756835]
/// ```
///
/// **Evidence provenance:** observed against the unfixed code with the
/// pre-rewrite weight layout, where `h` had `d_state*d_inner = 2` elements.
/// The new config gives `h` 8 elements, so the literal values will not
/// reproduce; the assertion (and the defect it pins) is unchanged.  The conv
/// ring half of this test is new — it could not exist before the ring did.
#[test]
fn shared_reset_sequence_clears_recurrent_state() {
    use oxillama_arch::common::sequence_state::SequenceState;

    let mut model = conv_probe_model([0.5, 0.25, 0.125]);
    let mut kv = NullKv;

    let first = model
        .forward(&[1u32, 2, 3], &mut kv)
        .expect("forward must succeed");

    assert!(
        model
            .state
            .layers
            .iter()
            .any(|l| l.h.iter().any(|&v| v != 0.0)),
        "precondition: forward() must leave a non-zero SSM state"
    );
    assert!(
        model
            .conv_state
            .layers
            .iter()
            .any(|r| r.history().iter().any(|&v| v != 0.0)),
        "precondition: forward() must leave a non-zero conv ring"
    );

    ForwardPass::reset_sequence(&mut model);

    for (idx, layer) in model.state.layers.iter().enumerate() {
        assert!(
            layer.h.iter().all(|&v| v == 0.0),
            "layer {idx}: reset_sequence() must zero the SSM hidden state, got {:?}",
            layer.h
        );
    }
    for (idx, ring) in model.conv_state.layers.iter().enumerate() {
        assert!(
            ring.history().iter().all(|&v| v == 0.0),
            "layer {idx}: reset_sequence() must zero the conv ring, got {:?}",
            ring.history()
        );
    }
    assert_eq!(
        model.state.step_position(),
        0,
        "reset_sequence() must reset the step position"
    );

    // A clean state must reproduce the first run exactly.
    let second = model
        .forward(&[1u32, 2, 3], &mut kv)
        .expect("second forward must succeed");
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] differs after reset_sequence(): {a} vs {b}"
        );
    }
}

/// **SHARED** — a short weight matrix is a typed error, not a truncated dot
/// product.
#[test]
fn shared_short_projection_is_an_error() {
    let mut model = conv_probe_model([0.1, 0.2, 0.3]);
    model.layers[0].w_out.truncate(2);

    let mut kv = NullKv;
    let err = model
        .forward(&[1u32], &mut kv)
        .expect_err("a truncated ssm_out must be reported, not silently zipped short");
    assert!(
        format!("{err}").contains("ssm_out.weight"),
        "unexpected error: {err}"
    );
}
