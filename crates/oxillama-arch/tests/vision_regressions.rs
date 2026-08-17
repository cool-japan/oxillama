//! Regression tests for the vision architectures (LLaVA, LLaVA-NeXT, Qwen2-VL).
//!
//! Each test names the defect it pins down.  They live in an integration-test
//! target rather than `#[cfg(test)]` modules so they exercise only the public
//! API — and so they keep running while other architectures' unit tests are
//! mid-refactor.

#![cfg(all(feature = "llava", feature = "llava16", feature = "qwen2-vl"))]

use oxillama_arch::common::mrope::{MRopeAxis, MRopeTable};
use oxillama_arch::config::ModelConfig;
use oxillama_arch::error::{ArchError, ArchResult};
use oxillama_arch::llava::{ClipFfnOp, ClipVisionParams, Prompt, Segment, VisualTokens};
use oxillama_arch::llava_next::unpad_extent;
use oxillama_arch::qwen2_vl::{
    apply_vision_rope, load_qwen2vl_from_gguf, MRopePos, Qwen2VlModel, Qwen2VlVisionEncoder,
};
use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_gguf::test_utils::build_minimal_qwen2vl_gguf;
use oxillama_gguf::{GgufModel, MetadataStore, MetadataValue};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Flat KV cache good enough for a 1-layer fixture.
struct FlatKv {
    keys: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
    pos: usize,
}

impl FlatKv {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            vals: Vec::new(),
            pos: 0,
        }
    }
}

impl KvCacheAccess for FlatKv {
    fn store_kv(&mut self, layer: usize, k: &[f32], v: &[f32]) -> ArchResult<()> {
        while self.keys.len() <= layer {
            self.keys.push(Vec::new());
            self.vals.push(Vec::new());
        }
        self.keys[layer].extend_from_slice(k);
        self.vals[layer].extend_from_slice(v);
        Ok(())
    }
    fn get_keys(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(self.keys.get(layer).map_or(&[][..], |v| v.as_slice()))
    }
    fn get_values(&self, layer: usize) -> ArchResult<&[f32]> {
        Ok(self.vals.get(layer).map_or(&[][..], |v| v.as_slice()))
    }
    fn seq_len(&self) -> usize {
        self.pos
    }
    fn advance(&mut self) {
        self.pos += 1;
    }
    fn kv_dim(&self) -> usize {
        32
    }
}

/// Metadata for the synthetic qwen2vl fixture's backbone.
fn qwen2vl_backbone_meta() -> MetadataStore {
    let mut meta = MetadataStore::new();
    for (k, v) in [
        ("qwen2vl.embedding_length", 32u32),
        ("qwen2vl.feed_forward_length", 64),
        ("qwen2vl.block_count", 1),
        ("qwen2vl.attention.head_count", 2),
        ("qwen2vl.attention.head_count_kv", 2),
        ("qwen2vl.context_length", 128),
        ("qwen2vl.vocab_size", 32),
    ] {
        meta.insert(k.to_string(), MetadataValue::Uint32(v));
    }
    meta.insert(
        "general.architecture".to_string(),
        MetadataValue::String("qwen2vl".to_string()),
    );
    meta
}

/// Build a small **non-degenerate** Qwen2-VL backbone.
///
/// The synthetic GGUF fixture stores every tensor as zeros, so a model loaded
/// from it emits constant logits and cannot demonstrate that an input reaches
/// the output.  This builds one with deterministic non-zero weights instead.
fn tiny_qwen2vl() -> Qwen2VlModel {
    use oxillama_arch::common::linear::QuantLinear;
    use oxillama_arch::common::rms_norm::RmsNorm;
    use oxillama_gguf::GgufTensorType;
    use oxillama_quant::QuantTensor;

    const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const FFN: usize = 16;
    const VOCAB: usize = 8;

    // Deterministic, well-conditioned pseudo-random weights.
    fn weights(out: usize, inp: usize, seed: usize) -> QuantLinear {
        let vals: Vec<f32> = (0..out * inp)
            .map(|i| {
                let t = ((i * 37 + seed * 101) % 71) as f32 / 71.0;
                (t - 0.5) * 0.6
            })
            .collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        QuantLinear::new(
            QuantTensor::new(bytes, vec![out, inp], GgufTensorType::F32),
            None,
        )
    }

    let mut config = ModelConfig {
        architecture: "qwen2vl".to_string(),
        hidden_size: HIDDEN,
        intermediate_size: FFN,
        num_layers: 1,
        num_attention_heads: HEADS,
        num_kv_heads: HEADS,
        head_dim: HEAD_DIM,
        vocab_size: VOCAB,
        max_context_length: 32,
        ..ModelConfig::default()
    };
    config.rms_norm_eps = 1e-5;

    let token_embd: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| ((i * 13 % 29) as f32 / 29.0) - 0.5)
        .collect();

    let layers = vec![oxillama_arch::qwen2_vl::Qwen2Layer {
        attn_norm: RmsNorm::new(vec![1.0; HIDDEN], config.rms_norm_eps),
        attn_q: weights(HEADS * HEAD_DIM, HIDDEN, 1),
        attn_k: weights(HEADS * HEAD_DIM, HIDDEN, 2),
        attn_v: weights(HEADS * HEAD_DIM, HIDDEN, 3),
        attn_output: weights(HIDDEN, HEADS * HEAD_DIM, 4),
        ffn_norm: RmsNorm::new(vec![1.0; HIDDEN], config.rms_norm_eps),
        ffn_gate: weights(FFN, HIDDEN, 5),
        ffn_up: weights(FFN, HIDDEN, 6),
        ffn_down: weights(HIDDEN, FFN, 7),
    }];

    Qwen2VlModel::new(
        config,
        None,
        token_embd,
        layers,
        RmsNorm::new(vec![1.0; HIDDEN], 1e-5),
        weights(VOCAB, HIDDEN, 8),
        Some(&[1, 1, 0, 0]),
    )
    .expect("hand-built qwen2vl model")
}

// ---------------------------------------------------------------------------
// V2 — the Qwen2-VL vision tower must never be fabricated from zeros
// ---------------------------------------------------------------------------

/// A checkpoint with no vision metadata has **no** tower, and says so.
///
/// Before: `vis_num_layers = vis_cfg.map_or(0, …)` produced a zero-block tower
/// whose patch projection was `vec![0.0; …]`, and `encode_image` cheerfully
/// returned an all-zeros buffer with no error at all.
#[test]
fn qwen2vl_without_vision_metadata_reports_no_tower_instead_of_zeros() {
    let gguf = GgufModel::from_bytes(build_minimal_qwen2vl_gguf()).expect("parse fixture");
    let config = ModelConfig::from_metadata(&qwen2vl_backbone_meta()).expect("config");

    let model = load_qwen2vl_from_gguf(&gguf, &config).expect("text-only load must still succeed");
    assert!(
        !model.has_vision(),
        "no clip.vision.* metadata ⇒ no vision tower"
    );

    let pixels = vec![0.5f32; 3 * 16 * 16];
    let err = model
        .encode_image(&pixels, 16, 16)
        .expect_err("encode_image must not silently return zeros");
    assert!(
        matches!(err, ArchError::NotSupported { .. }),
        "expected NotSupported naming the missing tower, got {err:?}"
    );
}

/// Declaring a tower and then omitting its tensors is an error, not a zero fill.
///
/// The fixture carries `v.patch_embd.weight` but no `v.blk.*`, which the old
/// loader turned into `vec![1.0; hidden]` norms and `vec![0.0; h*h]` projections.
#[test]
fn qwen2vl_declared_tower_with_missing_block_tensors_errors() {
    let gguf = GgufModel::from_bytes(build_minimal_qwen2vl_gguf()).expect("parse fixture");
    let mut meta = qwen2vl_backbone_meta();
    // Declare a real vision tower.
    for (k, v) in [
        ("clip.vision.block_count", 1u32),
        ("clip.vision.embedding_length", 8),
        ("clip.vision.attention.head_count", 2),
        ("clip.vision.patch_size", 4),
        ("clip.vision.image_size", 16),
    ] {
        meta.insert(k.to_string(), MetadataValue::Uint32(v));
    }
    let config = ModelConfig::from_metadata(&meta).expect("config");
    assert!(
        config.vision_config.is_some(),
        "clip.vision.block_count must populate vision_config"
    );

    let err = match load_qwen2vl_from_gguf(&gguf, &config) {
        Ok(_) => panic!("a declared tower with no v.blk.* tensors must fail to load"),
        Err(e) => e,
    };
    match err {
        ArchError::MissingTensor { name } => {
            assert!(
                name.contains("v.blk.0"),
                "the error must name the missing block tensor, got {name}"
            );
        }
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

/// The text-only path is unaffected: the backbone still produces finite logits.
#[test]
fn qwen2vl_text_only_forward_still_works() {
    let gguf = GgufModel::from_bytes(build_minimal_qwen2vl_gguf()).expect("parse fixture");
    let config = ModelConfig::from_metadata(&qwen2vl_backbone_meta()).expect("config");
    let mut model = load_qwen2vl_from_gguf(&gguf, &config).expect("load");

    let mut kv = FlatKv::new();
    let logits = model.forward(&[1u32, 2, 3], &mut kv).expect("forward");
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// Out-of-vocabulary ids are rejected rather than reading past the table.
#[test]
fn qwen2vl_rejects_out_of_vocabulary_tokens() {
    let gguf = GgufModel::from_bytes(build_minimal_qwen2vl_gguf()).expect("parse fixture");
    let config = ModelConfig::from_metadata(&qwen2vl_backbone_meta()).expect("config");
    let mut model = load_qwen2vl_from_gguf(&gguf, &config).expect("load");
    let mut kv = FlatKv::new();
    assert!(model.forward(&[9_999u32], &mut kv).is_err());
}

// ---------------------------------------------------------------------------
// V1 — image-embedding injection actually reaches the backbone
// ---------------------------------------------------------------------------

/// `forward_embeds` runs the backbone over precomputed embeddings, and the
/// injected content genuinely changes the logits.
///
/// Before this landed there was no method anywhere that accepted hidden states
/// instead of token ids, so the vision stack could not influence a forward pass
/// at all — `encode_image` had zero callers in the workspace.
///
/// The synthetic GGUF fixture is all zeros, which makes any model built from it
/// output constant logits, so this test builds a small non-degenerate backbone
/// by hand.
#[test]
fn qwen2vl_forward_embeds_consumes_injected_embeddings() {
    let mut model = tiny_qwen2vl();
    let hidden = model.config.hidden_size;

    let a: Vec<f32> = (0..3 * hidden).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let b: Vec<f32> = (0..3 * hidden).map(|i| 0.4 - (i as f32) * 0.07).collect();

    let mut kv_a = FlatKv::new();
    let logits_a = model.forward_embeds(&a, 3, &mut kv_a).expect("embeds a");
    let mut kv_b = FlatKv::new();
    let logits_b = model.forward_embeds(&b, 3, &mut kv_b).expect("embeds b");

    assert_eq!(logits_a.len(), model.config.vocab_size);
    assert!(logits_a.iter().all(|v| v.is_finite()));
    assert!(
        logits_a.iter().any(|v| v.abs() > 1e-6),
        "the hand-built backbone must produce non-trivial logits"
    );

    let diff = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "different injected embeddings must produce different logits (max diff {diff})"
    );
}

/// Swapping one visual row changes the logits: the image content reaches the
/// backbone rather than being dropped on the floor.
#[test]
fn qwen2vl_injected_image_content_changes_the_output() {
    let mut model = tiny_qwen2vl();
    let hidden = model.config.hidden_size;

    // 2×2 merged grid = 4 visual tokens, placeholder id 5.
    let image_a: Vec<f32> = (0..4 * hidden).map(|i| (i as f32) * 0.02).collect();
    let mut image_b = image_a.clone();
    image_b[0] += 1.0;

    let tokens = [1u32, 5, 2];
    let (embeds_a, positions) = model
        .splice_image_prompt(&tokens, 5, &image_a, 2, 2, 0)
        .expect("splice a");
    let (embeds_b, _) = model
        .splice_image_prompt(&tokens, 5, &image_b, 2, 2, 0)
        .expect("splice b");

    let mut kv_a = FlatKv::new();
    let logits_a = model
        .forward_embeds_mrope(&embeds_a, &positions, &mut kv_a)
        .expect("forward a");
    let mut kv_b = FlatKv::new();
    let logits_b = model
        .forward_embeds_mrope(&embeds_b, &positions, &mut kv_b)
        .expect("forward b");

    let diff = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-6,
        "perturbing one visual token must change the logits (max diff {diff})"
    );
}

/// Feeding the embedding rows of a token sequence through `forward_embeds`
/// reproduces `forward` on those tokens exactly.
#[test]
fn qwen2vl_forward_embeds_matches_token_forward() {
    let mut model = tiny_qwen2vl();
    let hidden = model.config.hidden_size;
    let tokens = [1u32, 5, 3];

    let mut rows = Vec::with_capacity(tokens.len() * hidden);
    for &t in &tokens {
        let off = t as usize * hidden;
        rows.extend_from_slice(&model.token_embd[off..off + hidden]);
    }

    let mut kv1 = FlatKv::new();
    let from_tokens = model.forward(&tokens, &mut kv1).expect("token forward");
    let mut kv2 = FlatKv::new();
    let from_embeds = model
        .forward_embeds(&rows, tokens.len(), &mut kv2)
        .expect("embed forward");

    for (i, (a, b)) in from_tokens.iter().zip(from_embeds.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit {i}: token path {a} vs embed path {b}"
        );
    }
}

/// The splicer replaces the placeholder with the image rows and gives each
/// visual token its own `(t, h, w)` M-RoPE triple.
#[test]
fn qwen2vl_splice_image_prompt_places_visual_tokens_and_positions() {
    let gguf = GgufModel::from_bytes(build_minimal_qwen2vl_gguf()).expect("parse fixture");
    let config = ModelConfig::from_metadata(&qwen2vl_backbone_meta()).expect("config");
    let model = load_qwen2vl_from_gguf(&gguf, &config).expect("load");
    let hidden = config.hidden_size;

    // A 2×2 merged grid = 4 visual tokens.
    let image: Vec<f32> = vec![0.25f32; 4 * hidden];
    let tokens = [1u32, 7, 2];
    let (embeds, positions) = model
        .splice_image_prompt(&tokens, 7, &image, 2, 2, 0)
        .expect("splice");

    assert_eq!(positions.len(), 6, "2 text tokens + 4 visual tokens");
    assert_eq!(embeds.len(), 6 * hidden);

    // Row 0 is the text token, rows 1..5 the image, row 5 the trailing text.
    for v in &embeds[hidden..5 * hidden] {
        assert_eq!(*v, 0.25, "the image rows must be verbatim");
    }
    assert_eq!(positions[0], MRopePos::text(0));
    assert_eq!(positions[1], MRopePos { t: 1, h: 1, w: 1 });
    assert_eq!(positions[2], MRopePos { t: 1, h: 1, w: 2 });
    assert_eq!(positions[3], MRopePos { t: 1, h: 2, w: 1 });
    assert_eq!(positions[4], MRopePos { t: 1, h: 2, w: 2 });
    // n_past advances by max(h, w) = 2 across the image.
    assert_eq!(positions[5], MRopePos::text(3));
}

/// A spliced multimodal prompt runs end to end through the backbone.
#[test]
fn qwen2vl_multimodal_prompt_runs_end_to_end() {
    let gguf = GgufModel::from_bytes(build_minimal_qwen2vl_gguf()).expect("parse fixture");
    let config = ModelConfig::from_metadata(&qwen2vl_backbone_meta()).expect("config");
    let mut model = load_qwen2vl_from_gguf(&gguf, &config).expect("load");
    let hidden = config.hidden_size;

    let image: Vec<f32> = (0..4 * hidden).map(|i| (i as f32) * 0.003).collect();
    let (embeds, positions) = model
        .splice_image_prompt(&[1u32, 7, 2], 7, &image, 2, 2, 0)
        .expect("splice");

    let mut kv = FlatKv::new();
    let logits = model
        .forward_embeds_mrope(&embeds, &positions, &mut kv)
        .expect("multimodal forward");
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// The LLaVA-side splicer produces the same shape without a loaded backbone.
#[test]
fn llava_prompt_splicer_substitutes_image_embeddings() {
    let hidden = 4usize;
    let img = VisualTokens::new(vec![2.0f32; 3 * hidden], hidden).expect("visual tokens");
    let images = [img];
    let tokens = [10u32, 32_000, 11];
    let prompt = Prompt::from_placeholder(&tokens, 32_000, &images).expect("split");

    assert_eq!(
        prompt.segments(),
        &[
            Segment::Text(&[10]),
            Segment::Image(0),
            Segment::Text(&[11])
        ]
    );
    assert_eq!(prompt.seq_len(), 5, "1 + 3 + 1");

    let embeds = prompt
        .build_embeddings(hidden, |tok, row| {
            row.fill(tok as f32);
            Ok(())
        })
        .expect("build");
    assert_eq!(embeds.len(), 5 * hidden);
    assert_eq!(embeds[0], 10.0);
    for v in &embeds[hidden..4 * hidden] {
        assert_eq!(*v, 2.0);
    }
    assert_eq!(embeds[4 * hidden], 11.0);
}

// ---------------------------------------------------------------------------
// V3 — the vision 2-D RoPE must stay inside its head
// ---------------------------------------------------------------------------

/// Rotating head `h` must leave heads `h ± 1` bit-identical.
///
/// The previous code computed `head_start + half/2 + half + i`, whose maximum
/// is `head_start + head_dim + half/2 - 1` — inside the *next* head — and
/// bounds-checked against the whole hidden vector rather than the head, so
/// every head except the last was corrupted.
#[test]
fn vision_rope_never_writes_into_a_neighbouring_head() {
    let head_dim = 8usize;
    let heads = 4usize;
    let buf: Vec<f32> = (0..head_dim * heads).map(|i| 1.0 + i as f32).collect();
    let before = buf.clone();

    for h in 0..heads {
        let mut scratch = buf.clone();
        apply_vision_rope(
            &mut scratch[h * head_dim..(h + 1) * head_dim],
            2,
            3,
            10000.0,
        )
        .expect("rope");
        for (i, (a, b)) in scratch.iter().zip(before.iter()).enumerate() {
            let inside = i / head_dim == h;
            if !inside {
                assert_eq!(a, b, "rotating head {h} disturbed index {i}");
            }
        }
    }
}

/// The rotation is exact: no index is touched twice.
///
/// The old code rotated `[half/2, half)` under both the row and the column
/// angle, so the composition was not a rotation and the inverse did not restore
/// the input.
#[test]
fn vision_rope_round_trips_under_negated_angles() {
    let head_dim = 16usize;
    let half = head_dim / 2;
    let row_pairs = head_dim / 4;
    let (row, col, base) = (3usize, 5usize, 10000.0f32);

    let x: Vec<f32> = (0..head_dim).map(|i| 0.7 + i as f32 * 0.11).collect();
    let mut y = x.clone();
    apply_vision_rope(&mut y, row, col, base).expect("rope");

    let inv_n_dims = 2.0 / half as f32;
    let mut restored = y.clone();
    for j in 0..half {
        let (pos, fi) = if j < row_pairs {
            (row, j)
        } else {
            (col, j - row_pairs)
        };
        let theta = pos as f32 * base.powf(-(fi as f32) * inv_n_dims);
        let (s, c) = theta.sin_cos();
        restored[j] = y[j] * c + y[j + half] * s;
        restored[j + half] = -y[j] * s + y[j + half] * c;
    }
    for (i, (a, b)) in restored.iter().zip(x.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "dim {i}: {a} vs {b}");
    }
}

/// Row and column own disjoint quarters of the head, per the reference's
/// `mrope_sections = {d_head/4, …}` with `indep_sects` set for vision mode.
#[test]
fn vision_rope_row_and_column_sections_are_disjoint() {
    let head_dim = 16usize;
    let half = head_dim / 2;
    let row_pairs = head_dim / 4;
    let x: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32).collect();

    let mut only_col_changed = x.clone();
    apply_vision_rope(&mut only_col_changed, 4, 1, 10000.0).expect("rope");
    let mut baseline = x.clone();
    apply_vision_rope(&mut baseline, 4, 9, 10000.0).expect("rope");

    for j in 0..half {
        let changed = (only_col_changed[j] - baseline[j]).abs() > 1e-6
            || (only_col_changed[j + half] - baseline[j + half]).abs() > 1e-6;
        assert_eq!(
            changed,
            j >= row_pairs,
            "pair {j}: only pairs >= {row_pairs} may depend on the column position"
        );
    }
}

/// Tokens are emitted in 2×2 blocks, as clip.cpp fills its `positions` buffer.
#[test]
fn qwen2vl_vision_token_order_is_2x2_blocks() {
    // A 4-wide, 2-tall grid → 2 blocks.
    let want = [
        (0usize, 0usize),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ];
    for (i, expected) in want.iter().enumerate() {
        assert_eq!(
            Qwen2VlVisionEncoder::token_grid_pos(i, 2, 4),
            Some(*expected),
            "token {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// V4 — M-RoPE must use the checkpoint's real sections
// ---------------------------------------------------------------------------

/// `[16, 24, 24]` is not an equal three-way split of 64 rotation pairs.
///
/// Under the equal-thirds default (`[21, 21, 22]`) the axis boundaries land at
/// 21 and 42 instead of 16 and 40, so pairs 16..21 and 40..42 are driven by the
/// wrong positional axis.
#[test]
fn mrope_real_sections_move_the_axis_boundaries() {
    let head_dim = 128usize;
    let real = MRopeTable::new_with_sections(head_dim, 8, 10000.0, &[16, 24, 24, 0]).expect("real");
    let equal_thirds = MRopeTable::new(head_dim, 8, 10000.0);

    assert_eq!(real.sections(), [16, 24, 24, 0]);
    assert_eq!(
        equal_thirds.sections(),
        [21, 21, 22, 0],
        "the default is an equal split, which no Qwen2-VL checkpoint uses"
    );

    for j in 0..16 {
        assert_eq!(real.axis_of(j), Some(MRopeAxis::Time), "pair {j}");
    }
    for j in 16..40 {
        assert_eq!(real.axis_of(j), Some(MRopeAxis::Height), "pair {j}");
    }
    for j in 40..64 {
        assert_eq!(real.axis_of(j), Some(MRopeAxis::Width), "pair {j}");
    }

    // The two layouts genuinely disagree — this is the failing-before evidence.
    assert_ne!(
        real.axis_of(16),
        equal_thirds.axis_of(16),
        "pair 16 is Height under [16,24,24] but Time under equal thirds"
    );
    assert_ne!(
        real.axis_of(41),
        equal_thirds.axis_of(41),
        "pair 41 is Width under [16,24,24] but Height under equal thirds"
    );
}

/// With `t == h == w` the result is exactly full-width 1-D NeoX RoPE, whichever
/// section layout is in force.
#[test]
fn mrope_text_positions_reduce_to_1d_rope() {
    use oxillama_arch::common::rope::RopeTable;

    let head_dim = 128usize;
    let base = 10000.0f32;
    let pos = 5usize;

    let table =
        MRopeTable::new_with_sections(head_dim, 32, base, &[16, 24, 24, 0]).expect("sections");
    let rope = RopeTable::new_standard(head_dim, 32, base);

    let x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.05).collect();
    let mut a = x.clone();
    table
        .try_apply_mrope(&mut a, pos, pos, pos, head_dim)
        .expect("mrope");
    let mut b = x;
    rope.apply(&mut b, pos);

    for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
        assert!((p - q).abs() < 1e-5, "dim {i}: mrope {p} vs 1-D rope {q}");
    }
}

/// Sections that oversubscribe the head are rejected.
#[test]
fn mrope_rejects_impossible_sections() {
    assert!(MRopeTable::new_with_sections(128, 4, 10000.0, &[50, 50, 0, 0]).is_err());
    assert!(MRopeTable::new_with_sections(128, 4, 10000.0, &[]).is_err());
}

// ---------------------------------------------------------------------------
// V5 — the CLIP tower
// ---------------------------------------------------------------------------

/// Geometry is read from metadata, not hard-coded to ViT-L/14.
#[test]
fn clip_geometry_is_metadata_driven() {
    let mut meta = MetadataStore::new();
    for (k, v) in [
        ("clip.vision.embedding_length", 768u32),
        ("clip.vision.attention.head_count", 12),
        ("clip.vision.block_count", 12),
        ("clip.vision.patch_size", 16),
        ("clip.vision.image_size", 224),
    ] {
        meta.insert(k.to_string(), MetadataValue::Uint32(v));
    }
    let p = oxillama_arch::llava::clip_params_from_metadata(&meta);
    assert_eq!(p.hidden_size, 768);
    assert_eq!(p.num_layers, 12);
    assert_eq!(p.num_patches(), 196);
    assert_eq!(p.executed_layers(), 11, "the last block is skipped");
    p.validate().expect("valid");
}

/// The tower defaults to QuickGELU, which is a different function from the
/// tanh GELU approximation the old code used everywhere.
#[test]
fn clip_defaults_to_quick_gelu() {
    let p = ClipVisionParams::default();
    assert_eq!(p.ffn_op, ClipFfnOp::QuickGelu);

    let x = 1.5f32;
    let quick = ClipFfnOp::QuickGelu.apply(x);
    let tanh = ClipFfnOp::Gelu.apply(x);
    assert!(
        (quick - tanh).abs() > 1e-3,
        "QuickGELU {quick} must differ from tanh GELU {tanh}"
    );
    let want = x * (1.0 / (1.0 + (-1.702f32 * x).exp()));
    assert!((quick - want).abs() < 1e-6);
}

/// `clip.use_gelu` / `clip.use_silu` override the default.
#[test]
fn clip_activation_keys_are_honoured() {
    let mut meta = MetadataStore::new();
    meta.insert("clip.use_gelu".to_string(), MetadataValue::Bool(true));
    assert_eq!(
        oxillama_arch::llava::clip_params_from_metadata(&meta).ffn_op,
        ClipFfnOp::Gelu
    );

    let mut meta = MetadataStore::new();
    meta.insert("clip.use_silu".to_string(), MetadataValue::Bool(true));
    assert_eq!(
        oxillama_arch::llava::clip_params_from_metadata(&meta).ffn_op,
        ClipFfnOp::Silu
    );
}

/// Inconsistent geometry is rejected instead of producing garbage.
#[test]
fn clip_rejects_inconsistent_geometry() {
    let p = ClipVisionParams {
        num_heads: 7,
        ..ClipVisionParams::default()
    };
    assert!(p.validate().is_err(), "1024 is not divisible by 7 heads");

    let p = ClipVisionParams {
        image_size: 300,
        ..ClipVisionParams::default()
    };
    assert!(p.validate().is_err(), "300 is not a whole number of 14s");
}

// ---------------------------------------------------------------------------
// V6 — LLaVA-NeXT anyres unpad
// ---------------------------------------------------------------------------

/// A wide image on a square feature map is cropped top and bottom.
#[test]
fn llava_next_unpad_crops_the_padded_axis() {
    assert_eq!(unpad_extent(48, 48, 100, 200), (12, 0, 24, 48));
    assert_eq!(unpad_extent(48, 48, 200, 100), (0, 12, 48, 24));
    assert_eq!(unpad_extent(48, 48, 100, 100), (0, 0, 48, 48));
}

/// Unpadding strictly reduces the visual token count for a non-square image.
#[test]
fn llava_next_unpad_reduces_token_count() {
    let (_, _, h, w) = unpad_extent(48, 48, 100, 200);
    // With one newline token per surviving row.
    let with_unpad = h * (w + 1);
    let without_unpad = 48 * 48;
    assert!(
        with_unpad < without_unpad,
        "unpad must drop padded rows: {with_unpad} vs {without_unpad}"
    );
}

// ---------------------------------------------------------------------------
// Shared: VisualTokens invariants
// ---------------------------------------------------------------------------

#[test]
fn visual_tokens_validate_their_shape() {
    assert!(VisualTokens::new(vec![0.0; 7], 4).is_err());
    assert!(VisualTokens::new(vec![0.0; 8], 0).is_err());
    let vt = VisualTokens::new(vec![0.0; 12], 4).expect("ok");
    assert_eq!(vt.len(), 3);
    assert_eq!(vt.hidden_size(), 4);
}

// ---------------------------------------------------------------------------
// End-to-end: pixels → CLIP → projector → prompt injection
// ---------------------------------------------------------------------------

/// Drive an actual image through the whole pipeline with hand-built weights.
///
/// The GGUF test-fixture builder is `pub(crate)` inside `oxillama-gguf`, so this
/// cannot start from a real mmproj file; it starts one step later, from a
/// populated [`ClipEncoder`].  Everything after that — patch extraction, the
/// class token, QuickGELU, attention, `post_ln`, projection and the prompt
/// splice — is the production code path.
#[test]
fn image_pipeline_runs_end_to_end() {
    use oxillama_arch::llava::{ClipEncoder, ClipEncoderLayer, MmProjector};

    const CLIP_HIDDEN: usize = 8;
    const LLM_HIDDEN: usize = 6;
    const PATCH: usize = 4;
    const IMAGE: usize = 16; // 4×4 = 16 patches

    fn varied(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 11 + seed * 17) % 31) as f32 / 31.0 - 0.5) * 0.5)
            .collect()
    }

    let params = ClipVisionParams {
        hidden_size: CLIP_HIDDEN,
        num_heads: 2,
        num_layers: 2, // executed_layers() == 1
        patch_size: PATCH,
        image_size: IMAGE,
        eps: 1e-5,
        ffn_op: ClipFfnOp::QuickGelu,
        feature_layer: None,
    };
    params.validate().expect("geometry");
    assert_eq!(params.num_patches(), 16);

    let layer = |seed: usize| ClipEncoderLayer {
        ln1_weight: vec![1.0; CLIP_HIDDEN],
        ln1_bias: vec![0.0; CLIP_HIDDEN],
        ln2_weight: vec![1.0; CLIP_HIDDEN],
        ln2_bias: vec![0.0; CLIP_HIDDEN],
        q_weight: varied(CLIP_HIDDEN * CLIP_HIDDEN, seed + 1),
        k_weight: varied(CLIP_HIDDEN * CLIP_HIDDEN, seed + 2),
        v_weight: varied(CLIP_HIDDEN * CLIP_HIDDEN, seed + 3),
        out_weight: varied(CLIP_HIDDEN * CLIP_HIDDEN, seed + 4),
        q_bias: vec![0.0; CLIP_HIDDEN],
        k_bias: vec![0.0; CLIP_HIDDEN],
        v_bias: vec![0.0; CLIP_HIDDEN],
        out_bias: vec![0.0; CLIP_HIDDEN],
        fc1_weight: varied(4 * CLIP_HIDDEN * CLIP_HIDDEN, seed + 5),
        fc1_bias: vec![0.0; 4 * CLIP_HIDDEN],
        fc2_weight: varied(CLIP_HIDDEN * 4 * CLIP_HIDDEN, seed + 6),
        fc2_bias: vec![0.0; CLIP_HIDDEN],
    };

    let encoder = ClipEncoder {
        class_embd: (0..CLIP_HIDDEN).map(|i| 0.2 + i as f32 * 0.07).collect(),
        patch_embd_weight: varied(CLIP_HIDDEN * PATCH * PATCH * 3, 21),
        patch_embd_bias: vec![0.0; CLIP_HIDDEN],
        position_embd: varied(17 * CLIP_HIDDEN, 22), // 16 patches + CLS
        pre_ln_weight: vec![],
        pre_ln_bias: vec![],
        post_ln_weight: vec![1.0; CLIP_HIDDEN],
        post_ln_bias: vec![0.0; CLIP_HIDDEN],
        layers: vec![layer(0), layer(30)],
        params,
    };
    assert!(encoder.has_class_token());

    // A deterministic "image": a diagonal ramp, channels-first.
    let pixels: Vec<f32> = (0..3 * IMAGE * IMAGE)
        .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
        .collect();

    let clip_features = encoder.encode(&pixels).expect("clip encode");
    assert_eq!(
        clip_features.len(),
        16 * CLIP_HIDDEN,
        "CLS row must be dropped"
    );

    let projector = MmProjector {
        fc1_weight: varied(16 * CLIP_HIDDEN, 41),
        fc1_bias: vec![0.0; 16],
        fc2_weight: varied(LLM_HIDDEN * 16, 42),
        fc2_bias: vec![0.0; LLM_HIDDEN],
        clip_hidden_size: CLIP_HIDDEN,
        mm_hidden_size: 16,
        llm_hidden_size: LLM_HIDDEN,
    };
    let projected = projector.project(&clip_features).expect("project");
    assert_eq!(projected.len(), 16 * LLM_HIDDEN);
    assert!(projected.iter().all(|v| v.is_finite()));
    assert!(
        projected.iter().any(|v| v.abs() > 1e-6),
        "the pipeline must not collapse to zeros"
    );

    let visual = VisualTokens::new(projected, LLM_HIDDEN).expect("visual tokens");
    assert_eq!(visual.len(), 16);

    let images = [visual];
    let tokens = [101u32, 32_000, 102];
    let prompt = Prompt::from_placeholder(&tokens, 32_000, &images).expect("splice");
    assert_eq!(prompt.seq_len(), 18, "1 + 16 + 1");

    let embeds = prompt
        .build_embeddings(LLM_HIDDEN, |tok, row| {
            row.fill(tok as f32 * 0.001);
            Ok(())
        })
        .expect("build embeddings");
    assert_eq!(embeds.len(), 18 * LLM_HIDDEN);

    println!(
        "clip features [16 x {CLIP_HIDDEN}] first 6: {:?}",
        &clip_features[..6]
    );
    println!(
        "projected     [16 x {LLM_HIDDEN}] first 6: {:?}",
        &images[0].as_slice()[..6]
    );
    println!("spliced input [18 x {LLM_HIDDEN}] rows 0..3:");
    for r in 0..3 {
        println!(
            "  row {r}: {:?}",
            &embeds[r * LLM_HIDDEN..(r + 1) * LLM_HIDDEN]
        );
    }
    // Row 1 onward must be the image, verbatim.
    assert_eq!(
        &embeds[LLM_HIDDEN..2 * LLM_HIDDEN],
        &images[0].as_slice()[..LLM_HIDDEN]
    );
}
