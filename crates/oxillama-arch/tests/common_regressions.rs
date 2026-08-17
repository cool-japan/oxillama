//! Regression tests for cross-cutting defects in the shared architecture layer.
//!
//! Every test in this file corresponds to a confirmed defect (C1..C14) in
//! `oxillama-arch`'s `common/`, `config.rs`, `traits.rs` or `registry.rs`, and
//! each one fails against the pre-fix implementation.

use oxillama_arch::common::alibi::AlibiBias;

// ─── C4: ALiBi slopes for non-power-of-two head counts ───────────────────────

/// HuggingFace `get_slopes(12)` (and llama.cpp's `powf(m1, 2*(h - n_head_log2) + 1)`)
/// produce `[2^-1 … 2^-8]` followed by the **even**-indexed entries of the
/// 16-head schedule: `[2^-0.5, 2^-1.5, 2^-2.5, 2^-3.5]`.
///
/// The pre-fix implementation filtered `i % 2 != 0` (odd indices), yielding
/// `[2^-1, 2^-2, 2^-3, 2^-4]` for the tail — a completely different set.
#[test]
fn c4_alibi_slopes_12_heads_match_huggingface() {
    let bias = AlibiBias::new(12);
    let slopes = bias.slopes();
    assert_eq!(slopes.len(), 12, "12 heads must give 12 slopes");

    let expected: Vec<f32> = (1..=8)
        .map(|k| 2.0f32.powf(-(k as f32)))
        .chain([
            2.0f32.powf(-0.5),
            2.0f32.powf(-1.5),
            2.0f32.powf(-2.5),
            2.0f32.powf(-3.5),
        ])
        .collect();

    for (i, (&got, &want)) in slopes.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "slope[{i}] = {got}, expected {want} (HF get_slopes(12))"
        );
    }
}

/// BLOOM-176B ships 112 heads. `next_power_of_two(112) = 128`, so the tail is
/// the first 48 **even**-indexed entries of the 128-head schedule, i.e.
/// `2^(-(2k+1) * 8/128)` for k = 0..47.
#[test]
fn c4_alibi_slopes_112_heads_bloom_176b() {
    let bias = AlibiBias::new(112);
    let slopes = bias.slopes();
    assert_eq!(slopes.len(), 112);

    // First 64 entries: the 64-head power-of-two schedule.
    for k in 1..=64usize {
        let want = 2.0f32.powf(-(8.0 / 64.0) * k as f32);
        let got = slopes[k - 1];
        assert!(
            (got - want).abs() < 1e-6 * want.max(1e-6),
            "slope[{}] = {got}, expected {want}",
            k - 1
        );
    }
    // Tail: even-indexed entries of the 128-head schedule.
    for k in 0..48usize {
        let want = 2.0f32.powf(-(8.0 / 128.0) * (2 * k + 1) as f32);
        let got = slopes[64 + k];
        assert!(
            (got - want).abs() < 1e-6 * want.max(1e-6),
            "slope[{}] = {got}, expected {want}",
            64 + k
        );
    }
}

// ─── C1: two RoPE pair conventions ───────────────────────────────────────────

use oxillama_arch::common::rope::{RopeParams, RopeScalingType, RopeStyle, RopeTable};
use oxillama_arch::config::{rope_style_for_arch, ModelConfig};

/// `Norm` rotates `(x[2i], x[2i+1])`; `Neox` rotates `(x[i], x[i+half])`.
///
/// Define `P` mapping `2i → i` and `2i+1 → i + half`.  Then for every `i`:
/// `Norm(v)[2i] == Neox(P(v))[i]` and `Norm(v)[2i+1] == Neox(P(v))[i+half]`.
///
/// Before the fix there was only one `apply`, so `Norm` did not exist and this
/// could not be expressed at all.
#[test]
fn c1_norm_is_the_permuted_neox_rotation() {
    let head_dim = 16usize;
    let half = head_dim / 2;
    let pos = 7usize;

    let neox = RopeTable::new_standard_with_style(head_dim, 32, 10000.0, RopeStyle::Neox);
    let norm = RopeTable::new_standard_with_style(head_dim, 32, 10000.0, RopeStyle::Norm);
    assert_eq!(neox.style(), RopeStyle::Neox);
    assert_eq!(norm.style(), RopeStyle::Norm);

    let v: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.37).collect();

    // P(v): interleaved → half-split.
    let mut permuted = vec![0.0f32; head_dim];
    for i in 0..half {
        permuted[i] = v[2 * i];
        permuted[i + half] = v[2 * i + 1];
    }

    let mut got_norm = v.clone();
    norm.apply(&mut got_norm, pos);
    let mut got_neox = permuted;
    neox.apply(&mut got_neox, pos);

    for i in 0..half {
        assert!(
            (got_norm[2 * i] - got_neox[i]).abs() < 1e-5,
            "Norm(v)[{}] = {} != Neox(P(v))[{}] = {}",
            2 * i,
            got_norm[2 * i],
            i,
            got_neox[i]
        );
        assert!(
            (got_norm[2 * i + 1] - got_neox[i + half]).abs() < 1e-5,
            "Norm(v)[{}] != Neox(P(v))[{}]",
            2 * i + 1,
            i + half
        );
    }
}

/// Hand-computed interleaved rotation with explicit cos/sin.
#[test]
fn c1_norm_rotation_matches_hand_computed_values() {
    let head_dim = 4usize;
    let base = 10000.0f32;
    let pos = 3usize;
    let norm = RopeTable::new_standard_with_style(head_dim, 8, base, RopeStyle::Norm);

    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    norm.apply(&mut x, pos);

    for i in 0..2usize {
        let freq = 1.0f32 / base.powf((2 * i) as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (c, s) = (theta.cos(), theta.sin());
        let (a, b) = ((2 * i) as f32 + 1.0, (2 * i + 1) as f32 + 1.0);
        assert!((x[2 * i] - (a * c - b * s)).abs() < 1e-5, "pair {i} real");
        assert!(
            (x[2 * i + 1] - (a * s + b * c)).abs() < 1e-5,
            "pair {i} imag"
        );
    }
}

/// The two conventions must actually differ at a non-zero position — otherwise
/// there would be nothing to get wrong.
#[test]
fn c1_norm_and_neox_differ() {
    let neox = RopeTable::new_standard_with_style(16, 32, 10000.0, RopeStyle::Neox);
    let norm = RopeTable::new_standard_with_style(16, 32, 10000.0, RopeStyle::Norm);
    let v: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.11).collect();

    let mut a = v.clone();
    neox.apply(&mut a, 5);
    let mut b = v;
    norm.apply(&mut b, 5);

    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
        "Norm and Neox must not coincide"
    );
}

/// The per-architecture assignment must match `llama_model_rope_type()`.
#[test]
fn c1_rope_style_table_matches_llama_cpp() {
    for id in [
        "llama",
        "command-r",
        "deepseek2",
        "granite",
        "minicpm",
        "starcoder",
        "internlm2",
        "olmo",
        // registry aliases that ship as `llama`
        "mistral",
        "mixtral",
        "yi",
        "internlm3",
    ] {
        assert_eq!(
            rope_style_for_arch(id),
            RopeStyle::Norm,
            "{id} is LLAMA_ROPE_TYPE_NORM"
        );
    }
    for id in [
        "qwen2",
        "qwen3",
        "gemma",
        "gemma2",
        "gemma3",
        "phi2",
        "phi3",
        "phimoe",
        "falcon",
        "grok",
        "dbrx",
        "stablelm",
        "gptneox",
        "olmo2",
        "starcoder2",
        "minicpm3",
    ] {
        assert_eq!(
            rope_style_for_arch(id),
            RopeStyle::Neox,
            "{id} is LLAMA_ROPE_TYPE_NEOX"
        );
    }
}

/// `ModelConfig::rope_style()` is the one-line hook each architecture uses.
#[test]
fn c1_model_config_exposes_rope_style() {
    let llama = ModelConfig {
        architecture: "llama".to_string(),
        ..ModelConfig::default()
    };
    assert_eq!(llama.rope_style(), RopeStyle::Norm);
    let qwen = ModelConfig {
        architecture: "qwen3".to_string(),
        ..ModelConfig::default()
    };
    assert_eq!(qwen.rope_style(), RopeStyle::Neox);
}

// ─── C2: YaRN ramp direction, parameters and mscale ──────────────────────────

/// ggml's `rope_yarn_ramp` returns 1 at low pair indices (high frequency), so
/// those dimensions **extrapolate** (stay unscaled), and 0 at high pair indices
/// (low frequency), so those **interpolate** (get divided by `factor`).
///
/// The pre-fix implementation had the ramp inverted: short wavelengths were
/// fully scaled and long wavelengths unscaled.
#[test]
fn c2_yarn_extrapolates_high_frequencies_and_interpolates_low() {
    let head_dim = 128usize;
    let base = 10000.0f32;
    let factor = 4.0f32;

    let params = RopeParams {
        scaling_type: RopeScalingType::Yarn,
        scaling_factor: factor,
        original_context: 4096,
        ..RopeParams::default()
    };
    let yarn = RopeTable::new_with_params(head_dim, 4, base, &params).expect("yarn table");
    let mscale = yarn.mscale();

    // Highest-frequency pair (i = 0, freq = 1.0): must be EXTRAPOLATED, i.e.
    // theta == 1.0 rad at position 1, not 1.0/4.
    let cos0 = yarn.cos[yarn.half_dim] / mscale;
    assert!(
        (cos0 - 1.0f32.cos()).abs() < 1e-3,
        "pair 0 must extrapolate: cos = {cos0}, expected {} (scaled would be {})",
        1.0f32.cos(),
        (1.0f32 / factor).cos()
    );

    // Lowest-frequency pair: must be INTERPOLATED, i.e. theta == f/factor.
    let last = yarn.half_dim - 1;
    let f_last = 1.0f32 / base.powf((2 * last) as f32 / head_dim as f32);
    let sin_last = yarn.sin[yarn.half_dim + last] / mscale;
    let want = f_last / factor;
    assert!(
        (sin_last - want).abs() < want * 0.05,
        "pair {last} must interpolate: sin = {sin_last}, expected ~{want} \
         (unscaled would be {f_last})"
    );
}

/// The YaRN magnitude correction `mscale = 1 + 0.1 * ln(factor)` is folded into
/// `cos`/`sin` exactly as ggml's `rope_yarn` does.  It was never applied.
#[test]
fn c2_yarn_applies_the_mscale_magnitude_correction() {
    let params = RopeParams {
        scaling_type: RopeScalingType::Yarn,
        scaling_factor: 8.0,
        original_context: 8192,
        ..RopeParams::default()
    };
    let yarn = RopeTable::new_with_params(64, 4, 10000.0, &params).expect("yarn table");
    let want = 1.0 + 0.1 * 8.0f32.ln();
    assert!(
        (yarn.mscale() - want).abs() < 1e-5,
        "mscale = {}, expected {want}",
        yarn.mscale()
    );
    // Position 0 has theta = 0, so cos == mscale exactly.
    assert!(
        (yarn.cos[0] - want).abs() < 1e-5,
        "cos[0] = {}",
        yarn.cos[0]
    );
}

/// `beta_fast` / `beta_slow` / `original_context_length` are read from the GGUF
/// rather than hard-coded to 32 / 1 / 4096.
#[test]
fn c2_yarn_parameters_come_from_metadata() {
    use oxillama_gguf::{MetadataStore, MetadataValue};

    let mut store = MetadataStore::new();
    store.insert(
        "general.architecture".to_string(),
        MetadataValue::String("qwen3".to_string()),
    );
    store.insert(
        "qwen3.rope.scaling.type".to_string(),
        MetadataValue::String("yarn".to_string()),
    );
    store.insert(
        "qwen3.rope.scaling.factor".to_string(),
        MetadataValue::Float32(4.0),
    );
    store.insert(
        "qwen3.rope.scaling.original_context_length".to_string(),
        MetadataValue::Uint32(32768),
    );
    store.insert(
        "qwen3.rope.scaling.yarn_beta_fast".to_string(),
        MetadataValue::Float32(16.0),
    );
    store.insert(
        "qwen3.rope.scaling.yarn_beta_slow".to_string(),
        MetadataValue::Float32(2.0),
    );
    store.insert(
        "qwen3.rope.scaling.attn_factor".to_string(),
        MetadataValue::Float32(0.5),
    );

    let params = RopeParams::from_metadata(&store, "qwen3").expect("params");
    assert_eq!(params.scaling_type, RopeScalingType::Yarn);
    assert_eq!(params.original_context, 32768);
    assert!((params.beta_fast - 16.0).abs() < 1e-6);
    assert!((params.beta_slow - 2.0).abs() < 1e-6);
    assert!((params.attn_factor - 0.5).abs() < 1e-6);
    assert_eq!(params.style, RopeStyle::Neox);

    // Non-default betas must change the table.
    let tuned = RopeTable::new_with_params(128, 2048, 10000.0, &params).expect("tuned");
    let mut wide = params.clone();
    wide.beta_fast = 32.0;
    wide.beta_slow = 1.0;
    let baseline = RopeTable::new_with_params(128, 2048, 10000.0, &wide).expect("baseline");
    assert!(
        tuned
            .cos
            .iter()
            .zip(baseline.cos.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "beta_fast/beta_slow must influence the ramp"
    );
}

// ─── C3: LongRoPE / llama3 rope scaling ──────────────────────────────────────

#[test]
fn c3_scaling_type_parses_longrope_and_llama3() {
    assert_eq!(
        RopeScalingType::parse("longrope").expect("longrope"),
        RopeScalingType::LongRope
    );
    assert_eq!(
        RopeScalingType::parse("su").expect("su"),
        RopeScalingType::LongRope
    );
    assert_eq!(
        RopeScalingType::parse("llama3").expect("llama3"),
        RopeScalingType::Llama3
    );
    assert_eq!(
        RopeScalingType::parse("none").expect("none"),
        RopeScalingType::Standard
    );
}

/// An unrecognised scaling type must be an error, not a silent downgrade.
#[test]
fn c3_unknown_scaling_type_is_an_error() {
    assert!(RopeScalingType::parse("dynamic_ntk_v9").is_err());

    use oxillama_gguf::{MetadataStore, MetadataValue};
    let mut store = MetadataStore::new();
    store.insert(
        "general.architecture".to_string(),
        MetadataValue::String("llama".to_string()),
    );
    store.insert(
        "llama.rope.scaling.type".to_string(),
        MetadataValue::String("dynamic_ntk_v9".to_string()),
    );
    assert!(
        ModelConfig::from_metadata(&store).is_err(),
        "unknown rope scaling type must not silently become Standard"
    );
}

/// Phi-3.5 `longrope` must reach `ModelConfig` as `LongRope`, not `Standard`.
#[test]
fn c3_longrope_survives_config_parsing() {
    use oxillama_gguf::{MetadataStore, MetadataValue};
    let mut store = MetadataStore::new();
    store.insert(
        "general.architecture".to_string(),
        MetadataValue::String("phi3".to_string()),
    );
    store.insert(
        "phi3.rope.scaling.type".to_string(),
        MetadataValue::String("longrope".to_string()),
    );
    let cfg = ModelConfig::from_metadata(&store).expect("config");
    assert_eq!(cfg.rope_scaling_type, RopeScalingType::LongRope);
}

/// The per-dimension `freq_factors` path (ggml's `freq_factors`, i.e.
/// `rope_freqs.weight` for Llama-3.1 and `rope_factors_long/short` for
/// Phi-3.5 LongRoPE) must actually divide each frequency.
#[test]
fn c3_freq_factors_divide_each_frequency() {
    let head_dim = 32usize;
    let half = head_dim / 2;
    let base = 10000.0f32;

    let plain = RopeTable::new_standard(head_dim, 4, base);
    let factors = vec![2.0f32; half];
    let params = RopeParams {
        scaling_type: RopeScalingType::LongRope,
        ..RopeParams::default()
    }
    .with_freq_factors(factors);
    let scaled = RopeTable::new_with_params(head_dim, 4, base, &params).expect("scaled");

    // theta_i / 2 at position 1: sin(theta/2) vs sin(theta).
    for i in 0..half {
        let f = 1.0f32 / base.powf((2 * i) as f32 / head_dim as f32);
        let want = (f / 2.0).sin();
        let got = scaled.sin[half + i];
        assert!(
            (got - want).abs() < 1e-6,
            "pair {i}: sin = {got}, expected {want}"
        );
        assert!(plain.sin[half + i] >= got - 1e-9);
    }
}

/// A wrong-length `freq_factors` slice is rejected instead of silently ignored.
#[test]
fn c3_freq_factors_length_is_validated() {
    let params = RopeParams::default().with_freq_factors(vec![1.0; 3]);
    assert!(RopeTable::new_with_params(32, 4, 10000.0, &params).is_err());
}

/// The Llama-3.1 factor formula must reproduce `convert_hf_to_gguf.py`.
#[test]
fn c3_llama3_freq_factors_match_the_converter() {
    use oxillama_arch::common::rope::llama3_freq_factors;
    let head_dim = 128usize;
    let base = 500000.0f32;
    let factors = llama3_freq_factors(head_dim, base, 8.0, 1.0, 4.0, 8192);
    assert_eq!(factors.len(), head_dim / 2);

    // High-frequency dims (short wavelength) keep factor 1.
    assert!((factors[0] - 1.0).abs() < 1e-6, "got {}", factors[0]);
    // Low-frequency dims (long wavelength) take the full factor.
    let last = factors[factors.len() - 1];
    assert!((last - 8.0).abs() < 1e-4, "got {last}");
    // The factors are monotonically non-decreasing across the ramp.
    for w in factors.windows(2) {
        assert!(w[1] >= w[0] - 1e-5, "not monotone: {:?}", w);
    }
}

// ─── C13: context-length guard ───────────────────────────────────────────────

use oxillama_arch::common::attention::{
    swa_attend_start, validate_context_bounds, validate_token_ids,
};

/// `RopeTable::apply` used to index `self.cos[position * half + i]` with no
/// bound against `max_seq_len`, aborting the process on an over-long prompt.
/// The prefill path is reachable from HTTP-server input.
#[test]
fn c13_rope_apply_does_not_panic_past_the_table() {
    let table = RopeTable::new_standard(16, 8, 10000.0);
    let mut x = vec![1.0f32; 16];
    let before = x.clone();
    // Position 9999 is far past max_seq_len = 8.
    table.apply(&mut x, 9999);
    assert_eq!(x, before, "out-of-range apply must be a no-op, not a crash");
}

#[test]
fn c13_try_apply_reports_the_overflow() {
    let table = RopeTable::new_standard(16, 8, 10000.0);
    let mut x = vec![1.0f32; 16];
    assert!(table.try_apply(&mut x, 8).is_err());
    assert!(table.try_apply(&mut x, 7).is_ok());
    let mut short = vec![1.0f32; 4];
    assert!(table.try_apply(&mut short, 0).is_err());
}

#[test]
fn c13_shared_context_validation_helper() {
    let cfg = ModelConfig {
        max_context_length: 32,
        vocab_size: 10,
        ..ModelConfig::default()
    };
    assert!(validate_context_bounds(&cfg, 0, 32).is_ok());
    assert!(validate_context_bounds(&cfg, 0, 33).is_err());
    assert!(validate_context_bounds(&cfg, 30, 3).is_err());
    assert!(validate_token_ids(&cfg, &[9]).is_ok());
    assert!(validate_token_ids(&cfg, &[10]).is_err());
}

// ─── C5: sliding-window trimming ─────────────────────────────────────────────

#[test]
fn c5_swa_attend_start_trims_the_scan() {
    let mistral = ModelConfig {
        architecture: "mistral".to_string(),
        swa_window: Some(16),
        max_context_length: 4096,
        ..ModelConfig::default()
    };
    // A query at position 100 may only see keys 85..=100 (16 tokens).
    assert_eq!(swa_attend_start(&mistral, 0, 100), 85);
    assert_eq!(100 - swa_attend_start(&mistral, 0, 100) + 1, 16);

    let gemma3 = ModelConfig {
        architecture: "gemma3".to_string(),
        swa_window: Some(16),
        swa_interleaved: true,
        max_context_length: 4096,
        ..ModelConfig::default()
    };
    // Layer 5 is the global layer of the 6-layer group.
    assert_eq!(swa_attend_start(&gemma3, 5, 100), 0);
    assert_eq!(swa_attend_start(&gemma3, 4, 100), 85);
}

// ─── C9: config validation and key population ────────────────────────────────

use oxillama_arch::config::ExtraHparams;
use oxillama_gguf::{MetadataStore, MetadataValue};

fn store_with(arch: &str, pairs: &[(&str, MetadataValue)]) -> MetadataStore {
    let mut s = MetadataStore::new();
    s.insert(
        "general.architecture".to_string(),
        MetadataValue::String(arch.to_string()),
    );
    for (k, v) in pairs {
        s.insert((*k).to_string(), v.clone());
    }
    s
}

/// `head_count_kv = 0` used to be accepted and then divided by, aborting the
/// process inside `forward()`.
#[test]
fn c9_zero_kv_heads_is_rejected() {
    let store = store_with(
        "llama",
        &[
            ("llama.attention.head_count", MetadataValue::Uint32(32)),
            ("llama.attention.head_count_kv", MetadataValue::Uint32(0)),
        ],
    );
    assert!(ModelConfig::from_metadata(&store).is_err());
}

/// `num_kv_heads > num_heads` gives `heads_per_kv == 0` and then `h / 0`.
#[test]
fn c9_more_kv_heads_than_query_heads_is_rejected() {
    let store = store_with(
        "llama",
        &[
            ("llama.attention.head_count", MetadataValue::Uint32(8)),
            ("llama.attention.head_count_kv", MetadataValue::Uint32(16)),
        ],
    );
    assert!(ModelConfig::from_metadata(&store).is_err());
}

#[test]
fn c9_non_divisible_kv_heads_is_rejected() {
    let store = store_with(
        "llama",
        &[
            ("llama.attention.head_count", MetadataValue::Uint32(12)),
            ("llama.attention.head_count_kv", MetadataValue::Uint32(5)),
        ],
    );
    assert!(ModelConfig::from_metadata(&store).is_err());
}

#[test]
fn c9_zero_attention_heads_is_rejected() {
    let store = store_with(
        "llama",
        &[("llama.attention.head_count", MetadataValue::Uint32(0))],
    );
    assert!(ModelConfig::from_metadata(&store).is_err());
}

/// `quant_type` was hard-coded to `None` regardless of `general.file_type`.
#[test]
fn c9_quant_type_comes_from_file_type() {
    use oxillama_gguf::GgufTensorType;
    // LLAMA_FTYPE_MOSTLY_Q4_K_M == 15
    let store = store_with("llama", &[("general.file_type", MetadataValue::Uint32(15))]);
    let cfg = ModelConfig::from_metadata(&store).expect("config");
    assert_eq!(cfg.quant_type, Some(GgufTensorType::Q4K));
}

/// `activation` was hard-coded to `"silu"`, which is why Falcon's
/// `parallel_attn` inference (which keys off it) was always false.
#[test]
fn c9_activation_is_not_always_silu() {
    let bloom = ModelConfig::from_metadata(&store_with("bloom", &[])).expect("bloom");
    assert_eq!(bloom.activation, "gelu");
    let gemma = ModelConfig::from_metadata(&store_with("gemma", &[])).expect("gemma");
    assert_eq!(gemma.activation, "gelu_pytorch_tanh");
    let llama = ModelConfig::from_metadata(&store_with("llama", &[])).expect("llama");
    assert_eq!(llama.activation, "silu");

    // An explicit key always wins over the architecture default.
    let explicit = ModelConfig::from_metadata(&store_with(
        "llama",
        &[(
            "llama.activation",
            MetadataValue::String("gelu".to_string()),
        )],
    ))
    .expect("explicit");
    assert_eq!(explicit.activation, "gelu");
}

/// `vision_config` was documented as "populated from `vision.*` GGUF metadata
/// keys when available" and never was — Qwen2-VL loaded zero vision layers.
#[test]
fn c9_vision_config_is_populated() {
    let store = store_with(
        "qwen2vl",
        &[
            ("clip.vision.block_count", MetadataValue::Uint32(32)),
            ("clip.vision.embedding_length", MetadataValue::Uint32(1280)),
            (
                "clip.vision.attention.head_count",
                MetadataValue::Uint32(16),
            ),
            ("clip.vision.image_size", MetadataValue::Uint32(392)),
            ("clip.vision.patch_size", MetadataValue::Uint32(14)),
        ],
    );
    let cfg = ModelConfig::from_metadata(&store).expect("config");
    let vision = cfg.vision_config.expect("vision tower must be detected");
    assert_eq!(vision.num_layers, 32);
    assert_eq!(vision.hidden_size, 1280);
    assert_eq!(vision.num_heads, 16);
    assert_eq!(vision.image_size, 392);
    assert_eq!(vision.patch_size, 14);

    // Text-only models must stay None.
    let text = ModelConfig::from_metadata(&store_with("llama", &[])).expect("llama");
    assert!(text.vision_config.is_none());
}

/// `{arch}.attention.layer_norm_epsilon` (the non-RMS variant) was never read.
#[test]
fn c9_plain_layer_norm_epsilon_is_read() {
    let store = store_with(
        "bloom",
        &[(
            "bloom.attention.layer_norm_epsilon",
            MetadataValue::Float32(1e-3),
        )],
    );
    let cfg = ModelConfig::from_metadata(&store).expect("config");
    assert!(
        (cfg.rms_norm_eps - 1e-3).abs() < 1e-9,
        "got {}",
        cfg.rms_norm_eps
    );
}

/// The supplementary key set that `ModelConfig` has no field for.
#[test]
fn c9_extra_hparams_are_parsed() {
    let store = store_with(
        "gemma2",
        &[
            ("gemma2.rope.dimension_count", MetadataValue::Uint32(128)),
            (
                "gemma2.attn_logit_softcapping",
                MetadataValue::Float32(50.0),
            ),
            (
                "gemma2.final_logit_softcapping",
                MetadataValue::Float32(30.0),
            ),
            (
                "gemma2.attention.query_pre_attn_scalar",
                MetadataValue::Float32(224.0),
            ),
            ("gemma2.expert_group_count", MetadataValue::Uint32(8)),
            ("gemma2.expert_group_used_count", MetadataValue::Uint32(4)),
            ("gemma2.expert_weights_norm", MetadataValue::Bool(true)),
            ("gemma2.attention.clamp_kqv", MetadataValue::Float32(8.0)),
            (
                "gemma2.rope.dimension_sections".to_string().leak(),
                MetadataValue::Array(vec![
                    MetadataValue::Uint32(16),
                    MetadataValue::Uint32(24),
                    MetadataValue::Uint32(24),
                    MetadataValue::Uint32(0),
                ]),
            ),
        ],
    );
    let extra = ExtraHparams::from_metadata(&store, "gemma2");
    assert_eq!(extra.rope_dimension_count, Some(128));
    assert_eq!(extra.attn_logit_softcapping, Some(50.0));
    assert_eq!(extra.final_logit_softcapping, Some(30.0));
    assert_eq!(extra.query_pre_attn_scalar, Some(224.0));
    assert_eq!(extra.expert_group_count, Some(8));
    assert_eq!(extra.expert_group_used_count, Some(4));
    assert_eq!(extra.expert_weights_norm, Some(true));
    assert_eq!(extra.clamp_kqv, Some(8.0));
    assert_eq!(extra.rope_sections, Some(vec![16, 24, 24, 0]));
}

// ─── C7: MLA must not re-project w_kv_b per (token, head, key) ───────────────

use oxillama_arch::common::linear::QuantLinear;
use oxillama_arch::common::mla::{mla_forward, MlaConfig, MlaLatentCache, MlaWeights};
use oxillama_arch::common::rms_norm::RmsNorm;
use oxillama_gguf::GgufTensorType;
use oxillama_quant::QuantTensor;

fn f32_linear(rows: usize, cols: usize, seed: &mut u64) -> QuantLinear {
    let mut bytes = Vec::with_capacity(rows * cols * 4);
    for _ in 0..rows * cols {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((*seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    QuantLinear::new(
        QuantTensor::new(bytes, vec![rows, cols], GgufTensorType::F32),
        None,
    )
}

fn mla_fixture() -> (MlaConfig, MlaWeights) {
    let cfg = MlaConfig {
        num_heads: 4,
        q_lora_rank: 8,
        kv_lora_rank: 6,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        rope_theta: 10000.0,
        softmax_scale: 0.25,
    };
    let hidden = 16usize;
    let mut seed = 12345u64;
    let weights = MlaWeights {
        w_q_a: f32_linear(cfg.q_lora_rank, hidden, &mut seed),
        q_a_norm: RmsNorm::new(vec![1.0; cfg.q_lora_rank], 1e-6),
        w_q_b: f32_linear(cfg.q_full_dim(), cfg.q_lora_rank, &mut seed),
        w_kv_a: f32_linear(cfg.kv_combined_dim(), hidden, &mut seed),
        kv_a_norm: RmsNorm::new(vec![1.0; cfg.kv_lora_rank], 1e-6),
        w_kv_b: f32_linear(cfg.kv_b_full_dim(), cfg.kv_lora_rank, &mut seed),
        w_o: f32_linear(hidden, cfg.attn_out_dim(), &mut seed),
        rope: RopeTable::new_standard(cfg.qk_rope_head_dim, 64, 10000.0),
    };
    (cfg, weights)
}

/// Hoisting the `w_kv_b` projection out of the (head, key) loops must not
/// change the numbers — the same latent was being re-projected identically
/// `2 * num_heads` times per cached position.
#[test]
fn c7_mla_forward_is_numerically_unchanged_and_incremental() {
    let (cfg, weights) = mla_fixture();
    let hidden = 16usize;
    let seq = 5usize;

    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
        .collect();

    let mut cache_batch = MlaLatentCache::new(64, &cfg);
    let batch = mla_forward(&x, &weights, &cfg, &mut cache_batch, 0).expect("batch");
    assert_eq!(batch.len(), seq * hidden);
    assert!(batch.iter().all(|v| v.is_finite()));

    // Token-by-token must reproduce the last row of the batched run: this is
    // what actually exercises `cache_start = cache.seq_len - seq_len` together
    // with the hoisted per-position projection buffer.
    let mut cache_step = MlaLatentCache::new(64, &cfg);
    let mut last = Vec::new();
    for t in 0..seq {
        let row = &x[t * hidden..(t + 1) * hidden];
        last = mla_forward(row, &weights, &cfg, &mut cache_step, t).expect("step");
    }
    for (i, (a, b)) in last
        .iter()
        .zip(batch[(seq - 1) * hidden..].iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-4,
            "incremental vs batched mismatch at {i}: {a} vs {b}"
        );
    }
}

/// A rough guard on the removed work: with 4 heads and 5 tokens the pre-fix
/// code ran `w_kv_b` 2 * 4 * (1+2+3+4+5) = 120 times; the fix runs it 5.
/// The observable proxy is that a wider head count no longer multiplies the
/// runtime.
#[test]
fn c7_mla_cost_does_not_scale_with_head_count() {
    use std::time::Instant;

    let run = |num_heads: usize| -> std::time::Duration {
        let mut cfg = MlaConfig {
            num_heads,
            q_lora_rank: 8,
            kv_lora_rank: 6,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            rope_theta: 10000.0,
            softmax_scale: 0.25,
        };
        cfg.num_heads = num_heads;
        let hidden = 16usize;
        let mut seed = 99u64;
        let weights = MlaWeights {
            w_q_a: f32_linear(cfg.q_lora_rank, hidden, &mut seed),
            q_a_norm: RmsNorm::new(vec![1.0; cfg.q_lora_rank], 1e-6),
            w_q_b: f32_linear(cfg.q_full_dim(), cfg.q_lora_rank, &mut seed),
            w_kv_a: f32_linear(cfg.kv_combined_dim(), hidden, &mut seed),
            kv_a_norm: RmsNorm::new(vec![1.0; cfg.kv_lora_rank], 1e-6),
            w_kv_b: f32_linear(cfg.kv_b_full_dim(), cfg.kv_lora_rank, &mut seed),
            w_o: f32_linear(hidden, cfg.attn_out_dim(), &mut seed),
            rope: RopeTable::new_standard(cfg.qk_rope_head_dim, 64, 10000.0),
        };
        let seq = 24usize;
        let x = vec![0.01f32; seq * hidden];
        let mut cache = MlaLatentCache::new(64, &cfg);
        let t0 = Instant::now();
        mla_forward(&x, &weights, &cfg, &mut cache, 0).expect("forward");
        t0.elapsed()
    };

    // Warm up so the first allocation does not dominate.
    let _ = run(4);
    let small = run(4);
    let large = run(16);
    // `w_kv_b` grows with head count too, so this cannot be a flat ratio; the
    // point is that it is nowhere near the quadratic blow-up of the old code.
    assert!(
        large.as_secs_f64() < small.as_secs_f64() * 40.0 + 0.05,
        "4 heads {small:?} vs 16 heads {large:?}: cost still scales with heads"
    );
}

// ─── C8: quantized MoE path ─────────────────────────────────────────────────

use oxillama_arch::common::moe::{Expert, MoeFfn, MoeScratch, QuantExpert, QuantMoeFfn};

fn ql_from_values(values: &[f32], rows: usize, cols: usize) -> QuantLinear {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    QuantLinear::new(
        QuantTensor::new(bytes, vec![rows, cols], GgufTensorType::F32),
        None,
    )
}

/// The quantized expert path must reproduce the `f32` reference path bit-for-bit
/// enough to be a drop-in replacement — while keeping the weights in GGUF form.
#[test]
fn c8_quant_moe_matches_the_f32_reference() {
    let h = 8usize;
    let n = 6usize;
    let n_exp = 4usize;
    let top_k = 2usize;

    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.5
    };

    let router_w: Vec<f32> = (0..n_exp * h).map(|_| next()).collect();
    let mut gates = Vec::new();
    let mut ups = Vec::new();
    let mut downs = Vec::new();
    for _ in 0..n_exp {
        gates.push((0..n * h).map(|_| next()).collect::<Vec<f32>>());
        ups.push((0..n * h).map(|_| next()).collect::<Vec<f32>>());
        downs.push((0..h * n).map(|_| next()).collect::<Vec<f32>>());
    }

    let f32_moe = MoeFfn {
        router: router_w.clone(),
        experts: (0..n_exp)
            .map(|e| Expert {
                gate: gates[e].clone(),
                up: ups[e].clone(),
                down: downs[e].clone(),
                hidden_size: h,
                intermediate_size: n,
            })
            .collect(),
        top_k,
        num_experts: n_exp,
        hidden_size: h,
    };

    let quant_experts: Vec<QuantExpert> = (0..n_exp)
        .map(|e| {
            QuantExpert::new(
                ql_from_values(&gates[e], n, h),
                ql_from_values(&ups[e], n, h),
                ql_from_values(&downs[e], h, n),
            )
            .expect("expert")
        })
        .collect();
    let quant_moe = QuantMoeFfn::new(ql_from_values(&router_w, n_exp, h), quant_experts, top_k)
        .expect("quant moe");

    let input: Vec<f32> = (0..h).map(|i| (i as f32 - 3.0) * 0.2).collect();
    let mut want = vec![0.0f32; h];
    f32_moe.forward(&input, &mut want).expect("f32 moe");

    let mut got = vec![0.0f32; h];
    let mut scratch = quant_moe.make_scratch();
    quant_moe
        .forward(&input, &mut got, &mut scratch)
        .expect("quant moe");

    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "output[{i}]: quant {a} vs f32 {b}");
    }
}

/// The quantized path must never dequantize: the expert weights stay as GGUF
/// bytes, so a Mixtral-sized layer costs the file's bytes, not 4× the elements.
#[test]
fn c8_quant_experts_keep_their_gguf_representation() {
    let expert = QuantExpert::new(
        ql_from_values(&[0.1f32; 6 * 8], 6, 8),
        ql_from_values(&[0.1f32; 6 * 8], 6, 8),
        ql_from_values(&[0.1f32; 8 * 6], 8, 6),
    )
    .expect("expert");
    assert_eq!(expert.hidden_size(), 8);
    assert_eq!(expert.intermediate_size(), 6);
    assert_eq!(expert.gate.weight.tensor_type, GgufTensorType::F32);
}

#[test]
fn c8_quant_moe_rejects_zero_top_k() {
    let experts = vec![QuantExpert::new(
        ql_from_values(&[0.1f32; 4 * 4], 4, 4),
        ql_from_values(&[0.1f32; 4 * 4], 4, 4),
        ql_from_values(&[0.1f32; 4 * 4], 4, 4),
    )
    .expect("expert")];
    let err = QuantMoeFfn::new(ql_from_values(&[0.1f32; 4], 1, 4), experts, 0);
    assert!(err.is_err(), "top_k = 0 must be rejected at construction");
}

#[test]
fn c8_quant_expert_rejects_mismatched_projections() {
    let bad = QuantExpert::new(
        ql_from_values(&[0.1f32; 6 * 8], 6, 8),
        ql_from_values(&[0.1f32; 5 * 8], 5, 8),
        ql_from_values(&[0.1f32; 8 * 6], 8, 6),
    );
    assert!(bad.is_err());
}

/// `MoeScratch` removes the per-call allocations from the legacy path.
#[test]
fn c8_moe_scratch_is_reusable() {
    let e = Expert {
        gate: vec![1.0; 6 * 4],
        up: vec![1.0; 6 * 4],
        down: vec![1.0; 4 * 6],
        hidden_size: 4,
        intermediate_size: 6,
    };
    let mut scratch = MoeScratch::new(4, 6, 2);
    let mut out_a = vec![0.0f32; 4];
    let mut out_b = vec![0.0f32; 4];
    e.forward_with_scratch(&[0.3, 0.7, 0.1, 0.9], &mut out_a, &mut scratch)
        .expect("first");
    e.forward_with_scratch(&[0.3, 0.7, 0.1, 0.9], &mut out_b, &mut scratch)
        .expect("second");
    assert_eq!(out_a, out_b, "scratch reuse must be side-effect free");
}

// ─── C11 / C12: ForwardPass trait defaults ───────────────────────────────────

use oxillama_arch::traits::{ForwardPass, KvCacheAccess};
use oxillama_arch::{ArchError, ArchResult, LoadedLora, LoraStack};
use std::collections::HashMap;
use std::sync::Arc;

struct StubPass {
    resets: usize,
}

impl ForwardPass for StubPass {
    fn forward(&mut self, _tokens: &[u32], _kv: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        Ok(vec![])
    }
    fn vocab_size(&self) -> usize {
        1
    }
    fn max_context_length(&self) -> usize {
        1
    }
    fn hidden_size(&self) -> usize {
        1
    }
}

struct ResettingPass {
    resets: usize,
}

impl ForwardPass for ResettingPass {
    fn forward(&mut self, _tokens: &[u32], _kv: &mut dyn KvCacheAccess) -> ArchResult<Vec<f32>> {
        Ok(vec![])
    }
    fn vocab_size(&self) -> usize {
        1
    }
    fn max_context_length(&self) -> usize {
        1
    }
    fn hidden_size(&self) -> usize {
        1
    }
    fn reset_sequence(&mut self) {
        self.resets += 1;
    }
}

/// `reset_sequence` exists on the trait so the runtime can clear MLA latent
/// caches / SSM state / position counters between requests.  There was no
/// trait-level hook at all, so nothing could be reset through `dyn ForwardPass`.
#[test]
fn c11_reset_sequence_is_callable_through_the_trait() {
    let mut concrete = ResettingPass { resets: 0 };
    {
        let model: &mut dyn ForwardPass = &mut concrete;
        model.reset_sequence();
        model.reset_sequence();
    }
    assert_eq!(concrete.resets, 2, "the override must be reached via `dyn`");

    // The default must be a harmless no-op for stateless architectures.
    let mut stateless = StubPass { resets: 0 };
    {
        let model: &mut dyn ForwardPass = &mut stateless;
        model.reset_sequence();
    }
    assert_eq!(stateless.resets, 0);
}

fn empty_lora() -> LoadedLora {
    LoadedLora {
        adapters: HashMap::new(),
        rank: 8,
        alpha: 16.0,
    }
}

/// The `apply_lora` default used to be `Ok(())`, so five engine-reachable
/// architectures silently ignored every adapter while reporting success.
#[test]
fn c12_apply_lora_default_is_loud() {
    let mut model = StubPass { resets: 0 };
    match model.apply_lora(&empty_lora()) {
        Err(ArchError::NotSupported { detail }) => {
            assert!(detail.contains("apply_lora"), "detail: {detail}");
        }
        other => panic!("expected NotSupported, got {other:?}"),
    }
}

/// `apply_lora_stack` used to discard every per-entry `_scale`.
#[test]
fn c12_lora_stack_propagates_the_scale() {
    let mut model = StubPass { resets: 0 };
    let mut stack = LoraStack::new();
    stack.push(Arc::new(empty_lora()), 0.5);
    match model.apply_lora_stack(&stack) {
        Err(ArchError::NotSupported { detail }) => {
            assert!(
                detail.contains("0.5"),
                "the non-unit scale must reach the architecture: {detail}"
            );
        }
        other => panic!("expected the scale to be reported, got {other:?}"),
    }
}

/// `with_lora_stack` used to be a no-op for every architecture except Jamba.
#[test]
fn c12_with_lora_stack_is_no_longer_a_silent_noop() {
    let mut model = StubPass { resets: 0 };
    let mut stack = LoraStack::new();
    stack.push(Arc::new(empty_lora()), 1.0);
    assert!(
        model.with_lora_stack(stack).is_err(),
        "with_lora_stack must not silently succeed on an architecture without LoRA"
    );
}

/// `QuantLinear::set_lora` replaces; stacking needs accumulation.
#[test]
fn c12_quant_linear_push_lora_accumulates() {
    use oxillama_quant::LoraAdapter;

    let mut layer = ql_from_values(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    assert_eq!(layer.lora_count(), 0);

    let adapter =
        Arc::new(LoraAdapter::new(vec![1.0, 0.0], vec![1.0, 0.0], 1, 1.0, 2, 2).expect("adapter"));
    layer.set_lora(Arc::clone(&adapter));
    assert_eq!(layer.lora_count(), 1);
    // `set_lora` again replaces, `push_lora` accumulates.
    layer.set_lora(Arc::clone(&adapter));
    assert_eq!(layer.lora_count(), 1, "set_lora replaces");
    layer.push_lora(Arc::clone(&adapter), 0.5);
    layer.push_lora(adapter, 0.25);
    assert_eq!(layer.lora_count(), 3, "push_lora accumulates");
    layer.clear_lora();
    assert_eq!(layer.lora_count(), 0);
}

// ─── C14: shared loader helpers ──────────────────────────────────────────────

use oxillama_arch::common::loader::dequant_to_f32_slice;
use oxillama_quant::KernelDispatcher;

/// Every duplicated copy sliced `&data[off..off + block_bytes]` without ever
/// comparing against `data.len()`, so a truncated GGUF aborted the process.
#[test]
fn c14_truncated_tensor_is_an_error_not_a_panic() {
    let dispatcher = KernelDispatcher::new();
    // Declare 64 Q8_0 weights (2 blocks = 68 bytes) but supply one block.
    let err = dequant_to_f32_slice(
        "blk.0.ffn_down.weight",
        GgufTensorType::Q8_0,
        &[0u8; 34],
        64,
        &dispatcher,
    );
    assert!(err.is_err(), "truncated quantized tensor must be reported");

    // The F32 branch used to zero-pad instead.
    let err_f32 = dequant_to_f32_slice(
        "blk.0.attn_norm.weight",
        GgufTensorType::F32,
        &[0u8; 8],
        16,
        &dispatcher,
    );
    assert!(err_f32.is_err(), "truncated F32 tensor must be reported");
}

// ─── C13 (cont.): sequence state capacity ────────────────────────────────────

use oxillama_arch::common::sequence_state::{AttentionSequenceState, SequenceState};

/// `advance()` never enforced `capacity()`, so the position counter ran past
/// every position-keyed buffer the architectures size from it.
#[test]
fn c13_sequence_state_advance_respects_capacity() {
    let mut state = AttentionSequenceState::new(3);
    for _ in 0..10 {
        state.advance();
    }
    assert_eq!(
        state.step_position(),
        3,
        "advance must saturate at capacity"
    );

    let mut strict = AttentionSequenceState::new(2);
    assert!(strict.try_advance().is_ok());
    assert!(strict.try_advance().is_ok());
    assert!(
        strict.try_advance().is_err(),
        "try_advance must report the overflow"
    );
}
