//! Port of llama.cpp's per-tensor quantization-type selection.
//!
//! A quantization *target* such as `Q4_K_M` is not one type — it is a mixture.
//! llama.cpp starts from a default type for the requested `llama_ftype` and
//! then, in `llama_tensor_get_type` (`src/llama-quant.cpp`), promotes specific
//! tensors to higher-precision types: the output head, the value projections
//! of some layers, the FFN down projections of some layers. Reproducing the
//! mixture is what makes an OxiLLaMa-produced `Q4_K_M` file the same shape and
//! quality as a llama.cpp-produced one.
//!
//! ## The selection is stateful, and the order is the file's
//!
//! `llama_tensor_get_type` keeps running counters (`i_attention_wv`,
//! `i_ffn_down`, …) that it increments as it visits tensors, and
//! `use_more_bits` is evaluated against those counters, not against the layer
//! number parsed out of the tensor name (except for MoE models, where the
//! counter would not track the layer). So the visit order is part of the
//! algorithm: llama.cpp iterates the model loader's weight list, which is
//! GGUF tensor-info order. The driver in `super` therefore sorts tensors by
//! their data offset — file order — before calling [`TypeSelector::select`],
//! rather than iterating the name-keyed `HashMap`, whose order is arbitrary.
//!
//! ## Verified against real files
//!
//! `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` and
//! `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` were both produced by llama.cpp, so
//! their per-tensor types *are* the expected output of this module for
//! `ftype = Q4_K_M`. `super::mixture_tests` replays both tensor lists
//! through it and requires an exact match on all 291 + 398 tensors.

use oxillama_gguf::GgufTensorType;

/// The `llama_ftype` values this CLI can produce.
///
/// The discriminants are llama.cpp's `enum llama_ftype` values and are written
/// verbatim into the output's `general.file_type` metadata, so a file produced
/// here reports the same type to llama.cpp as a natively produced one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ftype {
    /// `LLAMA_FTYPE_MOSTLY_Q4_0`
    Q4_0 = 2,
    /// `LLAMA_FTYPE_MOSTLY_Q8_0`
    Q8_0 = 7,
    /// `LLAMA_FTYPE_MOSTLY_Q5_0`
    Q5_0 = 8,
    /// `LLAMA_FTYPE_MOSTLY_Q5_1`
    Q5_1 = 9,
    /// `LLAMA_FTYPE_MOSTLY_Q2_K`
    Q2K = 10,
    /// `LLAMA_FTYPE_MOSTLY_Q3_K_S`
    Q3KS = 11,
    /// `LLAMA_FTYPE_MOSTLY_Q3_K_M`
    Q3KM = 12,
    /// `LLAMA_FTYPE_MOSTLY_Q3_K_L`
    Q3KL = 13,
    /// `LLAMA_FTYPE_MOSTLY_Q4_K_S`
    Q4KS = 14,
    /// `LLAMA_FTYPE_MOSTLY_Q4_K_M`
    Q4KM = 15,
    /// `LLAMA_FTYPE_MOSTLY_Q5_K_S`
    Q5KS = 16,
    /// `LLAMA_FTYPE_MOSTLY_Q5_K_M`
    Q5KM = 17,
    /// `LLAMA_FTYPE_MOSTLY_Q6_K`
    Q6K = 18,
}

impl Ftype {
    /// The `general.file_type` value llama.cpp writes for this mixture.
    pub fn file_type_value(self) -> u32 {
        self as u32
    }

    /// The bulk type: llama.cpp's `default_type` switch in
    /// `llama_model_quantize_impl`.
    pub fn default_type(self) -> GgufTensorType {
        match self {
            Ftype::Q4_0 => GgufTensorType::Q4_0,
            Ftype::Q5_0 => GgufTensorType::Q5_0,
            Ftype::Q5_1 => GgufTensorType::Q5_1,
            Ftype::Q8_0 => GgufTensorType::Q8_0,
            Ftype::Q2K => GgufTensorType::Q2K,
            Ftype::Q3KS | Ftype::Q3KM | Ftype::Q3KL => GgufTensorType::Q3K,
            Ftype::Q4KS | Ftype::Q4KM => GgufTensorType::Q4K,
            Ftype::Q5KS | Ftype::Q5KM => GgufTensorType::Q5K,
            Ftype::Q6K => GgufTensorType::Q6K,
        }
    }

    /// The name used on the command line and in log output.
    pub fn name(self) -> &'static str {
        match self {
            Ftype::Q4_0 => "Q4_0",
            Ftype::Q5_0 => "Q5_0",
            Ftype::Q5_1 => "Q5_1",
            Ftype::Q8_0 => "Q8_0",
            Ftype::Q2K => "Q2_K",
            Ftype::Q3KS => "Q3_K_S",
            Ftype::Q3KM => "Q3_K_M",
            Ftype::Q3KL => "Q3_K_L",
            Ftype::Q4KS => "Q4_K_S",
            Ftype::Q4KM => "Q4_K_M",
            Ftype::Q5KS => "Q5_K_S",
            Ftype::Q5KM => "Q5_K_M",
            Ftype::Q6K => "Q6_K",
        }
    }
}

/// The handful of model hyper-parameters `llama_tensor_get_type` consults.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// `general.architecture`.
    pub arch: String,
    /// `{arch}.block_count`.
    pub n_layer: i32,
    /// `{arch}.attention.head_count`.
    pub n_head: u32,
    /// `{arch}.attention.head_count_kv` (defaults to `n_head`).
    pub n_head_kv: u32,
    /// `{arch}.expert_count` (0 for dense models).
    pub n_expert: u32,
    /// Whether the model has a separate `output.weight`.
    ///
    /// When it does not, `token_embd.weight` is the output head (weights are
    /// tied) and llama.cpp routes it through the *output* branch — which is
    /// why a tied-embedding `Q4_K_M` model stores `token_embd.weight` as Q6_K
    /// while an untied one leaves it at Q4_K.
    pub has_output: bool,
    /// Number of `attn_v` / `attn_qkv` / `attn_kv_b` tensors in the file.
    pub n_attention_wv: i32,
}

impl ModelInfo {
    /// `hparams.n_gqa()`: heads per KV head, 0 when there are no KV heads.
    fn n_gqa(&self) -> u32 {
        self.n_head.checked_div(self.n_head_kv).unwrap_or(0)
    }

    /// `qs.model.type == LLM_TYPE_70B`.
    ///
    /// llama.cpp derives the model "type" label from the architecture and
    /// layer count in `llama-model.cpp`; for LLaMA, 80 layers means 65B when
    /// attention is not grouped and 70B when it is. Only the 70B case changes
    /// a quantization decision (it bumps `attn_v` from Q3_K/Q4_K to Q5_K).
    fn is_70b(&self) -> bool {
        self.arch == "llama" && self.n_layer == 80 && self.n_head != self.n_head_kv
    }
}

/// `llama_tensor_get_type`'s `use_more_bits` helper.
///
/// Selects roughly a third of the layers — the first eighth, the last eighth,
/// and every third layer in between — for a higher-precision type. The
/// arithmetic is done in `i32` because C does it in `int`: `i_layer -
/// n_layers/8` is negative for the first eighth, and `usize` would underflow.
fn use_more_bits(i_layer: i32, n_layers: i32) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

/// A tensor name that `layer_info` could not parse a layer index out of.
#[derive(Debug)]
pub struct LayerParseError {
    /// The tensor whose name could not be parsed.
    pub tensor: String,
    /// Why parsing failed.
    pub reason: String,
}

impl std::fmt::Display for LayerParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tensor '{}': {}", self.tensor, self.reason)
    }
}

impl std::error::Error for LayerParseError {}

/// Stateful per-tensor type chooser.
///
/// Create one per output file and call [`Self::select`] once per tensor **in
/// GGUF file order**; the running counters it maintains are part of the
/// algorithm.
pub struct TypeSelector<'a> {
    info: &'a ModelInfo,
    ftype: Ftype,
    i_attention_wv: i32,
    i_ffn_down: i32,
    i_ffn_gate: i32,
    i_ffn_up: i32,
    /// `qs.n_ffn_down`/`n_ffn_gate`/`n_ffn_up`, all set to `n_layer`.
    n_ffn: i32,
}

impl<'a> TypeSelector<'a> {
    /// Start a fresh selection pass.
    pub fn new(info: &'a ModelInfo, ftype: Ftype) -> Self {
        Self {
            info,
            ftype,
            i_attention_wv: 0,
            i_ffn_down: 0,
            i_ffn_gate: 0,
            i_ffn_up: 0,
            n_ffn: info.n_layer,
        }
    }

    /// `layer_info`: the layer index to feed `use_more_bits`.
    ///
    /// For dense models this is the running counter. For MoE models the expert
    /// tensors are not laid out consecutively, so llama.cpp parses the index
    /// out of the `blk.N.` prefix instead — and throws if it cannot.
    fn layer_info(&self, counter: i32, name: &str) -> Result<i32, LayerParseError> {
        let n_expert = self.info.n_expert.max(1);
        if n_expert <= 1 {
            return Ok(counter);
        }
        let idx = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|digits| digits.parse::<i32>().ok())
            .ok_or_else(|| LayerParseError {
                tensor: name.to_string(),
                reason: "failed to determine layer (expected a 'blk.N.' prefix)".to_string(),
            })?;
        if idx < 0 || idx >= self.info.n_layer {
            return Err(LayerParseError {
                tensor: name.to_string(),
                reason: format!("bad layer {idx}; must be in [0, {})", self.info.n_layer),
            });
        }
        Ok(idx)
    }

    /// Choose the quantization type for `name`, whose first dimension is `ne0`.
    ///
    /// Literal port of `llama_tensor_get_type`, restricted to the branches
    /// reachable from the [`Ftype`]s this CLI exposes. The I-quant and
    /// MXFP4 mixtures are omitted because their encoders do not exist here;
    /// they are unreachable, not silently dropped, because [`Ftype`] cannot
    /// name them.
    pub fn select(&mut self, name: &str, ne0: i64) -> Result<GgufTensorType, LayerParseError> {
        let ftype = self.ftype;
        let mut new_type = ftype.default_type();
        let arch = self.info.arch.as_str();
        let n_expert = self.info.n_expert;
        let is_falcon = arch == "falcon";

        // for arches that share the same tensor between the token embeddings
        // and the output, we quantize the token embeddings with the
        // quantization of the output tensor
        if name == "output.weight" || (!self.info.has_output && name == "token_embd.weight") {
            let qk_k = new_type.block_size() as i64;
            if is_falcon || ne0 % qk_k != 0 {
                new_type = GgufTensorType::Q8_0;
            } else if new_type != GgufTensorType::Q8_0 {
                new_type = GgufTensorType::Q6K;
            }
        } else if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
            // The overrides in this branch are all for I-quant / ternary
            // ftypes, which this CLI cannot produce: `new_type` is unchanged.
        } else if name.contains("attn_v.weight") {
            match ftype {
                Ftype::Q2K => {
                    new_type = if self.info.n_gqa() >= 4 {
                        GgufTensorType::Q4K
                    } else {
                        GgufTensorType::Q3K
                    };
                }
                Ftype::Q3KM => {
                    new_type = if self.i_attention_wv < 2 {
                        GgufTensorType::Q5K
                    } else {
                        GgufTensorType::Q4K
                    };
                }
                Ftype::Q3KL => new_type = GgufTensorType::Q5K,
                Ftype::Q4KM | Ftype::Q5KM
                    if use_more_bits(self.i_attention_wv, self.info.n_attention_wv) =>
                {
                    new_type = GgufTensorType::Q6K;
                }
                Ftype::Q4KS if self.i_attention_wv < 4 => new_type = GgufTensorType::Q5K,
                _ => {}
            }
            if self.info.is_70b() {
                // In the 70B model 8 heads share one attn_v, so this tensor is
                // 8x smaller than attn_q and the extra bits are nearly free.
                if new_type == GgufTensorType::Q3K || new_type == GgufTensorType::Q4K {
                    new_type = GgufTensorType::Q5K;
                }
            }
            if n_expert == 8 {
                new_type = GgufTensorType::Q8_0;
            }
            self.i_attention_wv += 1;
        } else if name.contains("attn_k.weight") {
            if n_expert == 8 {
                new_type = GgufTensorType::Q8_0;
            }
        } else if name.contains("attn_q.weight") {
            // Only the IQ3 mixtures touch attn_q; unreachable here.
        } else if name.contains("ffn_down") {
            let i_layer = self.layer_info(self.i_ffn_down, name)?;
            let n_layer = self.n_ffn;
            match ftype {
                Ftype::Q2K => new_type = GgufTensorType::Q3K,
                Ftype::Q3KM => {
                    new_type = if i_layer < n_layer / 16 {
                        GgufTensorType::Q5K
                    } else if !is_falcon || use_more_bits(i_layer, n_layer) {
                        GgufTensorType::Q4K
                    } else {
                        GgufTensorType::Q3K
                    };
                }
                Ftype::Q3KL => {
                    new_type = if is_falcon {
                        GgufTensorType::Q4K
                    } else {
                        GgufTensorType::Q5K
                    };
                }
                Ftype::Q4KM => {
                    if is_falcon {
                        new_type = if i_layer < n_layer / 16 {
                            GgufTensorType::Q6K
                        } else if use_more_bits(i_layer, n_layer) {
                            GgufTensorType::Q5K
                        } else {
                            GgufTensorType::Q4K
                        };
                    } else if use_more_bits(i_layer, n_layer) {
                        new_type = GgufTensorType::Q6K;
                    }
                }
                Ftype::Q5KM if use_more_bits(i_layer, n_layer) => new_type = GgufTensorType::Q6K,
                Ftype::Q4KS if !is_falcon && i_layer < n_layer / 8 => {
                    new_type = GgufTensorType::Q5K;
                }
                // The Q4_0/Q5_0 -> Q4_1/Q5_1 guard only fires with an
                // importance matrix, which this CLI does not accept.
                _ => {}
            }
            self.i_ffn_down += 1;
        } else if name.contains("attn_output.weight") {
            if !is_falcon {
                if n_expert == 8 {
                    if matches!(
                        ftype,
                        Ftype::Q2K | Ftype::Q3KS | Ftype::Q3KM | Ftype::Q4KS | Ftype::Q4KM
                    ) {
                        new_type = GgufTensorType::Q5K;
                    }
                } else {
                    match ftype {
                        Ftype::Q2K => new_type = GgufTensorType::Q3K,
                        Ftype::Q3KM => new_type = GgufTensorType::Q4K,
                        Ftype::Q3KL => new_type = GgufTensorType::Q5K,
                        _ => {}
                    }
                }
            } else if ftype == Ftype::Q3KL {
                new_type = GgufTensorType::Q4K;
            }
        } else if name.contains("attn_qkv.weight") {
            match ftype {
                Ftype::Q3KM | Ftype::Q3KL => new_type = GgufTensorType::Q4K,
                Ftype::Q4KM => new_type = GgufTensorType::Q5K,
                Ftype::Q5KM => new_type = GgufTensorType::Q6K,
                _ => {}
            }
        } else if name.contains("ffn_gate") {
            // Only IQ3_XS overrides ffn_gate, but the counter still advances.
            let _ = self.layer_info(self.i_ffn_gate, name)?;
            self.i_ffn_gate += 1;
        } else if name.contains("ffn_up") {
            let _ = self.layer_info(self.i_ffn_up, name)?;
            self.i_ffn_up += 1;
        }

        Ok(new_type)
    }
}

/// llama.cpp's `quantize` predicate: is this tensor a candidate at all?
///
/// Literal port of the `quantize &= …` chain in `llama_model_quantize_impl`.
/// Everything it rejects is copied through with its original type and bytes:
/// norm gains and biases (1-D, and separately excluded by name), expert
/// gating logits, the tiny 2-D RWKV/Mamba/Gemma helper tensors, and position
/// embeddings.
pub fn should_quantize(name: &str, n_dims: usize) -> bool {
    // "ends with 'weight'?"
    if !name.ends_with("weight") {
        return false;
    }
    // quantize only 2D and 3D tensors (experts)
    if n_dims < 2 {
        return false;
    }
    // do not quantize norm tensors
    if name.contains("_norm.weight") {
        return false;
    }
    // do not quantize expert gating tensors
    if name.contains("ffn_gate_inp.weight") {
        return false;
    }
    // these are very small (e.g. 4x4)
    if name.contains("altup") || name.contains("laurel") {
        return false;
    }
    // these are not too big so keep them as it is
    if name.contains("per_layer_model_proj") {
        return false;
    }
    // do not quantize positional embeddings and token types (BERT)
    if name == "position_embd.weight" || name == "token_types.weight" {
        return false;
    }
    // do not quantize Mamba / Kimi's small conv1d weights
    if name.contains("ssm_conv1d") || name.contains("shortconv.conv.weight") {
        return false;
    }
    // do not quantize RWKV's small yet 2D weights
    const RWKV_SMALL_2D: [&str; 15] = [
        "time_mix_first.weight",
        "time_mix_w0.weight",
        "time_mix_w1.weight",
        "time_mix_w2.weight",
        "time_mix_v0.weight",
        "time_mix_v1.weight",
        "time_mix_v2.weight",
        "time_mix_a0.weight",
        "time_mix_a1.weight",
        "time_mix_a2.weight",
        "time_mix_g1.weight",
        "time_mix_g2.weight",
        "time_mix_decay_w1.weight",
        "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight",
    ];
    if RWKV_SMALL_2D.iter().any(|needle| name.contains(needle)) {
        return false;
    }
    // do not quantize relative position bias (T5)
    if name.contains("attn_rel_b.weight") {
        return false;
    }
    // do not quantize specific multimodal tensors
    if name.contains(".position_embd.") {
        return false;
    }
    true
}

/// What to do when `ne0` is not divisible by `new_type`'s block size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// Use this type instead.
    Use(GgufTensorType),
    /// llama.cpp falls back to `IQ4_NL` here, which OxiLLaMa can decode but
    /// not encode. The driver turns this into a descriptive error rather than
    /// silently substituting a different type and producing a file that is not
    /// what llama.cpp would have produced.
    NeedsIq4Nl,
    /// llama.cpp's `default:` arm, which throws
    /// `"Unsupported tensor size encountered"`. Reached when a non-K type's
    /// block size does not divide the row — e.g. `--target Q8_0` on a tensor
    /// with 8 columns. The driver reports it as an error, as llama.cpp does.
    Unsupported,
}

/// llama.cpp's `convert_incompatible_tensor` switch, plus its `-> F16` guard.
///
/// Returns `None` when `ne0` is already a whole number of `new_type` blocks.
pub fn fallback_for_incompatible_row(new_type: GgufTensorType, ne0: i64) -> Option<Fallback> {
    if ne0 % new_type.block_size() as i64 == 0 {
        return None;
    }
    let fallback = match new_type {
        GgufTensorType::Q2K | GgufTensorType::Q3K => return Some(Fallback::NeedsIq4Nl),
        GgufTensorType::Q4K => GgufTensorType::Q5_0,
        GgufTensorType::Q5K => GgufTensorType::Q5_1,
        GgufTensorType::Q6K => GgufTensorType::Q8_0,
        // llama.cpp's `default:` arm throws "Unsupported tensor size
        // encountered"; the driver turns this into the same failure.
        _ => return Some(Fallback::Unsupported),
    };
    if ne0 % fallback.block_size() as i64 != 0 {
        return Some(Fallback::Use(GgufTensorType::F16));
    }
    Some(Fallback::Use(fallback))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(arch: &str, n_layer: i32, n_head: u32, n_head_kv: u32, has_output: bool) -> ModelInfo {
        ModelInfo {
            arch: arch.to_string(),
            n_layer,
            n_head,
            n_head_kv,
            n_expert: 0,
            has_output,
            n_attention_wv: n_layer,
        }
    }

    /// The exact layer set `use_more_bits` picks for the two real models this
    /// work is verified against — read off the shipped llama.cpp files.
    #[test]
    fn use_more_bits_matches_shipped_models() {
        let picked_36: Vec<i32> = (0..36).filter(|&i| use_more_bits(i, 36)).collect();
        assert_eq!(
            picked_36,
            vec![0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35],
            "Qwen3-4B (36 layers) Q4_K_M has exactly these attn_v/ffn_down at Q6_K"
        );
        let picked_32: Vec<i32> = (0..32).filter(|&i| use_more_bits(i, 32)).collect();
        assert_eq!(
            picked_32,
            vec![0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31],
            "Llama-3-8B (32 layers) Q4_K_M has exactly these attn_v/ffn_down at Q6_K"
        );
    }

    /// The first eighth makes `i_layer - n_layers/8` negative; doing that in
    /// `usize` would wrap and select every layer.
    #[test]
    fn use_more_bits_handles_negative_offsets() {
        assert!(use_more_bits(0, 32));
        assert!(use_more_bits(3, 32));
        assert!(!use_more_bits(4, 32));
        assert!(!use_more_bits(5, 32));
        assert!(use_more_bits(6, 32));
    }

    #[test]
    fn tied_embeddings_route_token_embd_through_the_output_branch() {
        let info = dense("qwen3", 36, 32, 8, /* has_output */ false);
        let mut sel = TypeSelector::new(&info, Ftype::Q4KM);
        assert_eq!(
            sel.select("token_embd.weight", 2560).expect("select"),
            GgufTensorType::Q6K,
            "with no output.weight, token_embd IS the output head"
        );
    }

    #[test]
    fn untied_embeddings_leave_token_embd_at_the_bulk_type() {
        let info = dense("llama", 32, 32, 8, /* has_output */ true);
        let mut sel = TypeSelector::new(&info, Ftype::Q4KM);
        assert_eq!(
            sel.select("token_embd.weight", 4096).expect("select"),
            GgufTensorType::Q4K
        );
        assert_eq!(
            sel.select("output.weight", 4096).expect("select"),
            GgufTensorType::Q6K
        );
    }

    /// `qk_k` in the output branch is the block size of the *default* type, so
    /// a Q4_0 target with a row of 32 stays on the Q6_K path and only the
    /// later shape check demotes it.
    #[test]
    fn output_head_falls_back_to_q8_0_on_an_indivisible_row() {
        let info = dense("llama", 4, 4, 4, /* has_output */ true);
        let mut sel = TypeSelector::new(&info, Ftype::Q4KM);
        // 100 is not a multiple of 256 (Q4_K's block size).
        assert_eq!(
            sel.select("output.weight", 100).expect("select"),
            GgufTensorType::Q8_0
        );
    }

    #[test]
    fn q6_k_target_keeps_the_output_head_at_q6_k() {
        let info = dense("llama", 32, 32, 8, true);
        let mut sel = TypeSelector::new(&info, Ftype::Q6K);
        assert_eq!(
            sel.select("output.weight", 4096).expect("select"),
            GgufTensorType::Q6K
        );
    }

    #[test]
    fn q8_0_target_keeps_the_output_head_at_q8_0() {
        let info = dense("llama", 32, 32, 8, true);
        let mut sel = TypeSelector::new(&info, Ftype::Q8_0);
        assert_eq!(
            sel.select("output.weight", 4096).expect("select"),
            GgufTensorType::Q8_0,
            "the `new_type != Q8_0` guard keeps Q8_0 rather than dropping to Q6_K"
        );
    }

    #[test]
    fn attn_qkv_is_promoted_for_the_medium_mixtures() {
        let info = dense("phi2", 32, 32, 32, true);
        let mut sel = TypeSelector::new(&info, Ftype::Q4KM);
        assert_eq!(
            sel.select("blk.0.attn_qkv.weight", 2560).expect("select"),
            GgufTensorType::Q5K
        );
        let mut sel = TypeSelector::new(&info, Ftype::Q5KM);
        assert_eq!(
            sel.select("blk.0.attn_qkv.weight", 2560).expect("select"),
            GgufTensorType::Q6K
        );
    }

    #[test]
    fn q4_k_s_promotes_only_the_first_four_attn_v() {
        let info = dense("llama", 32, 32, 8, true);
        let mut sel = TypeSelector::new(&info, Ftype::Q4KS);
        for i in 0..6 {
            let ty = sel
                .select(&format!("blk.{i}.attn_v.weight"), 4096)
                .expect("select");
            let want = if i < 4 {
                GgufTensorType::Q5K
            } else {
                GgufTensorType::Q4K
            };
            assert_eq!(ty, want, "layer {i}");
        }
    }

    #[test]
    fn seventy_b_bumps_attn_v_to_q5_k() {
        let info = dense("llama", 80, 64, 8, true);
        let mut sel = TypeSelector::new(&info, Ftype::Q4KS);
        // i_attention_wv >= 4 would normally leave this at Q4_K.
        for _ in 0..4 {
            sel.select("blk.0.attn_v.weight", 8192).expect("select");
        }
        assert_eq!(
            sel.select("blk.4.attn_v.weight", 8192).expect("select"),
            GgufTensorType::Q5K
        );
    }

    #[test]
    fn eight_expert_models_bump_attn_v_and_attn_k_to_q8_0() {
        let mut info = dense("mixtral", 32, 32, 8, true);
        info.n_expert = 8;
        let mut sel = TypeSelector::new(&info, Ftype::Q4KM);
        assert_eq!(
            sel.select("blk.0.attn_v.weight", 4096).expect("select"),
            GgufTensorType::Q8_0
        );
        assert_eq!(
            sel.select("blk.0.attn_k.weight", 4096).expect("select"),
            GgufTensorType::Q8_0
        );
    }

    /// For a MoE model the counter no longer tracks the layer, so `layer_info`
    /// parses `blk.N.` instead — and rejects a name it cannot parse.
    #[test]
    fn moe_layer_index_comes_from_the_tensor_name() {
        let mut info = dense("mixtral", 32, 32, 8, true);
        info.n_expert = 8;
        let mut sel = TypeSelector::new(&info, Ftype::Q4KM);
        // Layer 4 is not in use_more_bits(·, 32); layer 6 is.
        assert_eq!(
            sel.select("blk.4.ffn_down_exps.weight", 4096)
                .expect("select"),
            GgufTensorType::Q4K
        );
        assert_eq!(
            sel.select("blk.6.ffn_down_exps.weight", 4096)
                .expect("select"),
            GgufTensorType::Q6K
        );
        assert!(sel.select("ffn_down.weight", 4096).is_err());
        assert!(sel.select("blk.999.ffn_down.weight", 4096).is_err());
    }

    #[test]
    fn should_quantize_rejects_the_documented_exclusions() {
        assert!(should_quantize("blk.0.attn_q.weight", 2));
        assert!(!should_quantize("blk.0.attn_norm.weight", 1));
        assert!(!should_quantize("blk.0.attn_norm.weight", 2));
        assert!(!should_quantize("output_norm.weight", 2));
        assert!(!should_quantize("blk.0.attn_q.bias", 2));
        assert!(!should_quantize("blk.0.ffn_gate_inp.weight", 2));
        assert!(!should_quantize("blk.0.ssm_conv1d.weight", 2));
        assert!(!should_quantize("blk.0.time_mix_w1.weight", 2));
        assert!(!should_quantize("position_embd.weight", 2));
        assert!(!should_quantize("token_types.weight", 2));
        assert!(!should_quantize("v.position_embd.weight", 2));
        assert!(!should_quantize("blk.0.attn_rel_b.weight", 2));
        assert!(!should_quantize("blk.0.altup_proj.weight", 2));
    }

    #[test]
    fn fallback_chain_matches_llama_cpp() {
        use GgufTensorType as T;
        // Divisible rows need no fallback.
        assert_eq!(fallback_for_incompatible_row(T::Q4K, 4096), None);
        assert_eq!(fallback_for_incompatible_row(T::Q6K, 512), None);
        // 96 is a multiple of 32 but not of 256.
        assert_eq!(
            fallback_for_incompatible_row(T::Q4K, 96),
            Some(Fallback::Use(T::Q5_0))
        );
        assert_eq!(
            fallback_for_incompatible_row(T::Q5K, 96),
            Some(Fallback::Use(T::Q5_1))
        );
        assert_eq!(
            fallback_for_incompatible_row(T::Q6K, 96),
            Some(Fallback::Use(T::Q8_0))
        );
        // 100 divides neither 256 nor 32, so the second guard picks F16.
        assert_eq!(
            fallback_for_incompatible_row(T::Q4K, 100),
            Some(Fallback::Use(T::F16))
        );
        // Q2_K/Q3_K want IQ4_NL, which has no encoder here.
        assert_eq!(
            fallback_for_incompatible_row(T::Q2K, 96),
            Some(Fallback::NeedsIq4Nl)
        );
        assert_eq!(
            fallback_for_incompatible_row(T::Q3K, 96),
            Some(Fallback::NeedsIq4Nl)
        );
        // A legacy type whose block size does not divide the row hits
        // llama.cpp's `default:` throw, not a silent F16 substitution.
        assert_eq!(
            fallback_for_incompatible_row(T::Q8_0, 8),
            Some(Fallback::Unsupported)
        );
        assert_eq!(
            fallback_for_incompatible_row(T::Q4_0, 20),
            Some(Fallback::Unsupported)
        );
    }

    #[test]
    fn ftype_values_match_llama_cpp() {
        assert_eq!(Ftype::Q4_0.file_type_value(), 2);
        assert_eq!(Ftype::Q8_0.file_type_value(), 7);
        assert_eq!(Ftype::Q2K.file_type_value(), 10);
        assert_eq!(Ftype::Q4KM.file_type_value(), 15);
        assert_eq!(Ftype::Q5KM.file_type_value(), 17);
        assert_eq!(Ftype::Q6K.file_type_value(), 18);
    }
}
