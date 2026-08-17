//! Command-line argument types shared by more than one subcommand.
//!
//! These live outside `main.rs` so that `run` and `serve` can be defined in
//! their own modules while still spelling the same flag values. Everything
//! here is `pub(crate)`: it is CLI surface, not a library API.

use clap::ValueEnum;
use colored::Colorize;

/// Element type used for KV cache storage, as spelled on the command line.
///
/// A local mirror of [`oxillama_runtime::KvCacheDtype`]: `ValueEnum` cannot be
/// derived for a type defined in another crate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum KvDtypeArg {
    /// 32-bit float storage — lossless, and the OxiLLaMa default.
    #[default]
    F32,
    /// 16-bit float storage — half the KV memory, `f16` rounding per element.
    F16,
}

/// Override for whether BOS is prepended to the prompt, as spelled on the
/// command line.
///
/// llama.cpp's default for a GGUF that lacks `tokenizer.ggml.pre` is
/// `add_bos = false`. OxiLLaMa instead probes the vocabulary for the
/// LLaMA-3 special tokens and, on a match, infers the LLaMA-3 pre-tokenizer
/// — which implies `add_bos = true` — so it can tokenize correctly without
/// that metadata (see `pretok::implies_add_bos`). That inference is
/// deliberate, but it means OxiLLaMa's token ids differ from llama.cpp's by
/// exactly one leading BOS token on such files. This flag lets a caller
/// override the inferred policy in either direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, ValueEnum)]
pub(crate) enum AddBosArg {
    /// Use OxiLLaMa's own policy: the tokenizer's `add_bos_token` metadata,
    /// or the pre-tokenizer's inferred default when metadata is silent.
    /// Unchanged default behavior.
    #[default]
    Auto,
    /// Force a BOS token onto the front of the prompt even when the
    /// tokenizer's own policy would not add one.
    Always,
    /// Never prepend BOS, regardless of the tokenizer's own policy. This is
    /// the override that reproduces llama.cpp's `add_bos = false` behavior
    /// on a LLaMA-3 GGUF lacking `tokenizer.ggml.pre`.
    Never,
}

impl From<KvDtypeArg> for oxillama_runtime::KvCacheDtype {
    fn from(arg: KvDtypeArg) -> Self {
        match arg {
            KvDtypeArg::F32 => Self::F32,
            KvDtypeArg::F16 => Self::F16,
        }
    }
}

/// Tokenize `text`, honoring `mode`'s BOS override (see [`AddBosArg`]).
///
/// * `Auto` defers entirely to the tokenizer's own baked-in policy
///   (`add_special: true`) — bit-for-bit the same ids `run` produced before
///   this flag existed.
/// * `Never` disables BOS/EOS insertion outright (`add_special: false`).
/// * `Always` starts from that same BOS-free encoding and manually prepends
///   the model's BOS id when the tokenizer's own policy did not already add
///   one. This is the one case a plain `add_special` boolean cannot express:
///   it can only switch the tokenizer's baked-in policy on or off, never
///   force an "off" policy to "on".
///
/// # Errors
///
/// Propagates any tokenizer encoding failure.
pub(crate) fn tokenize_with_add_bos(
    tokenizer: &oxillama_runtime::TokenizerBridge,
    text: &str,
    mode: AddBosArg,
) -> anyhow::Result<Vec<u32>> {
    match mode {
        AddBosArg::Auto => Ok(tokenizer.encode_with(text, true, true)?),
        AddBosArg::Never => Ok(tokenizer.encode_with(text, false, true)?),
        AddBosArg::Always => {
            let mut ids = tokenizer.encode_with(text, false, true)?;
            if let Some(bos) = tokenizer.bos_token_id() {
                if ids.first() != Some(&bos) {
                    ids.insert(0, bos);
                }
            }
            Ok(ids)
        }
    }
}

/// Build the engine's GPU policy from the three `--gpu*` flags.
///
/// `gpu == false` yields [`GpuPolicy::Off`](oxillama_runtime::GpuPolicy::Off)
/// regardless of the other two arguments; clap's `requires = "gpu"` already
/// rejects that combination on the command line.
///
/// `device` is read as a device index when it parses as a `usize` and as a
/// case-insensitive adapter-name substring otherwise, so `--gpu-device 0` and
/// `--gpu-device "Apple M3"` both work without a second flag.
///
/// `min_weight_elems` is deliberately left at `None` (the runtime's built-in
/// threshold): it is a tuning constant, not user-facing surface.
///
/// These flags bypass the TOML config and the model profile entirely — a GPU
/// is a property of the machine, not of the model, and the profile
/// precedence chain must never be able to silently switch an explicit `--gpu`
/// back off.
// Unused outside `#[cfg(test)]` until CLI/Python GPU wiring lands: neither
// `run` nor `serve` defines `--gpu`/`--gpu-device`/`--n-gpu-layers` yet (see
// TODO.md's "GPU CLI/Python wiring" entry).
#[allow(dead_code)]
pub(crate) fn gpu_policy_from_flags(
    gpu: bool,
    device: Option<String>,
    n_gpu_layers: Option<usize>,
) -> oxillama_runtime::GpuPolicy {
    if !gpu {
        return oxillama_runtime::GpuPolicy::Off;
    }
    let device = device.map(|spec| match spec.parse::<usize>() {
        Ok(index) => oxillama_runtime::GpuDeviceSelector::Index(index),
        Err(_) => oxillama_runtime::GpuDeviceSelector::Name(spec),
    });
    oxillama_runtime::GpuPolicy::On(oxillama_runtime::GpuOptions {
        device,
        n_gpu_layers,
        min_weight_elems: None,
    })
}

/// Report what the GPU ended up holding, on stderr, after a successful load.
///
/// `status` is `None` whenever no device was initialised (the CPU path, and
/// every build without the `gpu` feature), and nothing is printed then. A
/// `Some` status with zero resident tensors is a successful load on an idle
/// device, not a failure, so it is reported as a hint rather than an error.
///
/// stderr, not stdout: `run` streams generated text to stdout and callers
/// pipe it.
// Unused outside `#[cfg(test)]` until CLI/Python GPU wiring lands: neither
// `run` nor `serve` calls this yet (see TODO.md's "GPU CLI/Python wiring"
// entry).
#[allow(dead_code)]
pub(crate) fn print_gpu_banner(status: Option<&oxillama_runtime::GpuStatus>) {
    let Some(status) = status else {
        return;
    };
    const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let resident_gib = status.resident_bytes as f64 / BYTES_PER_GIB;
    eprintln!(
        "{}",
        format!(
            "GPU: {} ({}) — {} tensors resident ({resident_gib:.2} GiB), {} on CPU",
            status.backend, status.device_name, status.resident_tensors, status.cpu_tensors
        )
        .green()
    );
    if status.resident_tensors == 0 {
        eprintln!(
            "{}",
            "  hint: this model has no Q4_0 tensors above the offload size threshold; \
             inference runs entirely on the CPU"
                .yellow()
        );
    }
    if status.upload_failures > 0 {
        eprintln!(
            "{}",
            format!(
                "  warning: {} eligible tensor(s) failed to upload and stayed on the CPU",
                status.upload_failures
            )
            .yellow()
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_runtime::{GpuDeviceSelector, GpuOptions, GpuPolicy};

    #[test]
    fn gpu_policy_is_off_without_the_flag() {
        // Even a stray device/layer value cannot switch the GPU on: `--gpu`
        // is the single source of truth for the policy variant.
        assert_eq!(
            gpu_policy_from_flags(false, Some("0".to_string()), Some(8)),
            GpuPolicy::Off
        );
    }

    #[test]
    fn gpu_policy_defaults_to_auto_selection() {
        assert_eq!(
            gpu_policy_from_flags(true, None, None),
            GpuPolicy::On(GpuOptions {
                device: None,
                n_gpu_layers: None,
                min_weight_elems: None,
            })
        );
    }

    #[test]
    fn numeric_device_spec_parses_as_an_index() {
        match gpu_policy_from_flags(true, Some("2".to_string()), None) {
            GpuPolicy::On(opts) => {
                assert_eq!(opts.device, Some(GpuDeviceSelector::Index(2)));
            }
            GpuPolicy::Off => panic!("expected GpuPolicy::On"),
        }
    }

    #[test]
    fn non_numeric_device_spec_parses_as_a_name() {
        match gpu_policy_from_flags(true, Some("Apple M3 Max".to_string()), None) {
            GpuPolicy::On(opts) => {
                assert_eq!(
                    opts.device,
                    Some(GpuDeviceSelector::Name("Apple M3 Max".to_string()))
                );
            }
            GpuPolicy::Off => panic!("expected GpuPolicy::On"),
        }
    }

    #[test]
    fn device_spec_that_only_starts_with_digits_is_a_name() {
        // "1080" is an index but "1080 Ti" is an adapter name; the split is
        // whole-string `usize` parsing, not a leading-digit heuristic.
        match gpu_policy_from_flags(true, Some("1080 Ti".to_string()), None) {
            GpuPolicy::On(opts) => {
                assert_eq!(
                    opts.device,
                    Some(GpuDeviceSelector::Name("1080 Ti".to_string()))
                );
            }
            GpuPolicy::Off => panic!("expected GpuPolicy::On"),
        }
    }

    #[test]
    fn n_gpu_layers_is_forwarded_verbatim() {
        // Including 0, which the runtime reads as "offload nothing" — it must
        // not collapse into `None` ("offload everything").
        match gpu_policy_from_flags(true, None, Some(0)) {
            GpuPolicy::On(opts) => assert_eq!(opts.n_gpu_layers, Some(0)),
            GpuPolicy::Off => panic!("expected GpuPolicy::On"),
        }
    }
}
