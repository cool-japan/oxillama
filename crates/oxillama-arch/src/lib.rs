//! # oxillama-arch
//!
//! Model architecture implementations for OxiLLaMa.
//!
//! This crate provides a trait-based architecture plugin system where each
//! model family (LLaMA, Qwen3, Mistral, etc.) implements the
//! [`ModelArchitecture`] and [`ForwardPass`] traits.
//!
//! ## Supported Architectures
//!
//! | Architecture | Feature Flag | Status |
//! |-------------|-------------|--------|
//! | LLaMA 3.x/4.x | `llama` | Planned |
//! | Qwen3 | `qwen3` | Planned (from OxiBonsai) |
//! | Mistral / Mixtral | `mistral` | Planned |
//! | Gemma 2/3 | `gemma` | Planned |
//! | Phi-3/4 | `phi` | Planned |

#[cfg(feature = "bloom")]
pub mod bloom;
#[cfg(feature = "command-r")]
pub mod command_r;
pub mod common;
pub mod config;
#[cfg(feature = "dbrx")]
pub mod dbrx;
#[cfg(feature = "deepseek")]
pub mod deepseek;
pub mod error;
#[cfg(feature = "falcon")]
pub mod falcon;
#[cfg(feature = "gemma")]
pub mod gemma;
#[cfg(feature = "gptneox")]
pub mod gpt_neox;
#[cfg(feature = "granite")]
pub mod granite;
#[cfg(feature = "grok")]
pub mod grok;
pub mod internlm3;
#[cfg(feature = "jamba")]
pub mod jamba;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "llava")]
pub mod llava;
#[cfg(feature = "llava16")]
pub mod llava_next;
pub mod lora;
#[cfg(feature = "mamba2")]
pub mod mamba2;
#[cfg(feature = "minicpm")]
pub mod minicpm;
#[cfg(feature = "mistral")]
pub mod mistral;
#[cfg(feature = "mixtral")]
pub mod mixtral;
#[cfg(feature = "olmo2")]
pub mod olmo2;
#[cfg(feature = "phi")]
pub mod phi;
#[cfg(feature = "phimoe")]
pub mod phi_moe;
#[cfg(feature = "qwen2-vl")]
pub mod qwen2_vl;
#[cfg(feature = "qwen3")]
pub mod qwen3;
pub mod reference;
pub mod registry;
#[cfg(feature = "stablelm")]
pub mod stablelm;
#[cfg(feature = "starcoder")]
pub mod starcoder;
pub mod traits;
pub mod yi;

pub use common::alibi::AlibiBias;
pub use common::rope::RopeScalingType;
pub use config::{ModelConfig, VisionConfig};
pub use error::{ArchError, ArchResult};
pub use lora::{LoadedLora, LoraAdapterTrait, LoraDelta, LoraStack, TargetModule};
pub use registry::ArchitectureRegistry;
pub use traits::{
    BatchedKvView, ForwardPass, KvCacheAccess, KvSlot, ModelArchitecture, TensorNamePattern,
};
