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

#[cfg(feature = "command-r")]
pub mod command_r;
pub mod common;
pub mod config;
pub mod error;
#[cfg(feature = "gemma")]
pub mod gemma;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "llava")]
pub mod llava;
pub mod lora;
#[cfg(feature = "mistral")]
pub mod mistral;
#[cfg(feature = "phi")]
pub mod phi;
#[cfg(feature = "qwen3")]
pub mod qwen3;
pub mod registry;
#[cfg(feature = "starcoder")]
pub mod starcoder;
pub mod traits;

pub use config::ModelConfig;
pub use error::{ArchError, ArchResult};
pub use lora::LoadedLora;
pub use registry::ArchitectureRegistry;
pub use traits::{ForwardPass, KvCacheAccess, ModelArchitecture, TensorNamePattern};
