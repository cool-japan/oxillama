// PyO3 generates code that triggers `useless_conversion` and related lints
// because it wraps `PyResult<T>` at the function signature level.  This is a
// well-known false positive with PyO3 ≥ 0.20; suppress it crate-wide.
#![allow(clippy::useless_conversion)]

//! # oxillama-py
//!
//! PyO3 Python bindings for the OxiLLaMa Pure-Rust LLM inference engine.
//!
//! ## Quick start
//!
//! ```python
//! import oxillama_py
//!
//! config = oxillama_py.EngineConfig(model_path="model.gguf", context_size=4096)
//! engine = oxillama_py.Engine(config)
//! engine.load_model()
//!
//! text = engine.generate("Hello", max_tokens=128)
//! emb  = engine.embed("Hello world")   # List\[float\]
//! toks = engine.tokenize("Hello")      # List[int]
//!
//! engine.generate_streaming(
//!     "Hello",
//!     max_tokens=128,
//!     callback=lambda tok: print(tok, end="", flush=True),
//! )
//! ```
//!
//! ## Module structure
//!
//! | Python class         | Rust source       |
//! |----------------------|-------------------|
//! | `EngineConfig`       | `engine.rs`       |
//! | `Engine`             | `engine.rs`       |
//! | `SamplerConfig`      | `sampler.rs`      |
//! | `SpeculativeConfig`  | `speculative.rs`  |
//! | `SpeculativeEngine`  | `speculative.rs`  |
//! | `Lora`               | `lora.rs`         |

pub mod async_support;
pub mod callback;
pub mod cancel;
pub mod chat_template;
pub mod engine;
pub mod error;
#[cfg(feature = "hub")]
pub mod hub;
pub mod lora;
pub mod sampler;
pub mod speculative;
pub mod tokenizer;

use pyo3::prelude::*;

/// The `oxillama_py` Python extension module.
///
/// Registers all public Python classes.
#[pymodule]
fn oxillama_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyEngineConfig>()?;
    m.add_class::<engine::PyEngine>()?;
    m.add_class::<async_support::PyAsyncEngine>()?;
    m.add_class::<sampler::PySamplerConfig>()?;
    m.add_class::<speculative::PySpeculativeConfig>()?;
    m.add_class::<speculative::PySpeculativeEngine>()?;
    m.add_class::<lora::PyLora>()?;
    m.add_class::<tokenizer::PyTokenizer>()?;
    m.add_class::<cancel::PyCancellationToken>()?;
    error::register_exceptions(m)?;
    Ok(())
}
