//! Python wrapper for [`LoadedLora`].
//!
//! Exposes LoRA adapter loading and introspection to Python.

use pyo3::prelude::*;

use oxillama_arch::lora::LoadedLora;

use crate::error::arch_to_py;

/// A loaded LoRA adapter.
///
/// Load from a GGUF file with `Lora.load(path)`, then pass to
/// `Engine.apply_lora(path)`.
///
/// # Python Example
///
/// ```python
/// lora = Lora.load("adapter.gguf")
/// print(f"rank={lora.rank}, alpha={lora.alpha}, adapters={lora.num_adapters()}")
/// engine.apply_lora("adapter.gguf")
/// ```
#[pyclass(name = "Lora")]
pub struct PyLora {
    inner: LoadedLora,
}

#[pymethods]
#[allow(clippy::useless_conversion)]
impl PyLora {
    /// Load a LoRA adapter from a GGUF file.
    ///
    /// Args:
    ///     path: Filesystem path to the LoRA GGUF file.
    ///
    /// Returns:
    ///     Lora: loaded adapter object.
    ///
    /// Raises:
    ///     IOError:    if the file cannot be parsed.
    ///     ValueError: if tensors are missing or mismatched.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = LoadedLora::load(path).map_err(arch_to_py)?;
        Ok(Self { inner })
    }

    /// The LoRA rank (low-rank dimension `r`).
    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    /// The LoRA alpha (scale numerator).
    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    /// Number of adapted layers in this adapter.
    pub fn num_adapters(&self) -> usize {
        self.inner.num_adapters()
    }

    fn __repr__(&self) -> String {
        format!(
            "Lora(rank={}, alpha={}, num_adapters={})",
            self.inner.rank,
            self.inner.alpha,
            self.inner.num_adapters(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use oxillama_arch::lora::LoadedLora;
    use oxillama_quant::LoraAdapter;

    use super::*;

    /// A `PyLora` wrapping an empty adapter reports zero adapters.
    #[test]
    fn test_py_lora_empty_reports_zero_adapters() {
        let inner = LoadedLora {
            adapters: HashMap::new(),
            rank: 8,
            alpha: 8.0,
        };
        let py_lora = PyLora { inner };
        assert_eq!(py_lora.num_adapters(), 0);
        assert_eq!(py_lora.rank(), 8);
        assert!((py_lora.alpha() - 8.0).abs() < 1e-6);
    }

    /// Loading from a nonexistent path must return Err (not panic).
    #[test]
    fn test_load_missing_file_returns_err() {
        let path = std::env::temp_dir().join("oxillama_py_nonexistent_lora_xyz.gguf");
        let path_str = path.to_string_lossy();
        let result = LoadedLora::load(&path_str);
        assert!(
            result.is_err(),
            "loading a missing LoRA file should return Err"
        );
    }

    /// A `PyLora` with rank=16, alpha=32.0 reports those values correctly.
    #[test]
    fn test_py_lora_different_rank_and_alpha() {
        let inner = LoadedLora {
            adapters: HashMap::new(),
            rank: 16,
            alpha: 32.0,
        };
        let py_lora = PyLora { inner };
        assert_eq!(py_lora.rank(), 16);
        assert!((py_lora.alpha() - 32.0).abs() < 1e-6);
    }

    /// `__repr__` contains rank, alpha, and num_adapters.
    #[test]
    fn test_py_lora_repr_contains_rank_alpha_count() {
        let inner = LoadedLora {
            adapters: HashMap::new(),
            rank: 4,
            alpha: 16.0,
        };
        let py_lora = PyLora { inner };
        let repr = py_lora.__repr__();
        assert!(repr.contains('4'), "repr missing rank: {repr}");
        assert!(repr.contains("16"), "repr missing alpha: {repr}");
        assert!(repr.contains('0'), "repr missing num_adapters: {repr}");
    }

    /// A `PyLora` wrapping one adapter reports `num_adapters() == 1`.
    #[test]
    fn test_py_lora_one_adapter_reports_count() {
        let adapter =
            Arc::new(LoraAdapter::new(vec![1.0], vec![1.0], 1, 1.0, 1, 1).expect("valid adapter"));
        let mut adapters = HashMap::new();
        adapters.insert("blk.0.attn_q.weight".to_string(), adapter);

        let inner = LoadedLora {
            adapters,
            rank: 1,
            alpha: 1.0,
        };
        let py_lora = PyLora { inner };
        assert_eq!(py_lora.num_adapters(), 1);
    }

    /// `num_adapters()` returns the actual map size regardless of rank/alpha.
    #[test]
    fn test_py_lora_num_adapters_matches_map_size() {
        let a1 =
            Arc::new(LoraAdapter::new(vec![1.0], vec![1.0], 1, 1.0, 1, 1).expect("valid adapter"));
        let a2 =
            Arc::new(LoraAdapter::new(vec![2.0], vec![2.0], 1, 1.0, 1, 1).expect("valid adapter"));
        let mut adapters = HashMap::new();
        adapters.insert("layer.0".to_string(), a1);
        adapters.insert("layer.1".to_string(), a2);

        let inner = LoadedLora {
            adapters,
            rank: 8,
            alpha: 8.0,
        };
        let py_lora = PyLora { inner };
        assert_eq!(py_lora.num_adapters(), 2);
    }
}
