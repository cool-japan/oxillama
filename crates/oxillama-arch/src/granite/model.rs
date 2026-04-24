//! IBM Granite-3.x architecture plugin struct.

/// Granite-3.x (IBM) dense decoder-only architecture plugin.
///
/// Granite-3.x follows the same topology as LLaMA with:
/// - RMSNorm pre-normalization on attention and FFN sub-layers
/// - Additional per-head QKV RMSNorm projections (no bias)
/// - Grouped-query attention (GQA) with RoPE
/// - SwiGLU feed-forward network
/// - Tied input/output embeddings
///
/// Registered under GGUF `general.architecture` = `"granite"`.
pub struct GraniteArchitecture;

impl GraniteArchitecture {
    /// Create a new Granite architecture plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraniteArchitecture {
    fn default() -> Self {
        Self::new()
    }
}
