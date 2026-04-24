//! Yi architecture plugin struct.

/// Yi dense decoder-only architecture plugin.
///
/// Yi follows the same topology as LLaMA with:
/// - RMSNorm pre-normalization on attention and FFN sub-layers
/// - Grouped-query attention (GQA) with RoPE
/// - SwiGLU feed-forward network
/// - Tied input/output embeddings
///
/// Registered under GGUF `general.architecture` = `"yi"`.
pub struct YiArchitecture;

impl YiArchitecture {
    /// Create a new Yi architecture plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for YiArchitecture {
    fn default() -> Self {
        Self::new()
    }
}
