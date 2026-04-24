//! InternLM3 architecture plugin struct.

/// InternLM3 dense decoder-only architecture plugin.
///
/// InternLM3 follows the same topology as LLaMA with:
/// - RMSNorm pre-normalization on attention and FFN sub-layers
/// - Grouped-query attention (GQA) with RoPE
/// - SwiGLU feed-forward network (ReLU² variant treated as SwiGLU)
/// - Tied input/output embeddings
///
/// Registered under GGUF `general.architecture` = `"internlm3"`.
pub struct InternLm3Architecture;

impl InternLm3Architecture {
    /// Create a new InternLM3 architecture plugin instance.
    pub fn new() -> Self {
        Self
    }
}

impl Default for InternLm3Architecture {
    fn default() -> Self {
        Self::new()
    }
}
