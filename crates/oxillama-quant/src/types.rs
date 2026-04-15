//! Quantization data types and tensor wrapper.

use oxillama_gguf::GgufTensorType;

/// A quantized tensor — raw block data plus shape and type metadata.
///
/// The data is stored as raw bytes in the GGUF block format.
/// Use a [`QuantKernel`](crate::traits::QuantKernel) to dequantize or
/// perform fused operations.
#[derive(Debug, Clone)]
pub struct QuantTensor {
    /// Raw block data (packed quantized weights).
    pub data: Vec<u8>,
    /// Tensor shape (e.g., [out_features, in_features] for a linear layer).
    pub shape: Vec<usize>,
    /// The GGUF quantization type.
    pub tensor_type: GgufTensorType,
}

impl QuantTensor {
    /// Create a new quantized tensor.
    pub fn new(data: Vec<u8>, shape: Vec<usize>, tensor_type: GgufTensorType) -> Self {
        Self {
            data,
            shape,
            tensor_type,
        }
    }

    /// Total number of elements (product of all dimensions).
    pub fn n_elements(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().product()
    }

    /// Number of quantized blocks in this tensor.
    pub fn n_blocks(&self) -> usize {
        let block_size = self.tensor_type.block_size();
        if block_size == 0 {
            return 0;
        }
        self.n_elements().div_ceil(block_size)
    }

    /// Expected total data size in bytes.
    pub fn expected_data_size(&self) -> usize {
        self.n_blocks() * self.tensor_type.block_bytes()
    }
}

/// Information about a quantization block format.
#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    /// Number of weights per block.
    pub block_size: usize,
    /// Number of bytes per block.
    pub block_bytes: usize,
    /// Effective bits per weight.
    pub bits_per_weight: f32,
}

impl BlockInfo {
    /// Compute block info for a given GGUF tensor type.
    pub fn for_type(tensor_type: GgufTensorType) -> Self {
        let block_size = tensor_type.block_size();
        let block_bytes = tensor_type.block_bytes();
        let bits_per_weight = if block_size > 0 {
            (block_bytes as f32 * 8.0) / block_size as f32
        } else {
            0.0
        };
        Self {
            block_size,
            block_bytes,
            bits_per_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxillama_gguf::GgufTensorType;

    #[test]
    fn test_quant_tensor_n_elements_2d() {
        let t = QuantTensor::new(vec![0u8; 32], vec![4, 8], GgufTensorType::Q8_0);
        assert_eq!(t.n_elements(), 32);
    }

    #[test]
    fn test_quant_tensor_n_elements_empty_shape() {
        let t = QuantTensor::new(vec![], vec![], GgufTensorType::F32);
        assert_eq!(t.n_elements(), 0);
    }

    #[test]
    fn test_quant_tensor_n_blocks_q4_0() {
        // Q4_0: 32 weights per block
        // 64 elements → 2 blocks
        let block_bytes = GgufTensorType::Q4_0.block_bytes() * 2;
        let t = QuantTensor::new(vec![0u8; block_bytes], vec![64], GgufTensorType::Q4_0);
        assert_eq!(t.n_blocks(), 2);
    }

    #[test]
    fn test_quant_tensor_expected_data_size_f32() {
        // F32: 1 weight per block, 4 bytes per block
        let t = QuantTensor::new(vec![0u8; 20], vec![5], GgufTensorType::F32);
        assert_eq!(t.expected_data_size(), 20); // 5 * 4
    }

    #[test]
    fn test_block_info_for_q8_0() {
        let info = BlockInfo::for_type(GgufTensorType::Q8_0);
        assert_eq!(info.block_size, 32);
        assert_eq!(info.block_bytes, 34); // 2 (scale) + 32 (quants)
        assert!(info.bits_per_weight > 0.0);
    }

    #[test]
    fn test_block_info_bits_per_weight_q4_0() {
        let info = BlockInfo::for_type(GgufTensorType::Q4_0);
        // Q4_0: 18 bytes per 32 weights → (18*8)/32 = 4.5
        let expected = (18.0f32 * 8.0) / 32.0;
        assert!(
            (info.bits_per_weight - expected).abs() < 0.01,
            "bits_per_weight: {} vs expected {}",
            info.bits_per_weight,
            expected
        );
    }

    #[test]
    fn test_quant_tensor_clone() {
        let t = QuantTensor::new(vec![1u8, 2, 3, 4], vec![2, 2], GgufTensorType::F32);
        let t2 = t.clone();
        assert_eq!(t2.data, t.data);
        assert_eq!(t2.shape, t.shape);
    }
}
