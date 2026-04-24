//! Core types for the GGUF format — quantization type IDs, value types, and constants.

/// GGUF magic number: "GGUF" in little-endian (0x46475547).
pub const GGUF_MAGIC: u32 = 0x4755_4746;

/// Default tensor data alignment in bytes.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF metadata value type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    /// 8-bit unsigned integer.
    Uint8 = 0,
    /// 8-bit signed integer.
    Int8 = 1,
    /// 16-bit unsigned integer.
    Uint16 = 2,
    /// 16-bit signed integer.
    Int16 = 3,
    /// 32-bit unsigned integer.
    Uint32 = 4,
    /// 32-bit signed integer.
    Int32 = 5,
    /// 32-bit IEEE 754 float.
    Float32 = 6,
    /// Boolean value.
    Bool = 7,
    /// UTF-8 string.
    String = 8,
    /// Array of values (homogeneous type).
    Array = 9,
    /// 64-bit unsigned integer.
    Uint64 = 10,
    /// 64-bit signed integer.
    Int64 = 11,
    /// 64-bit IEEE 754 float.
    Float64 = 12,
}

impl GgufValueType {
    /// Try to convert a u32 to a `GgufValueType`.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// GGUF tensor data type (quantization format).
///
/// Maps to the `ggml_type` enum in llama.cpp. Each variant represents a
/// different quantization format with its own block structure and bits-per-weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufTensorType {
    /// 32-bit IEEE 754 float.
    F32 = 0,
    /// 16-bit IEEE 754 float.
    F16 = 1,
    /// 4-bit quantization, type 0 (legacy). Block: 2-byte scale + 16 nibbles = 18 bytes / 32 weights.
    Q4_0 = 2,
    /// 4-bit quantization, type 1 (legacy). Block: 2-byte scale + 2-byte min + 16 nibbles.
    Q4_1 = 3,
    /// 5-bit quantization, type 0 (legacy).
    Q5_0 = 6,
    /// 5-bit quantization, type 1 (legacy).
    Q5_1 = 7,
    /// 8-bit quantization, type 0. Block: 2-byte scale + 32 int8 = 34 bytes / 32 weights.
    Q8_0 = 8,
    /// 8-bit quantization, type 1.
    Q8_1 = 9,
    /// 2-bit K-quant.
    Q2K = 10,
    /// 3-bit K-quant.
    Q3K = 11,
    /// 4-bit K-quant.
    Q4K = 12,
    /// 5-bit K-quant.
    Q5K = 13,
    /// 6-bit K-quant.
    Q6K = 14,
    /// 8-bit K-quant.
    Q8K = 15,
    /// IQ2_XXS: 2-bit importance quantization (extra-extra-small).
    Iq2Xxs = 16,
    /// IQ2_XS: 2-bit importance quantization (extra-small).
    Iq2Xs = 17,
    /// IQ3_XXS: 3-bit importance quantization (extra-extra-small).
    Iq3Xxs = 18,
    /// IQ1_S: 1-bit importance quantization (small).
    Iq1S = 19,
    /// IQ4_NL: 4-bit importance quantization (non-linear).
    Iq4Nl = 20,
    /// IQ3_S: 3-bit importance quantization (small).
    Iq3S = 21,
    /// IQ2_S: 2-bit importance quantization (small).
    Iq2S = 22,
    /// IQ4_XS: 4-bit importance quantization (extra-small).
    Iq4Xs = 23,
    /// 8-bit integer (not quantized, direct storage).
    I8 = 24,
    /// 16-bit integer.
    I16 = 25,
    /// 32-bit integer.
    I32 = 26,
    /// 64-bit integer.
    I64 = 27,
    /// 64-bit IEEE 754 float.
    F64 = 28,
    /// IQ1_M: 1-bit importance quantization (medium).
    Iq1M = 29,
    /// BF16: Brain floating-point 16-bit.
    Bf16 = 30,
    /// Q4_0 with 4-bit offsets (experimental).
    Q4_0_4_4 = 31,
    /// Q4_0 with 4/8 offsets (experimental).
    Q4_0_4_8 = 32,
    /// Q4_0 with 8/8 offsets (experimental).
    Q4_0_8_8 = 33,
    /// TQ1_0: Ternary quantization (1-bit with ternary encoding).
    Tq1_0 = 34,
    /// TQ2_0: Ternary quantization (2-bit).
    Tq2_0 = 35,
    /// IQ4_NL with 4-bit offsets (experimental).
    Iq4Nl4x4 = 36,
    /// IQ4_NL with 4/8 offsets (experimental).
    Iq4Nl4x8 = 37,
    /// IQ4_NL with 8/8 offsets (experimental).
    Iq4Nl8x8 = 38,
    /// Q1_0_G128: 1-bit quantization with group size 128 (PrismML Bonsai format).
    /// Block: 2-byte FP16 scale + 16 bytes (128 bits) = 18 bytes / 128 weights.
    Q1_0G128 = 39,
}

impl GgufTensorType {
    /// Try to convert a u32 to a `GgufTensorType`.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::Iq1M),
            30 => Some(Self::Bf16),
            31 => Some(Self::Q4_0_4_4),
            32 => Some(Self::Q4_0_4_8),
            33 => Some(Self::Q4_0_8_8),
            34 => Some(Self::Tq1_0),
            35 => Some(Self::Tq2_0),
            36 => Some(Self::Iq4Nl4x4),
            37 => Some(Self::Iq4Nl4x8),
            38 => Some(Self::Iq4Nl8x8),
            39 => Some(Self::Q1_0G128),
            _ => None,
        }
    }

    /// Returns the block size (number of weights per block) for this type.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::Iq2Xxs | Self::Iq2Xs | Self::Iq2S => 256,
            Self::Iq3Xxs | Self::Iq3S => 256,
            Self::Iq1S | Self::Iq1M => 256,
            Self::Iq4Nl => 32,
            Self::Iq4Xs => 256,
            Self::Q4_0_4_4 | Self::Q4_0_4_8 | Self::Q4_0_8_8 => 32,
            Self::Iq4Nl4x4 | Self::Iq4Nl4x8 | Self::Iq4Nl8x8 => 32,
            Self::Tq1_0 | Self::Tq2_0 => 256,
            Self::Q1_0G128 => 128,
        }
    }

    /// Returns the number of bytes per block for this type.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18, // 2 (scale) + 16 (4-bit × 32)
            Self::Q4_1 => 20, // 2 (scale) + 2 (min) + 16
            Self::Q5_0 => 22, // 2 (scale) + 4 (high bits) + 16
            Self::Q5_1 => 24, // 2 + 2 + 4 + 16
            Self::Q8_0 => 34, // 2 (scale) + 32 (int8)
            Self::Q8_1 => 36, // 2 + 2 + 32
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq2S => 82,
            Self::Iq3Xxs => 98,
            Self::Iq3S => 110,
            Self::Iq1S => 50,
            Self::Iq1M => 56,
            Self::Iq4Nl => 18, // 2 (FP16 scale) + 16 (32 × 4-bit non-linear)
            Self::Iq4Xs => 136,
            Self::Q4_0_4_4 => 18,
            Self::Q4_0_4_8 => 18,
            Self::Q4_0_8_8 => 18,
            Self::Iq4Nl4x4 => 34,
            Self::Iq4Nl4x8 => 34,
            Self::Iq4Nl8x8 => 34,
            Self::Tq1_0 => 54,
            Self::Tq2_0 => 66,
            Self::Q1_0G128 => 18, // 2 (FP16 scale) + 16 (128 bits)
        }
    }

    /// Returns the display name for this quantization type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq2S => "IQ2_S",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq3S => "IQ3_S",
            Self::Iq1S => "IQ1_S",
            Self::Iq1M => "IQ1_M",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq4Xs => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Bf16 => "BF16",
            Self::Q4_0_4_4 => "Q4_0_4x4",
            Self::Q4_0_4_8 => "Q4_0_4x8",
            Self::Q4_0_8_8 => "Q4_0_8x8",
            Self::Iq4Nl4x4 => "IQ4_NL_4x4",
            Self::Iq4Nl4x8 => "IQ4_NL_4x8",
            Self::Iq4Nl8x8 => "IQ4_NL_8x8",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
            Self::Q1_0G128 => "Q1_0_G128",
        }
    }
}

impl core::fmt::Display for GgufTensorType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GgufValueType tests ──────────────────────────────────────────────────

    #[test]
    fn test_value_type_from_u32_all_valid() {
        let expected = [
            (0, GgufValueType::Uint8),
            (1, GgufValueType::Int8),
            (2, GgufValueType::Uint16),
            (3, GgufValueType::Int16),
            (4, GgufValueType::Uint32),
            (5, GgufValueType::Int32),
            (6, GgufValueType::Float32),
            (7, GgufValueType::Bool),
            (8, GgufValueType::String),
            (9, GgufValueType::Array),
            (10, GgufValueType::Uint64),
            (11, GgufValueType::Int64),
            (12, GgufValueType::Float64),
        ];
        for (raw, expected_variant) in expected {
            let got = GgufValueType::from_u32(raw)
                .unwrap_or_else(|| panic!("from_u32({raw}) should succeed"));
            assert_eq!(got, expected_variant, "from_u32({raw}) mismatch");
        }
    }

    #[test]
    fn test_value_type_from_u32_round_trip() {
        // Every defined variant should serialise to its discriminant and back.
        let variants = [
            GgufValueType::Uint8,
            GgufValueType::Int8,
            GgufValueType::Uint16,
            GgufValueType::Int16,
            GgufValueType::Uint32,
            GgufValueType::Int32,
            GgufValueType::Float32,
            GgufValueType::Bool,
            GgufValueType::String,
            GgufValueType::Array,
            GgufValueType::Uint64,
            GgufValueType::Int64,
            GgufValueType::Float64,
        ];
        for v in variants {
            let raw = v as u32;
            let back = GgufValueType::from_u32(raw)
                .unwrap_or_else(|| panic!("round-trip failed for {v:?} (discriminant {raw})"));
            assert_eq!(back, v, "round-trip value mismatch for {v:?}");
        }
    }

    #[test]
    fn test_value_type_from_u32_out_of_range() {
        // 13+ should return None.
        for bad in [13u32, 100, u32::MAX] {
            assert!(
                GgufValueType::from_u32(bad).is_none(),
                "from_u32({bad}) should return None"
            );
        }
    }

    // ── GgufTensorType tests ─────────────────────────────────────────────────

    #[test]
    fn test_tensor_type_from_u32_round_trip() {
        // Exhaustive list of all defined discriminants.
        let discriminants: &[u32] = &[
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ];
        for &d in discriminants {
            let t = GgufTensorType::from_u32(d)
                .unwrap_or_else(|| panic!("from_u32({d}) should succeed"));
            assert_eq!(t as u32, d, "discriminant mismatch for {t:?}");
        }
    }

    #[test]
    fn test_tensor_type_from_u32_gaps_return_none() {
        // Values 4 and 5 are intentionally absent (removed Q4_2, Q4_3).
        for gap in [4u32, 5] {
            assert!(
                GgufTensorType::from_u32(gap).is_none(),
                "from_u32({gap}) should return None (gap variant)"
            );
        }
    }

    #[test]
    fn test_tensor_type_from_u32_out_of_range_returns_none() {
        for bad in [40u32, 255, 1000, u32::MAX] {
            assert!(
                GgufTensorType::from_u32(bad).is_none(),
                "from_u32({bad}) should return None"
            );
        }
    }

    #[test]
    fn test_tensor_type_names_non_empty() {
        // Every defined type must have a non-empty display name.
        let all_discriminants: &[u32] = &[
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ];
        for &d in all_discriminants {
            let t =
                GgufTensorType::from_u32(d).unwrap_or_else(|| panic!("from_u32({d}) must succeed"));
            assert!(!t.name().is_empty(), "name() is empty for discriminant {d}");
        }
    }

    #[test]
    fn test_tensor_type_display_equals_name() {
        // Display impl delegates to name(); verify consistency.
        use std::fmt::Write as _;
        let all_discriminants: &[u32] = &[
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ];
        for &d in all_discriminants {
            let t =
                GgufTensorType::from_u32(d).unwrap_or_else(|| panic!("from_u32({d}) must succeed"));
            let mut s = String::new();
            write!(s, "{t}").expect("Display must not fail");
            assert_eq!(s, t.name(), "Display != name() for {t:?}");
        }
    }

    #[test]
    fn test_tensor_type_block_size_positive() {
        // block_size() must be >= 1 for every type.
        let all_discriminants: &[u32] = &[
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ];
        for &d in all_discriminants {
            let t =
                GgufTensorType::from_u32(d).unwrap_or_else(|| panic!("from_u32({d}) must succeed"));
            assert!(t.block_size() >= 1, "block_size() == 0 for {t:?}");
        }
    }

    #[test]
    fn test_tensor_type_block_bytes_positive() {
        // block_bytes() must be >= 1 for every type.
        let all_discriminants: &[u32] = &[
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ];
        for &d in all_discriminants {
            let t =
                GgufTensorType::from_u32(d).unwrap_or_else(|| panic!("from_u32({d}) must succeed"));
            assert!(t.block_bytes() >= 1, "block_bytes() == 0 for {t:?}");
        }
    }

    // ── Specific layout invariants ────────────────────────────────────────────

    #[test]
    fn test_q4_0_layout() {
        let t = GgufTensorType::Q4_0;
        assert_eq!(t.block_size(), 32, "Q4_0 block_size");
        assert_eq!(
            t.block_bytes(),
            18,
            "Q4_0 block_bytes: 2 scale + 16 nibbles"
        );
        assert_eq!(t.name(), "Q4_0");
    }

    #[test]
    fn test_q8_0_layout() {
        let t = GgufTensorType::Q8_0;
        assert_eq!(t.block_size(), 32, "Q8_0 block_size");
        assert_eq!(t.block_bytes(), 34, "Q8_0 block_bytes: 2 scale + 32 int8");
        assert_eq!(t.name(), "Q8_0");
    }

    #[test]
    fn test_q1_0g128_layout() {
        let t = GgufTensorType::Q1_0G128;
        assert_eq!(t.block_size(), 128, "Q1_0G128 block_size is 128");
        assert_eq!(
            t.block_bytes(),
            18,
            "Q1_0G128 block_bytes: 2 FP16 scale + 16 bytes"
        );
        assert_eq!(t.name(), "Q1_0_G128");
    }

    #[test]
    fn test_iq4_nl_layout() {
        let t = GgufTensorType::Iq4Nl;
        assert_eq!(t.block_size(), 32, "IQ4_NL block_size");
        assert_eq!(t.block_bytes(), 18, "IQ4_NL block_bytes");
        assert_eq!(t.name(), "IQ4_NL");
    }

    #[test]
    fn test_scalar_types_block_size_one() {
        // All floating-point and integer scalar types have block_size == 1.
        for t in [
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::Bf16,
            GgufTensorType::F64,
            GgufTensorType::I8,
            GgufTensorType::I16,
            GgufTensorType::I32,
            GgufTensorType::I64,
        ] {
            assert_eq!(
                t.block_size(),
                1,
                "scalar type {t:?} should have block_size 1"
            );
        }
    }

    #[test]
    fn test_k_quants_block_size_256() {
        for t in [
            GgufTensorType::Q2K,
            GgufTensorType::Q3K,
            GgufTensorType::Q4K,
            GgufTensorType::Q5K,
            GgufTensorType::Q6K,
            GgufTensorType::Q8K,
        ] {
            assert_eq!(
                t.block_size(),
                256,
                "K-quant {t:?} should have block_size 256"
            );
        }
    }

    #[test]
    fn test_scalar_type_bytes_match_width() {
        assert_eq!(GgufTensorType::F32.block_bytes(), 4);
        assert_eq!(GgufTensorType::F16.block_bytes(), 2);
        assert_eq!(GgufTensorType::Bf16.block_bytes(), 2);
        assert_eq!(GgufTensorType::F64.block_bytes(), 8);
        assert_eq!(GgufTensorType::I8.block_bytes(), 1);
        assert_eq!(GgufTensorType::I16.block_bytes(), 2);
        assert_eq!(GgufTensorType::I32.block_bytes(), 4);
        assert_eq!(GgufTensorType::I64.block_bytes(), 8);
    }

    #[test]
    fn test_gguf_magic_constant() {
        // Magic "GGUF" as little-endian u32.
        assert_eq!(GGUF_MAGIC, 0x4755_4746);
    }

    #[test]
    fn test_gguf_default_alignment() {
        assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
    }
}
