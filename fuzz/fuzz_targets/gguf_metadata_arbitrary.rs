#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use oxillama_gguf::MetadataValue;

/// A scalar leaf value that maps 1-to-1 onto [`MetadataValue`] variants.
#[derive(Debug, Arbitrary)]
enum ArbScalar {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
}

impl ArbScalar {
    fn into_metadata(self) -> MetadataValue {
        match self {
            ArbScalar::U8(v) => MetadataValue::Uint8(v),
            ArbScalar::I8(v) => MetadataValue::Int8(v),
            ArbScalar::U16(v) => MetadataValue::Uint16(v),
            ArbScalar::I16(v) => MetadataValue::Int16(v),
            ArbScalar::U32(v) => MetadataValue::Uint32(v),
            ArbScalar::I32(v) => MetadataValue::Int32(v),
            ArbScalar::U64(v) => MetadataValue::Uint64(v),
            ArbScalar::I64(v) => MetadataValue::Int64(v),
            ArbScalar::F32(v) => MetadataValue::Float32(v),
            ArbScalar::F64(v) => MetadataValue::Float64(v),
            ArbScalar::Bool(v) => MetadataValue::Bool(v),
            ArbScalar::Str(v) => MetadataValue::String(v),
        }
    }
}

/// A structured metadata entry: either a scalar or a homogeneous array of
/// up to 16 scalars (mirrors the `MetadataValue::Array` variant).
#[derive(Debug, Arbitrary)]
enum ArbMetadata {
    Scalar(ArbScalar),
    Array(Vec<ArbScalar>),
}

impl ArbMetadata {
    fn into_metadata(self) -> MetadataValue {
        match self {
            ArbMetadata::Scalar(s) => s.into_metadata(),
            ArbMetadata::Array(arr) => {
                MetadataValue::Array(arr.into_iter().map(ArbScalar::into_metadata).collect())
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);

    // Generate up to 8 key-value pairs with structured, typed values.
    let count: usize = u.int_in_range(0..=8).unwrap_or(0);
    let mut kvs: Vec<(String, MetadataValue)> = Vec::with_capacity(count);

    for _ in 0..count {
        let key: String = match u.arbitrary() {
            Ok(k) => k,
            Err(_) => break,
        };
        let arb: ArbMetadata = match u.arbitrary() {
            Ok(v) => v,
            Err(_) => break,
        };
        kvs.push((key, arb.into_metadata()));
    }

    // Exercise Display, Debug, and typed accessor methods on every value —
    // the goal is to surface panics or UB, not to assert correctness.
    for (key, val) in &kvs {
        let _ = key.len();
        let _ = format!("{}", val);
        let _ = format!("{:?}", val);

        let _ = val.as_str();
        let _ = val.as_u32();
        let _ = val.as_u64();
        let _ = val.as_f32();
        let _ = val.as_bool();
        let _ = val.as_array();

        if let MetadataValue::Array(arr) = val {
            for elem in arr {
                let _ = format!("{}", elem);
                let _ = elem.as_u32();
                let _ = elem.as_str();
            }
        }
    }
});
