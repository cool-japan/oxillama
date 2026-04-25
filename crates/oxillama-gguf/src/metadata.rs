//! GGUF metadata key-value store with typed access.

#[cfg(not(feature = "std"))]
use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
type MetaMap = HashMap<String, MetadataValue>;
#[cfg(not(feature = "std"))]
type MetaMap = BTreeMap<String, MetadataValue>;

use crate::error::{GgufError, GgufResult};

/// A typed metadata value from the GGUF KV store.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// 8-bit unsigned integer.
    Uint8(u8),
    /// 8-bit signed integer.
    Int8(i8),
    /// 16-bit unsigned integer.
    Uint16(u16),
    /// 16-bit signed integer.
    Int16(i16),
    /// 32-bit unsigned integer.
    Uint32(u32),
    /// 32-bit signed integer.
    Int32(i32),
    /// 32-bit float.
    Float32(f32),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Array of metadata values (homogeneous type).
    Array(Vec<MetadataValue>),
    /// 64-bit unsigned integer.
    Uint64(u64),
    /// 64-bit signed integer.
    Int64(i64),
    /// 64-bit float.
    Float64(f64),
}

impl core::fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Uint8(v) => write!(f, "{v}"),
            Self::Int8(v) => write!(f, "{v}"),
            Self::Uint16(v) => write!(f, "{v}"),
            Self::Int16(v) => write!(f, "{v}"),
            Self::Uint32(v) => write!(f, "{v}"),
            Self::Int32(v) => write!(f, "{v}"),
            Self::Float32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "\"{v}\""),
            Self::Uint64(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::Array(v) => write!(f, "[{} elements]", v.len()),
        }
    }
}

impl MetadataValue {
    /// Try to extract as a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as a u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint8(v) => Some(u32::from(*v)),
            Self::Uint16(v) => Some(u32::from(*v)),
            _ => None,
        }
    }

    /// Try to extract as a u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Uint16(v) => Some(u64::from(*v)),
            Self::Uint8(v) => Some(u64::from(*v)),
            _ => None,
        }
    }

    /// Try to extract as an i32.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            Self::Int8(v) => Some(i32::from(*v)),
            Self::Int16(v) => Some(i32::from(*v)),
            _ => None,
        }
    }

    /// Try to extract as an f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as an array reference.
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// A key-value metadata store parsed from GGUF files.
///
/// Provides typed access to model configuration, architecture info,
/// tokenizer data, and other metadata embedded in GGUF files.
#[derive(Debug, Clone, Default)]
pub struct MetadataStore {
    entries: MetaMap,
}

impl MetadataStore {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            entries: MetaMap::new(),
        }
    }

    /// Insert a key-value pair into the store.
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        self.entries.insert(key, value);
    }

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Get a required string value, returning an error if missing or wrong type.
    pub fn get_string(&self, key: &str) -> GgufResult<&str> {
        self.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| GgufError::InvalidMetadata {
                key: key.to_string(),
                reason: "expected string value".to_string(),
            })
    }

    /// Get a required u32 value, returning an error if missing or wrong type.
    pub fn get_u32(&self, key: &str) -> GgufResult<u32> {
        self.get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::InvalidMetadata {
                key: key.to_string(),
                reason: "expected u32 value".to_string(),
            })
    }

    /// Get a required u64 value, returning an error if missing or wrong type.
    pub fn get_u64(&self, key: &str) -> GgufResult<u64> {
        self.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| GgufError::InvalidMetadata {
                key: key.to_string(),
                reason: "expected u64 value".to_string(),
            })
    }

    /// Get a required f32 value, returning an error if missing or wrong type.
    pub fn get_f32(&self, key: &str) -> GgufResult<f32> {
        self.get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| GgufError::InvalidMetadata {
                key: key.to_string(),
                reason: "expected f32 value".to_string(),
            })
    }

    /// Get a string value with a default fallback.
    pub fn get_string_or(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| default.to_string())
    }

    /// Returns the number of metadata entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }

    /// Returns all keys in the store.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(pairs: &[(&str, MetadataValue)]) -> MetadataStore {
        let mut store = MetadataStore::new();
        for (k, v) in pairs {
            store.insert(k.to_string(), v.clone());
        }
        store
    }

    // ── Basic insert / get ───────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let store = MetadataStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let store = MetadataStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut store = MetadataStore::new();
        store.insert("key".to_string(), MetadataValue::Uint32(42));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
        let val = store.get("key").expect("key should be present");
        assert!(matches!(val, MetadataValue::Uint32(42)));
    }

    #[test]
    fn test_get_missing_returns_none() {
        let store = MetadataStore::new();
        assert!(store.get("no_such_key").is_none());
    }

    #[test]
    fn test_insert_overwrites() {
        let mut store = MetadataStore::new();
        store.insert("k".to_string(), MetadataValue::Uint32(1));
        store.insert("k".to_string(), MetadataValue::Uint32(2));
        assert_eq!(store.len(), 1);
        let v = store.get("k").expect("key must exist");
        assert!(matches!(v, MetadataValue::Uint32(2)));
    }

    // ── get_string ───────────────────────────────────────────────────────────

    #[test]
    fn test_get_string_ok() {
        let store = make_store(&[("name", MetadataValue::String("llama".to_string()))]);
        let s = store.get_string("name").expect("should be a string");
        assert_eq!(s, "llama");
    }

    #[test]
    fn test_get_string_missing_returns_err() {
        let store = MetadataStore::new();
        assert!(store.get_string("missing").is_err());
    }

    #[test]
    fn test_get_string_wrong_type_returns_err() {
        let store = make_store(&[("n", MetadataValue::Uint32(5))]);
        assert!(store.get_string("n").is_err());
    }

    // ── get_u32 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_u32_from_uint32() {
        let store = make_store(&[("layers", MetadataValue::Uint32(32))]);
        assert_eq!(store.get_u32("layers").expect("u32"), 32);
    }

    #[test]
    fn test_get_u32_from_uint8() {
        let store = make_store(&[("n", MetadataValue::Uint8(7))]);
        assert_eq!(store.get_u32("n").expect("u8 widens to u32"), 7);
    }

    #[test]
    fn test_get_u32_from_uint16() {
        let store = make_store(&[("n", MetadataValue::Uint16(1000))]);
        assert_eq!(store.get_u32("n").expect("u16 widens to u32"), 1000);
    }

    #[test]
    fn test_get_u32_missing_returns_err() {
        let store = MetadataStore::new();
        assert!(store.get_u32("nope").is_err());
    }

    #[test]
    fn test_get_u32_wrong_type_returns_err() {
        let store = make_store(&[("x", MetadataValue::Float32(1.5))]);
        assert!(store.get_u32("x").is_err());
    }

    // ── get_u64 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_u64_from_uint64() {
        let store = make_store(&[("big", MetadataValue::Uint64(u64::MAX))]);
        assert_eq!(store.get_u64("big").expect("u64"), u64::MAX);
    }

    #[test]
    fn test_get_u64_from_uint32() {
        let store = make_store(&[("n", MetadataValue::Uint32(999))]);
        assert_eq!(store.get_u64("n").expect("u32 widens to u64"), 999);
    }

    #[test]
    fn test_get_u64_from_uint8() {
        let store = make_store(&[("n", MetadataValue::Uint8(3))]);
        assert_eq!(store.get_u64("n").expect("u8 widens to u64"), 3);
    }

    #[test]
    fn test_get_u64_missing_returns_err() {
        let store = MetadataStore::new();
        assert!(store.get_u64("nope").is_err());
    }

    // ── get_f32 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_f32_ok() {
        let store = make_store(&[("eps", MetadataValue::Float32(1e-5))]);
        let v = store.get_f32("eps").expect("f32");
        assert!((v - 1e-5_f32).abs() < 1e-10);
    }

    #[test]
    fn test_get_f32_missing_returns_err() {
        let store = MetadataStore::new();
        assert!(store.get_f32("x").is_err());
    }

    #[test]
    fn test_get_f32_wrong_type_returns_err() {
        let store = make_store(&[("x", MetadataValue::Uint32(1))]);
        assert!(store.get_f32("x").is_err());
    }

    // ── get_string_or ────────────────────────────────────────────────────────

    #[test]
    fn test_get_string_or_returns_value_when_present() {
        let store = make_store(&[("arch", MetadataValue::String("mistral".to_string()))]);
        assert_eq!(store.get_string_or("arch", "default"), "mistral");
    }

    #[test]
    fn test_get_string_or_returns_default_when_missing() {
        let store = MetadataStore::new();
        assert_eq!(store.get_string_or("arch", "fallback"), "fallback");
    }

    #[test]
    fn test_get_string_or_returns_default_when_wrong_type() {
        let store = make_store(&[("arch", MetadataValue::Uint32(1))]);
        assert_eq!(store.get_string_or("arch", "default"), "default");
    }

    // ── MetadataValue conversions ────────────────────────────────────────────

    #[test]
    fn test_metadata_value_as_str() {
        assert_eq!(MetadataValue::String("hi".to_string()).as_str(), Some("hi"));
        assert!(MetadataValue::Uint32(1).as_str().is_none());
    }

    #[test]
    fn test_metadata_value_as_u32_coercions() {
        assert_eq!(MetadataValue::Uint32(100).as_u32(), Some(100));
        assert_eq!(MetadataValue::Uint8(255).as_u32(), Some(255));
        assert_eq!(MetadataValue::Uint16(1000).as_u32(), Some(1000));
        assert!(MetadataValue::Int32(-1).as_u32().is_none());
        assert!(MetadataValue::Float32(1.0).as_u32().is_none());
    }

    #[test]
    fn test_metadata_value_as_u64_coercions() {
        assert_eq!(
            MetadataValue::Uint64(999_999_999_999).as_u64(),
            Some(999_999_999_999)
        );
        assert_eq!(MetadataValue::Uint32(42).as_u64(), Some(42));
        assert_eq!(MetadataValue::Uint16(65535).as_u64(), Some(65535));
        assert_eq!(MetadataValue::Uint8(128).as_u64(), Some(128));
        assert!(MetadataValue::Int64(-1).as_u64().is_none());
    }

    #[test]
    fn test_metadata_value_as_i32_coercions() {
        assert_eq!(MetadataValue::Int32(-42).as_i32(), Some(-42));
        assert_eq!(MetadataValue::Int8(-1).as_i32(), Some(-1));
        assert_eq!(MetadataValue::Int16(32767).as_i32(), Some(32767));
        assert!(MetadataValue::Uint32(1).as_i32().is_none());
    }

    #[test]
    fn test_metadata_value_as_f32() {
        assert_eq!(
            MetadataValue::Float32(std::f32::consts::PI).as_f32(),
            Some(std::f32::consts::PI)
        );
        assert!(MetadataValue::Float64(f64::from(std::f32::consts::PI))
            .as_f32()
            .is_none());
    }

    #[test]
    fn test_metadata_value_as_bool() {
        assert_eq!(MetadataValue::Bool(true).as_bool(), Some(true));
        assert_eq!(MetadataValue::Bool(false).as_bool(), Some(false));
        assert!(MetadataValue::Uint8(1).as_bool().is_none());
    }

    #[test]
    fn test_metadata_value_as_array() {
        let arr = MetadataValue::Array(vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2)]);
        let slice = arr.as_array().expect("should be array");
        assert_eq!(slice.len(), 2);
        assert!(MetadataValue::Uint32(1).as_array().is_none());
    }

    // ── iter / keys ──────────────────────────────────────────────────────────

    #[test]
    fn test_iter_returns_all_entries() {
        let store = make_store(&[
            ("a", MetadataValue::Uint32(1)),
            ("b", MetadataValue::Uint32(2)),
        ]);
        let mut keys: Vec<&str> = store.iter().map(|(k, _)| k.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_keys_returns_all_keys() {
        let store = make_store(&[
            ("x", MetadataValue::Bool(true)),
            ("y", MetadataValue::Bool(false)),
        ]);
        let mut keys: Vec<&str> = store.keys().map(|s| s.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["x", "y"]);
    }

    // ── Display ──────────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_value_display_string_has_quotes() {
        let v = MetadataValue::String("hello".to_string());
        let s = v.to_string();
        assert!(
            s.contains("hello"),
            "display should contain the string value"
        );
    }

    #[test]
    fn test_metadata_value_display_array_shows_length() {
        let v = MetadataValue::Array(vec![MetadataValue::Uint32(1); 5]);
        let s = v.to_string();
        assert!(s.contains('5'), "display should mention element count");
    }
}
