//! CPU/disk offload with a pinned hot-layer set.
//!
//! This module provides the infrastructure for offloading model weights to disk
//! and loading them on-demand with an LRU eviction policy. A pinned hot-set
//! keeps embeddings, the output head, and the last N attention layers always
//! resident in RAM.
//!
//! # Overview
//!
//! - [`OffloadPolicy`] — declarative configuration: none, budget, or pinned hot-set.
//! - [`LayerPager`] — LRU weight pager with eviction, pinned tensors, and on-demand
//!   loads from a [`PagerSource`].
//! - [`MemoryPressureProbe`] — lightweight OS-level pressure monitor (Linux / macOS).

pub mod pager;
pub mod policy;
pub mod pressure;

pub use pager::{FilePagerSource, LayerPager, PagerSource, ResidentTensor, TensorEntry, TensorId};
pub use policy::OffloadPolicy;
pub use pressure::MemoryPressureProbe;
