//! `LoraStack` — ordered composition of multiple LoRA adapters.

use std::sync::Arc;

use super::adapter::{LoraAdapterTrait, TargetModule};
use super::loader::LoadedLora;
use crate::error::ArchResult;

/// A stack of LoRA adapters applied in order.
///
/// Adapters are applied additively:
/// `output += scale_0 · LoRA_0(x) + scale_1 · LoRA_1(x) + ...`
///
/// Each adapter's intrinsic `alpha/rank` scale is multiplied by the per-entry `stack_scale`.
///
/// The stack supports two independent adapter lists:
/// - **`entries`**: legacy [`LoadedLora`] adapters (GGUF-loaded, keyed by tensor name).
/// - **`adapter_list`**: trait-object adapters implementing [`LoraAdapterTrait`].
///
/// Both lists are consulted by the respective dispatch methods.
#[derive(Default)]
pub struct LoraStack {
    /// Ordered list of `(GGUF-loaded adapter set, per-entry scale multiplier)`.
    entries: Vec<(Arc<LoadedLora>, f32)>,
    /// Ordered list of trait-object adapters (new public API).
    adapter_list: Vec<Arc<dyn LoraAdapterTrait>>,
}

impl std::fmt::Debug for LoraStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraStack")
            .field("entries_len", &self.entries.len())
            .field("adapter_list_len", &self.adapter_list.len())
            .finish()
    }
}

impl Clone for LoraStack {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            // Arc clones are cheap reference-count bumps.
            adapter_list: self.adapter_list.clone(),
        }
    }
}

impl LoraStack {
    /// Create an empty stack.
    pub fn new() -> Self {
        Self::default()
    }

    // ─── Legacy `LoadedLora` API ──────────────────────────────────────────────

    /// Push a new GGUF-loaded adapter onto the stack with a given scale multiplier.
    ///
    /// `scale` multiplies the adapter's intrinsic `alpha/rank` scale factor.
    pub fn push(&mut self, adapter: Arc<LoadedLora>, scale: f32) {
        self.entries.push((adapter, scale));
    }

    /// Remove the last GGUF-loaded adapter from the stack.
    ///
    /// Returns `None` if the entries list is empty.
    pub fn pop(&mut self) -> Option<(Arc<LoadedLora>, f32)> {
        self.entries.pop()
    }

    /// Remove all adapters (both lists).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.adapter_list.clear();
    }

    /// Number of stacked legacy adapters.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no adapters are stacked in either list.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.adapter_list.is_empty()
    }

    /// Immutable view of the `(LoadedLora, scale)` entries.
    pub fn entries(&self) -> &[(Arc<LoadedLora>, f32)] {
        &self.entries
    }

    /// Immutable view of the trait-object adapter list.
    pub fn adapters(&self) -> &[Arc<dyn LoraAdapterTrait>] {
        &self.adapter_list
    }

    /// Apply the entire legacy stack to compute the additive LoRA delta for
    /// `tensor_name`.
    ///
    /// Returns `Ok(delta)` where `delta` is a `Vec<f32>` of length
    /// `out_features` representing `Σ_i stack_scale_i · (alpha_i/rank_i) · B_i @ A_i @ input`.
    /// Adapters that do not define `tensor_name` are silently skipped.
    /// The caller is responsible for adding `delta` to the base linear output.
    ///
    /// # Errors
    /// Returns [`crate::error::ArchError::Quant`] if a matching adapter's dimension checks fail.
    pub fn apply(
        &self,
        tensor_name: &str,
        input: &[f32],
        out_features: usize,
    ) -> ArchResult<Vec<f32>> {
        let mut delta = vec![0.0f32; out_features];
        for (lora, stack_scale) in &self.entries {
            let Some(adapter) = lora.get(tensor_name) else {
                continue;
            };
            let rank = adapter.rank;
            let in_f = adapter.in_features;
            let out_f = adapter.out_features.min(out_features);

            // Step 1: r = A @ input  (length = rank)
            let mut r_vec = vec![0.0f32; rank];
            for (i, r) in r_vec.iter_mut().enumerate() {
                let row = &adapter.a[i * in_f..(i + 1) * in_f];
                *r = row
                    .iter()
                    .zip(input.iter().take(in_f))
                    .map(|(&a, &x)| a * x)
                    .sum();
            }

            // Step 2: delta += B @ r * (intrinsic_scale * stack_scale)
            let combined = adapter.scale * stack_scale;
            for (i, d) in delta.iter_mut().enumerate().take(out_f) {
                let row = &adapter.b[i * rank..(i + 1) * rank];
                let v: f32 = row.iter().zip(r_vec.iter()).map(|(&b, &r)| b * r).sum();
                *d += v * combined;
            }
        }
        Ok(delta)
    }

    // ─── New trait-object adapter API ─────────────────────────────────────────

    /// Push a trait-object adapter onto the new adapter list.
    pub fn push_adapter(&mut self, adapter: Arc<dyn LoraAdapterTrait>) {
        self.adapter_list.push(adapter);
    }

    /// Compute the combined delta for `(target, layer)` across all
    /// trait-object adapters, applied to input vector `input`.
    ///
    /// Returns `None` if no adapter in the list covers this `(target, layer)`.
    ///
    /// Formula: `Δ_total = Σᵢ (αᵢ / rᵢ) * Bᵢ @ Aᵢ @ input`.
    pub fn applied_delta(
        &self,
        target: TargetModule,
        layer: usize,
        input: &[f32],
    ) -> Option<Vec<f32>> {
        let mut result: Option<Vec<f32>> = None;
        for adapter in &self.adapter_list {
            let scale = adapter.alpha() / adapter.rank().max(1) as f32;
            if let Some(delta) = adapter.delta(target, layer) {
                let contribution = delta.apply(input, scale);
                match result {
                    None => result = Some(contribution),
                    Some(ref mut acc) => {
                        for (a, c) in acc.iter_mut().zip(contribution.iter()) {
                            *a += c;
                        }
                    }
                }
            }
        }
        result
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::adapter::{LoraAdapterTrait, LoraDelta, TargetModule};
    use std::collections::HashMap;
    use std::sync::Arc;

    // ─── Helper: simple in-memory LoraAdapterTrait impl ──────────────────────

    /// A simple test adapter backed by a map of `(TargetModule, layer) → LoraDelta`.
    struct TestAdapter {
        rank: usize,
        alpha: f32,
        deltas: HashMap<(u32, usize), LoraDelta>,
        modules: Vec<TargetModule>,
    }

    impl TestAdapter {
        fn new(rank: usize, alpha: f32) -> Self {
            Self {
                rank,
                alpha,
                deltas: HashMap::new(),
                modules: Vec::new(),
            }
        }

        fn add_delta(&mut self, target: TargetModule, layer: usize, delta: LoraDelta) {
            let key = (target_to_u32(target), layer);
            if !self.modules.contains(&target) {
                self.modules.push(target);
            }
            self.deltas.insert(key, delta);
        }
    }

    fn target_to_u32(t: TargetModule) -> u32 {
        match t {
            TargetModule::QueryProj => 0,
            TargetModule::KeyProj => 1,
            TargetModule::ValueProj => 2,
            TargetModule::OutputProj => 3,
            TargetModule::GateProj => 4,
            TargetModule::UpProj => 5,
            TargetModule::DownProj => 6,
            TargetModule::Custom(id) => 100 + id,
        }
    }

    impl LoraAdapterTrait for TestAdapter {
        fn rank(&self) -> usize {
            self.rank
        }
        fn alpha(&self) -> f32 {
            self.alpha
        }
        fn target_modules(&self) -> &[TargetModule] {
            &self.modules
        }
        fn delta(&self, target: TargetModule, layer: usize) -> Option<&LoraDelta> {
            let key = (target_to_u32(target), layer);
            self.deltas.get(&key)
        }
    }

    // ─── Tests ────────────────────────────────────────────────────────────────

    /// Empty adapter list → applied_delta returns None.
    #[test]
    fn empty_stack_applied_delta_returns_none() {
        let stack = LoraStack::new();
        let result = stack.applied_delta(TargetModule::QueryProj, 0, &[1.0f32, 2.0, 3.0]);
        assert!(result.is_none(), "empty stack must return None");
    }

    /// Single adapter with an identity delta: result matches input.
    #[test]
    fn single_lora_identity_matches_input() {
        let rank = 4;
        let in_dim = 4;
        let out_dim = 4;

        // Build identity A and B.
        let mut a = vec![0.0f32; rank * in_dim];
        let mut b = vec![0.0f32; out_dim * rank];
        for i in 0..rank {
            a[i * in_dim + i] = 1.0;
            b[i * rank + i] = 1.0;
        }
        let delta = LoraDelta::new(a, b, rank, in_dim, out_dim);
        let alpha = rank as f32;

        let mut adapter = TestAdapter::new(rank, alpha);
        adapter.add_delta(TargetModule::QueryProj, 0, delta);

        let mut stack = LoraStack::new();
        stack.push_adapter(Arc::new(adapter));

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = stack
            .applied_delta(TargetModule::QueryProj, 0, &x)
            .expect("single adapter must produce a result");

        // scale = alpha/rank = 1.0, identity delta passes x through.
        for (r, xi) in result.iter().zip(x.iter()) {
            assert!((r - xi).abs() < 1e-5, "expected {xi} got {r}");
        }
    }

    /// Two adapters with the same delta compose additively.
    #[test]
    fn two_loras_compose_additively() {
        let rank = 2;
        let in_dim = 4;
        let out_dim = 4;
        let alpha = 2.0f32; // scale = alpha/rank = 1.0

        // A=[1,0,0,0; 0,1,0,0], B=[1,0; 0,1; 0,0; 0,0]
        let a = vec![
            1.0f32, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
        ];
        let b = vec![
            1.0f32, 0.0, // row 0
            0.0, 1.0, // row 1
            0.0, 0.0, // row 2
            0.0, 0.0, // row 3
        ];

        let make_delta = || LoraDelta::new(a.clone(), b.clone(), rank, in_dim, out_dim);

        let mut adapter1 = TestAdapter::new(rank, alpha);
        adapter1.add_delta(TargetModule::QueryProj, 0, make_delta());

        let mut adapter2 = TestAdapter::new(rank, alpha);
        adapter2.add_delta(TargetModule::QueryProj, 0, make_delta());

        let mut stack = LoraStack::new();
        stack.push_adapter(Arc::new(adapter1));
        stack.push_adapter(Arc::new(adapter2));

        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let combined = stack
            .applied_delta(TargetModule::QueryProj, 0, &x)
            .expect("two adapters must produce a result");

        // Compute single-adapter expected, multiply by 2.
        let single = LoraDelta::new(a.clone(), b.clone(), rank, in_dim, out_dim)
            .apply(&x, alpha / rank as f32);
        for (c, s) in combined.iter().zip(single.iter()) {
            let expected = s * 2.0;
            assert!(
                (c - expected).abs() < 1e-5,
                "combined={c} expected={expected}"
            );
        }
    }

    /// Adapter that doesn't cover a target/layer returns None for that slot.
    #[test]
    fn adapter_not_covering_target_is_skipped() {
        let mut adapter = TestAdapter::new(2, 2.0);
        // only covers KeyProj layer 0
        adapter.add_delta(
            TargetModule::KeyProj,
            0,
            LoraDelta::new(vec![1.0; 4], vec![1.0; 4], 2, 2, 2),
        );

        let mut stack = LoraStack::new();
        stack.push_adapter(Arc::new(adapter));

        // Ask for QueryProj — not covered.
        let result = stack.applied_delta(TargetModule::QueryProj, 0, &[1.0f32, 1.0]);
        assert!(result.is_none(), "uncovered target must return None");
    }

    /// `with_lora_stack` persistence test: set stack, retrieve entries count.
    #[test]
    fn lora_stack_stores_adapters() {
        let mut stack = LoraStack::new();
        let mut a1 = TestAdapter::new(4, 4.0);
        a1.add_delta(
            TargetModule::ValueProj,
            0,
            LoraDelta::new(vec![0.0; 16], vec![0.0; 16], 4, 4, 4),
        );
        stack.push_adapter(Arc::new(a1));
        assert_eq!(stack.adapter_list.len(), 1, "one adapter pushed");

        let mut a2 = TestAdapter::new(8, 8.0);
        a2.add_delta(
            TargetModule::ValueProj,
            1,
            LoraDelta::new(vec![0.0; 64], vec![0.0; 64], 8, 8, 8),
        );
        stack.push_adapter(Arc::new(a2));
        assert_eq!(stack.adapter_list.len(), 2, "two adapters pushed");
    }

    // ─── Legacy apply() tests (preserved from mod.rs) ────────────────────────

    fn make_loaded_lora(
        name: &str,
        in_f: usize,
        out_f: usize,
        rank: usize,
        fill: f32,
    ) -> Arc<LoadedLora> {
        use oxillama_quant::LoraAdapter;
        let scale = 1.0_f32;
        let adapter = Arc::new(
            LoraAdapter::new(
                vec![fill; rank * in_f],
                vec![fill; out_f * rank],
                rank,
                scale,
                in_f,
                out_f,
            )
            .expect("valid adapter"),
        );
        let mut adapters = std::collections::HashMap::new();
        adapters.insert(name.to_string(), adapter);
        Arc::new(LoadedLora {
            adapters,
            rank,
            alpha: rank as f32,
        })
    }

    #[test]
    fn empty_legacy_stack_returns_zeros() {
        let stack = LoraStack::new();
        let result = stack
            .apply("blk.0.attn_q.weight", &[1.0, 2.0, 3.0, 4.0], 4)
            .expect("apply ok");
        assert_eq!(result, vec![0.0f32; 4]);
    }

    #[test]
    fn legacy_stacked_adapters_add_linearly() {
        let in_f = 4;
        let out_f = 4;
        let rank = 2;
        let lora = make_loaded_lora("blk.0.attn_q.weight", in_f, out_f, rank, 0.5);

        let mut stack_double = LoraStack::new();
        stack_double.push(Arc::clone(&lora), 0.5);
        stack_double.push(Arc::clone(&lora), 0.5);

        let mut stack_single = LoraStack::new();
        stack_single.push(Arc::clone(&lora), 1.0);

        let input = vec![1.0f32; in_f];
        let double = stack_double
            .apply("blk.0.attn_q.weight", &input, out_f)
            .expect("apply ok");
        let single = stack_single
            .apply("blk.0.attn_q.weight", &input, out_f)
            .expect("apply ok");

        for (a, b) in double.iter().zip(single.iter()) {
            assert!((a - b).abs() < 1e-5, "double={a} single={b}");
        }
    }
}
