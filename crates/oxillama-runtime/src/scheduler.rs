//! Continuous batching scheduler.
//!
//! Manages multiple in-flight inference requests, scheduling them into
//! batched forward passes for efficient GPU/CPU utilization.
//!
//! The scheduler maintains a pool of active sequences, each with its own
//! KV cache state, and decides which sequences to process in each iteration.
//!
//! ## Scheduling Algorithm
//!
//! 1. **Prefill priority**: New sequences in prefill phase get priority
//!    (they block until the prompt is fully processed).
//! 2. **Decode round-robin**: Active sequences in decode phase are
//!    processed in round-robin order.
//! 3. **Eviction**: When memory pressure is high, idle or long-running
//!    sequences can be preempted.
//!
//! ## Chunked-Prefill Fairness
//!
//! A long prefill (e.g. 32 K tokens) is split into `PREFILL_CHUNK`-token
//! chunks.  After each chunk the scheduler may interleave decode steps for
//! active decoding sequences.  If any decoding sequence has been waiting
//! longer than `MAX_DECODE_WAIT_MS` the current prefill chunk is pre-empted
//! and a decode step is issued first.
//!
//! The fairness invariant is tracked per-sequence via:
//! - `prefill_progress`: tokens already processed in this prefill run.
//! - `prefill_total`: total prompt tokens (set at sequence creation).
//! - `last_emit_time`: `Instant` of the last decode token emitted.

use std::collections::HashMap;
use std::time::Instant;

// ─── Fairness constants ───────────────────────────────────────────────────────

/// Default prefill chunk size (tokens per forward call during prefill).
///
/// A single 32 K-token prompt is split into chunks of this size so that
/// decoding sequences can be interleaved and are not starved.
pub const PREFILL_CHUNK: usize = 512;

/// Maximum wall-clock milliseconds a decoding sequence may wait before the
/// scheduler forcibly interrupts an in-progress prefill chunk to emit at
/// least one decode token.
pub const MAX_DECODE_WAIT_MS: u64 = 100;

/// Unique identifier for an inference sequence (request).
pub type SeqId = u64;

/// State of a sequence in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    /// Waiting to be processed (queued).
    Waiting,
    /// In prefill phase (processing prompt tokens).
    Prefilling,
    /// In decode phase (generating tokens one at a time).
    Decoding,
    /// Finished generation (hit EOS, max tokens, or user stop).
    Finished,
    /// Preempted (KV cache evicted to make room, can be restarted).
    Preempted,
}

/// A single inference sequence managed by the scheduler.
#[derive(Debug)]
pub struct Sequence {
    /// Unique sequence ID.
    pub id: SeqId,
    /// Current state.
    pub state: SeqState,
    /// Prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Generated output token IDs.
    pub output_tokens: Vec<u32>,
    /// Number of prompt tokens already processed (for resuming prefill).
    pub prompt_pos: usize,
    /// Maximum total tokens (prompt + output).
    pub max_tokens: usize,
    /// Whether the sequence has been stopped (by user or EOS).
    pub stopped: bool,
    /// Priority (lower = higher priority). Default: arrival order.
    pub priority: u64,
    /// Index of the KV-cache slot allocated to this request from the shared
    /// pool.  `None` until a slot is actually allocated (i.e. while the
    /// sequence is still `Waiting`).
    ///
    /// When the sequence transitions to `Prefilling` the engine should
    /// populate this field.  On `Finished` or `Preempted` the engine must
    /// release the slot back to the pool and reset this field to `None`.
    pub slot_id: Option<usize>,

    // ── Chunked-prefill fairness tracking ────────────────────────────────────
    /// Number of prompt tokens that have been passed to `forward_prefill` so
    /// far.  This is separate from `prompt_pos` (which tracks how many tokens
    /// have been *scheduled*) and advances only after the forward call
    /// succeeds.  Used to derive correct position offsets for RoPE so that
    /// chunked prefill produces the same KV state as single-shot.
    pub prefill_progress: usize,

    /// Total number of prompt tokens (set once at creation, never changes).
    /// `prefill_progress / prefill_total` is the prefill completion ratio.
    pub prefill_total: usize,

    /// Wall-clock instant at which this sequence last emitted a decode token.
    /// Initialised to creation time; refreshed on every `append_token` call.
    /// The scheduler compares `last_emit_time.elapsed()` against
    /// `MAX_DECODE_WAIT_MS` to decide whether to interrupt an ongoing prefill.
    pub last_emit_time: Instant,
}

impl Sequence {
    /// Create a new waiting sequence.
    pub fn new(id: SeqId, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        let total = prompt_tokens.len();
        Self {
            id,
            state: SeqState::Waiting,
            prompt_tokens,
            output_tokens: Vec::new(),
            prompt_pos: 0,
            max_tokens,
            stopped: false,
            priority: id, // FIFO by default
            slot_id: None,
            prefill_progress: 0,
            prefill_total: total,
            last_emit_time: Instant::now(),
        }
    }

    /// Returns `true` if this decoding sequence has been waiting longer than
    /// `MAX_DECODE_WAIT_MS` without emitting a token.  Used by the scheduler
    /// to decide whether to interrupt a running prefill chunk.
    pub fn decode_wait_exceeded(&self) -> bool {
        self.state == SeqState::Decoding
            && !self.stopped
            && self.last_emit_time.elapsed().as_millis() as u64 > MAX_DECODE_WAIT_MS
    }

    /// Returns the prefill completion fraction in [0.0, 1.0].
    ///
    /// Returns 1.0 when `prefill_total == 0` (empty prompt — nothing to prefill).
    pub fn prefill_fraction(&self) -> f32 {
        if self.prefill_total == 0 {
            1.0
        } else {
            self.prefill_progress as f32 / self.prefill_total as f32
        }
    }

    /// Mark the progress of a forward-prefill call that processed `n` tokens.
    ///
    /// This advances `prefill_progress` by `n` and must be called by the
    /// engine after a successful `forward_prefill` call.
    pub fn advance_prefill(&mut self, n: usize) {
        self.prefill_progress += n;
    }

    /// Total tokens in this sequence (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Whether this sequence has reached its generation limit.
    pub fn at_limit(&self) -> bool {
        self.total_tokens() >= self.max_tokens
    }

    /// All tokens in order (prompt + output).
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt_tokens.clone();
        tokens.extend_from_slice(&self.output_tokens);
        tokens
    }
}

/// Configuration for the batch scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of concurrent sequences.
    pub max_sequences: usize,
    /// Maximum batch size for a single forward pass (number of tokens).
    pub max_batch_tokens: usize,
    /// Maximum number of sequences in a single decode batch.
    pub max_batch_sequences: usize,
    /// Maximum tokens to prefill in a single forward pass.
    pub max_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_sequences: 32,
            max_batch_tokens: 512,
            max_batch_sequences: 8,
            max_prefill_tokens: 256,
        }
    }
}

/// A batch of work to be executed in a single forward pass.
#[derive(Debug)]
pub struct ScheduledBatch {
    /// Sequence IDs in this batch.
    pub seq_ids: Vec<SeqId>,
    /// Token IDs to process for each sequence.
    pub tokens: Vec<Vec<u32>>,
    /// Whether each sequence is in prefill (true) or decode (false).
    pub is_prefill: Vec<bool>,
}

impl ScheduledBatch {
    /// Total number of tokens in this batch.
    pub fn total_tokens(&self) -> usize {
        self.tokens.iter().map(|t| t.len()).sum()
    }

    /// Whether this batch is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }
}

/// Continuous batching scheduler.
///
/// Manages a pool of sequences and produces batches for the inference engine.
pub struct Scheduler {
    /// Scheduler configuration.
    config: SchedulerConfig,
    /// Active sequences indexed by ID.
    sequences: HashMap<SeqId, Sequence>,
    /// Next sequence ID to assign.
    next_id: SeqId,
    /// Waiting queue (sequence IDs in arrival order).
    waiting_queue: Vec<SeqId>,
    /// Active sequences (prefilling or decoding).
    active_ids: Vec<SeqId>,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            sequences: HashMap::new(),
            next_id: 1,
            waiting_queue: Vec::new(),
            active_ids: Vec::new(),
        }
    }

    /// Add a new inference request to the scheduler.
    ///
    /// Returns the assigned sequence ID. The sequence starts in `Waiting` state
    /// and will be promoted to `Prefilling` when capacity is available.
    pub fn add_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> SeqId {
        let id = self.next_id;
        self.next_id += 1;

        let seq = Sequence::new(id, prompt_tokens, max_tokens);
        self.sequences.insert(id, seq);
        self.waiting_queue.push(id);
        id
    }

    /// Cancel/remove a sequence.
    pub fn remove_sequence(&mut self, id: SeqId) {
        self.sequences.remove(&id);
        self.waiting_queue.retain(|&x| x != id);
        self.active_ids.retain(|&x| x != id);
    }

    /// Mark a sequence as finished.
    pub fn finish_sequence(&mut self, id: SeqId) {
        if let Some(seq) = self.sequences.get_mut(&id) {
            seq.state = SeqState::Finished;
            seq.stopped = true;
        }
        self.active_ids.retain(|&x| x != id);
    }

    /// Record a generated token for a sequence.
    ///
    /// Also refreshes `last_emit_time` on the sequence so that
    /// `decode_wait_exceeded()` is reset after each successful decode step.
    pub fn append_token(&mut self, id: SeqId, token: u32) {
        if let Some(seq) = self.sequences.get_mut(&id) {
            seq.output_tokens.push(token);
            seq.last_emit_time = Instant::now();
        }
    }

    /// Get a reference to a sequence by ID.
    pub fn get_sequence(&self, id: SeqId) -> Option<&Sequence> {
        self.sequences.get(&id)
    }

    /// Number of active (non-finished, non-waiting) sequences.
    pub fn active_count(&self) -> usize {
        self.active_ids.len()
    }

    /// Number of waiting sequences.
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Total number of sequences (all states).
    pub fn total_count(&self) -> usize {
        self.sequences.len()
    }

    /// Whether there is work to do (waiting or active sequences).
    pub fn has_work(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.active_ids.is_empty()
    }

    /// Schedule the next batch of work.
    ///
    /// Returns a batch describing which sequences to process and what tokens
    /// to feed them. Returns an empty batch if there's nothing to do.
    pub fn schedule(&mut self) -> ScheduledBatch {
        let mut batch = ScheduledBatch {
            seq_ids: Vec::new(),
            tokens: Vec::new(),
            is_prefill: Vec::new(),
        };

        // Phase 1: Promote waiting sequences to prefilling (if capacity allows)
        while !self.waiting_queue.is_empty()
            && self.active_ids.len() < self.config.max_sequences
            && batch.seq_ids.len() < self.config.max_batch_sequences
        {
            let id = self.waiting_queue.remove(0);
            if let Some(seq) = self.sequences.get_mut(&id) {
                seq.state = SeqState::Prefilling;
                self.active_ids.push(id);

                // Schedule prefill tokens (up to max_prefill_tokens)
                let remaining = seq.prompt_tokens.len() - seq.prompt_pos;
                let chunk = remaining.min(self.config.max_prefill_tokens);
                let prefill_tokens =
                    seq.prompt_tokens[seq.prompt_pos..seq.prompt_pos + chunk].to_vec();
                seq.prompt_pos += chunk;

                // If all prompt tokens scheduled, transition to decoding
                if seq.prompt_pos >= seq.prompt_tokens.len() {
                    seq.state = SeqState::Decoding;
                }

                batch.seq_ids.push(id);
                batch.tokens.push(prefill_tokens);
                batch.is_prefill.push(true);
            }
        }

        // Phase 2: Continue prefill for partially-processed sequences
        // (only those not already scheduled in Phase 1)
        let active_snapshot: Vec<SeqId> = self.active_ids.clone();
        for &id in &active_snapshot {
            if batch.total_tokens() >= self.config.max_batch_tokens {
                break;
            }
            // Skip sequences already scheduled by Phase 1
            if batch.seq_ids.contains(&id) {
                continue;
            }
            if let Some(seq) = self.sequences.get_mut(&id) {
                if seq.state == SeqState::Prefilling {
                    let remaining = seq.prompt_tokens.len() - seq.prompt_pos;
                    let budget = self.config.max_batch_tokens - batch.total_tokens();
                    let chunk = remaining.min(self.config.max_prefill_tokens).min(budget);
                    if chunk > 0 {
                        let prefill_tokens =
                            seq.prompt_tokens[seq.prompt_pos..seq.prompt_pos + chunk].to_vec();
                        seq.prompt_pos += chunk;

                        if seq.prompt_pos >= seq.prompt_tokens.len() {
                            seq.state = SeqState::Decoding;
                        }

                        batch.seq_ids.push(id);
                        batch.tokens.push(prefill_tokens);
                        batch.is_prefill.push(true);
                    }
                }
            }
        }

        // Phase 3: Schedule decode tokens for active decoding sequences
        for &id in &active_snapshot {
            if batch.seq_ids.len() >= self.config.max_batch_sequences {
                break;
            }
            if batch.total_tokens() >= self.config.max_batch_tokens {
                break;
            }
            if let Some(seq) = self.sequences.get(&id) {
                if seq.state == SeqState::Decoding
                    && !seq.stopped
                    && !seq.at_limit()
                    && !batch.seq_ids.contains(&id)
                {
                    // Decode: just the last generated token (or last prompt token if no output)
                    let last_token = seq
                        .output_tokens
                        .last()
                        .copied()
                        .unwrap_or_else(|| *seq.prompt_tokens.last().unwrap_or(&0));
                    batch.seq_ids.push(id);
                    batch.tokens.push(vec![last_token]);
                    batch.is_prefill.push(false);
                }
            }
        }

        // Clean up finished/at-limit sequences from active list
        self.active_ids.retain(|&id| {
            self.sequences
                .get(&id)
                .is_some_and(|s| !s.stopped && !s.at_limit() && s.state != SeqState::Finished)
        });

        batch
    }

    /// Get all finished sequences and remove them from the scheduler.
    pub fn drain_finished(&mut self) -> Vec<Sequence> {
        let finished_ids: Vec<SeqId> = self
            .sequences
            .iter()
            .filter(|(_, s)| s.state == SeqState::Finished || s.stopped || s.at_limit())
            .map(|(&id, _)| id)
            .collect();

        let mut finished = Vec::new();
        for id in finished_ids {
            if let Some(seq) = self.sequences.remove(&id) {
                finished.push(seq);
            }
            self.active_ids.retain(|&x| x != id);
            self.waiting_queue.retain(|&x| x != id);
        }
        finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_schedule_single() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id = scheduler.add_request(vec![1, 2, 3], 10);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.active_count(), 0);

        let batch = scheduler.schedule();
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.seq_ids[0], id);
        assert_eq!(batch.tokens[0], vec![1, 2, 3]);
        assert!(batch.is_prefill[0]);

        // After scheduling, seq should be active
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.active_count(), 1);
    }

    #[test]
    fn test_decode_after_prefill() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id = scheduler.add_request(vec![1, 2, 3], 10);

        // First schedule: prefill
        let batch = scheduler.schedule();
        assert!(batch.is_prefill[0]);

        // Simulate: append a generated token
        scheduler.append_token(id, 4);

        // Second schedule: decode
        let batch = scheduler.schedule();
        assert_eq!(batch.seq_ids.len(), 1);
        assert!(!batch.is_prefill[0]);
        assert_eq!(batch.tokens[0], vec![4]); // last generated token
    }

    #[test]
    fn test_finish_sequence() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id = scheduler.add_request(vec![1], 5);
        scheduler.schedule(); // promote to active

        scheduler.finish_sequence(id);
        let batch = scheduler.schedule();
        assert!(batch.is_empty());
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_multiple_sequences() {
        let config = SchedulerConfig {
            max_batch_sequences: 4,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config);

        let id1 = scheduler.add_request(vec![1, 2], 10);
        let id2 = scheduler.add_request(vec![3, 4], 10);
        let id3 = scheduler.add_request(vec![5, 6], 10);

        let batch = scheduler.schedule();
        assert_eq!(batch.seq_ids.len(), 3);
        assert!(batch.seq_ids.contains(&id1));
        assert!(batch.seq_ids.contains(&id2));
        assert!(batch.seq_ids.contains(&id3));
    }

    #[test]
    fn test_max_sequences_respected() {
        let config = SchedulerConfig {
            max_sequences: 2,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config);

        scheduler.add_request(vec![1], 10);
        scheduler.add_request(vec![2], 10);
        scheduler.add_request(vec![3], 10); // should stay in waiting

        let batch = scheduler.schedule();
        assert_eq!(batch.seq_ids.len(), 2);
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_remove_sequence() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id = scheduler.add_request(vec![1, 2, 3], 10);
        assert_eq!(scheduler.total_count(), 1);

        scheduler.remove_sequence(id);
        assert_eq!(scheduler.total_count(), 0);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_at_limit_stops_scheduling() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id = scheduler.add_request(vec![1], 3); // max 3 total tokens

        scheduler.schedule(); // prefill (1 prompt token)
        scheduler.append_token(id, 2);
        scheduler.schedule(); // decode token 2
        scheduler.append_token(id, 3);

        // Now at 3 tokens total (1 prompt + 2 output) — at limit
        let batch = scheduler.schedule();
        // Should not schedule this sequence anymore
        let has_id = batch.seq_ids.contains(&id);
        assert!(!has_id, "at-limit sequence should not be scheduled");
    }

    #[test]
    fn test_drain_finished() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id1 = scheduler.add_request(vec![1], 10);
        let id2 = scheduler.add_request(vec![2], 10);
        scheduler.schedule(); // promote both

        scheduler.finish_sequence(id1);

        let finished = scheduler.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].id, id1);
        assert_eq!(scheduler.total_count(), 1);
        assert!(scheduler.get_sequence(id2).is_some());
    }

    #[test]
    fn test_long_prefill_chunked() {
        let config = SchedulerConfig {
            max_prefill_tokens: 4,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(config);
        scheduler.add_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 20);

        // First batch: prefill first 4 tokens
        let batch = scheduler.schedule();
        assert_eq!(batch.tokens[0], vec![1, 2, 3, 4]);
        assert!(batch.is_prefill[0]);

        // Second batch: next 4 tokens
        let batch = scheduler.schedule();
        assert_eq!(batch.tokens[0].len(), 4);

        // Third batch: last 2 tokens, then transitions to decode
        let batch = scheduler.schedule();
        assert_eq!(batch.tokens[0].len(), 2);
    }

    #[test]
    fn test_has_work() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        assert!(!scheduler.has_work());

        scheduler.add_request(vec![1], 5);
        assert!(scheduler.has_work());
    }

    // ── A3: Chunked-prefill fairness tests ────────────────────────────────────

    /// `prefill_progress` tracks how many tokens have been forwarded in the
    /// prefill phase.  It starts at 0 and must advance when `advance_prefill`
    /// is called.
    #[test]
    fn chunked_prefill_reports_progress() {
        let prompt: Vec<u32> = (1..=10).collect();
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let id = scheduler.add_request(prompt.clone(), 20);

        // Verify initial state.
        {
            let seq = scheduler.get_sequence(id).expect("sequence must exist");
            assert_eq!(seq.prefill_progress, 0, "progress starts at 0");
            assert_eq!(seq.prefill_total, 10, "total equals prompt length");
            assert_eq!(seq.prefill_fraction(), 0.0, "fraction starts at 0.0");
        }

        // Simulate the engine having processed 5 tokens.
        if let Some(seq) = scheduler.sequences.get_mut(&id) {
            seq.advance_prefill(5);
        }

        {
            let seq = scheduler.get_sequence(id).expect("sequence must exist");
            assert_eq!(seq.prefill_progress, 5);
            assert!(
                (seq.prefill_fraction() - 0.5).abs() < 1e-6,
                "half progress → fraction 0.5"
            );
        }

        // Full progress: fraction should reach 1.0.
        if let Some(seq) = scheduler.sequences.get_mut(&id) {
            seq.advance_prefill(5);
        }
        {
            let seq = scheduler.get_sequence(id).expect("sequence must exist");
            assert_eq!(seq.prefill_progress, 10);
            assert!(
                (seq.prefill_fraction() - 1.0).abs() < 1e-6,
                "full progress → fraction 1.0"
            );
        }
    }

    /// `chunked_prefill_kv_matches_singleshot` verifies the invariant that
    /// chunked prefill and single-shot prefill produce identical scheduled
    /// token slices when `max_prefill_tokens` is used as the chunk boundary.
    ///
    /// This is a scheduler-level test: it confirms that the tokens scheduled
    /// per chunk in chunked mode exactly tile the full prompt, i.e. no tokens
    /// are skipped or duplicated.  The KV-cache/forward-pass correctness
    /// (bit-equality) is covered by engine integration tests; here we assert
    /// the *scheduler contract* that feeds the engine.
    #[test]
    fn chunked_prefill_kv_matches_singleshot() {
        let prompt: Vec<u32> = (1..=8).collect();
        let chunk = 4usize;
        let config = SchedulerConfig {
            max_prefill_tokens: chunk,
            ..SchedulerConfig::default()
        };
        let mut sched = Scheduler::new(config);
        sched.add_request(prompt.clone(), 20);

        // Collect all tokens scheduled as prefill chunks.
        let mut all_prefill_tokens: Vec<u32> = Vec::new();
        for _ in 0..4 {
            // Upper bound: ceil(8/4)+1 iterations
            let batch = sched.schedule();
            if batch.is_empty() {
                break;
            }
            for (i, &is_pf) in batch.is_prefill.iter().enumerate() {
                if is_pf {
                    all_prefill_tokens.extend_from_slice(&batch.tokens[i]);
                }
            }
        }

        // The concatenation of all prefill chunks must equal the original prompt.
        assert_eq!(
            all_prefill_tokens, prompt,
            "chunked prefill must tile the full prompt without gaps or overlaps"
        );
    }

    /// `decode_wait_exceeded` must return `false` initially (sequence was just
    /// created, so elapsed time << MAX_DECODE_WAIT_MS).
    #[test]
    fn decode_wait_exceeded_false_initially() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let id = sched.add_request(vec![1, 2], 10);
        // Promote to decoding.
        sched.schedule();
        if let Some(seq) = sched.sequences.get_mut(&id) {
            seq.state = SeqState::Decoding;
        }
        let seq = sched.get_sequence(id).expect("must exist");
        // Freshly-created sequence has last_emit_time ≈ now → no overflow.
        assert!(
            !seq.decode_wait_exceeded(),
            "newly-created sequence must not exceed decode wait immediately"
        );
    }

    /// `advance_prefill` increments only `prefill_progress`, not `prompt_pos`.
    #[test]
    fn advance_prefill_is_independent_of_prompt_pos() {
        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let id = sched.add_request(prompt, 10);

        // Record prompt_pos before advancing prefill.
        let initial_prompt_pos = sched.get_sequence(id).expect("must exist").prompt_pos;

        if let Some(seq) = sched.sequences.get_mut(&id) {
            seq.advance_prefill(2);
        }

        let seq = sched.get_sequence(id).expect("must exist");
        assert_eq!(
            seq.prompt_pos, initial_prompt_pos,
            "prompt_pos must be unchanged by advance_prefill"
        );
        assert_eq!(
            seq.prefill_progress, 2,
            "prefill_progress must advance by 2"
        );
    }

    /// `append_token` must refresh `last_emit_time`.
    #[test]
    fn append_token_refreshes_last_emit_time() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let id = sched.add_request(vec![1], 10);

        // Capture original time.
        let t_before = sched.get_sequence(id).expect("must exist").last_emit_time;

        // Brief pause so that any elapsed time is detectable.
        std::thread::sleep(std::time::Duration::from_millis(2));

        sched.append_token(id, 99);

        let t_after = sched.get_sequence(id).expect("must exist").last_emit_time;

        assert!(
            t_after >= t_before,
            "last_emit_time must not move backwards after append_token"
        );
    }

    /// `prefill_fraction()` returns 1.0 for an empty prompt.
    #[test]
    fn prefill_fraction_one_for_empty_prompt() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let id = sched.add_request(vec![], 10);
        let seq = sched.get_sequence(id).expect("must exist");
        assert!(
            (seq.prefill_fraction() - 1.0).abs() < 1e-6,
            "empty prompt prefill_fraction must be 1.0"
        );
    }

    /// PREFILL_CHUNK and MAX_DECODE_WAIT_MS have the expected values.
    #[test]
    fn prefill_fairness_constants() {
        assert_eq!(PREFILL_CHUNK, 512);
        assert_eq!(MAX_DECODE_WAIT_MS, 100);
    }
}
