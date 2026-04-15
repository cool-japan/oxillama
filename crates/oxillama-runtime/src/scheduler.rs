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

use std::collections::HashMap;

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
}

impl Sequence {
    /// Create a new waiting sequence.
    pub fn new(id: SeqId, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            state: SeqState::Waiting,
            prompt_tokens,
            output_tokens: Vec::new(),
            prompt_pos: 0,
            max_tokens,
            stopped: false,
            priority: id, // FIFO by default
        }
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
    pub fn append_token(&mut self, id: SeqId, token: u32) {
        if let Some(seq) = self.sequences.get_mut(&id) {
            seq.output_tokens.push(token);
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
}
