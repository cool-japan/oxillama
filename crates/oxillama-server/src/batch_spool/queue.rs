//! Tokio mpsc-based work queue for the batch processing pipeline.
//!
//! The `BatchQueue` provides a bounded channel through which route handlers
//! submit job IDs to the background `BatchWorker`.  The worker receives job
//! IDs, reads the input from disk, processes each line through the inference
//! engine, and writes results back to disk.

use tokio::sync::mpsc;

/// A single work item: a job ID to process.
#[derive(Debug, Clone)]
pub struct BatchWorkItem {
    /// The batch job ID (subdirectory name under the spool dir).
    pub job_id: String,
}

/// Sender half of the batch work queue.
pub type BatchQueueSender = mpsc::Sender<BatchWorkItem>;

/// Receiver half of the batch work queue.
pub type BatchQueueReceiver = mpsc::Receiver<BatchWorkItem>;

/// Create a bounded batch work queue with the given capacity.
pub fn new_batch_queue(capacity: usize) -> (BatchQueueSender, BatchQueueReceiver) {
    mpsc::channel(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queue_send_and_receive() {
        let (tx, mut rx) = new_batch_queue(8);
        tx.send(BatchWorkItem {
            job_id: "batch_123".to_string(),
        })
        .await
        .expect("send should succeed");

        let item = rx.recv().await.expect("recv should yield an item");
        assert_eq!(item.job_id, "batch_123");
    }

    #[tokio::test]
    async fn queue_closed_when_sender_dropped() {
        let (tx, mut rx) = new_batch_queue(4);
        drop(tx);
        assert!(
            rx.recv().await.is_none(),
            "closed channel should yield None"
        );
    }
}
