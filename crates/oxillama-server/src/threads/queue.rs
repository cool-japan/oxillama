//! Unbounded MPSC work queue for the Assistants API run processor.
//!
//! Route handlers submit `RunWorkItem`s through the `RunQueueSender` when a
//! new run is created.  The `spawn_run_worker` background task drains
//! `RunQueueReceiver` and executes each run against the inference engine.

use tokio::sync::mpsc;

/// A single item submitted to the run worker queue.
#[derive(Debug, Clone)]
pub struct RunWorkItem {
    /// ID of the thread the run belongs to.
    pub thread_id: String,
    /// ID of the run to execute.
    pub run_id: String,
    /// Optional model override (empty means use server default).
    pub model: Option<String>,
    /// Optional system-level instruction prepended to the thread's messages.
    pub instructions: Option<String>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

/// Sender half of the run work queue.
///
/// Cloning this is cheap; all clones share the same underlying channel.
#[derive(Debug, Clone)]
pub struct RunQueueSender(pub mpsc::UnboundedSender<RunWorkItem>);

impl RunQueueSender {
    /// Submit a work item to the run queue.
    ///
    /// Returns an error if the receiving end has been dropped (worker exited).
    pub fn send(&self, item: RunWorkItem) -> Result<(), mpsc::error::SendError<RunWorkItem>> {
        self.0.send(item)
    }
}

/// Receiver half of the run work queue.
pub struct RunQueueReceiver(pub mpsc::UnboundedReceiver<RunWorkItem>);

impl RunQueueReceiver {
    /// Receive the next work item, waiting asynchronously.
    ///
    /// Returns `None` when all senders have been dropped.
    pub async fn recv(&mut self) -> Option<RunWorkItem> {
        self.0.recv().await
    }
}

/// Create an unbounded run work queue.
///
/// Returns `(sender, receiver)`.  The queue has no capacity bound; callers
/// are responsible for back-pressure at the API level (e.g. per-thread run
/// limits).
pub fn new_run_queue() -> (RunQueueSender, RunQueueReceiver) {
    let (tx, rx) = mpsc::unbounded_channel();
    (RunQueueSender(tx), RunQueueReceiver(rx))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queue_send_and_receive() {
        let (tx, mut rx) = new_run_queue();
        tx.send(RunWorkItem {
            thread_id: "thread_1".to_string(),
            run_id: "run_1".to_string(),
            model: None,
            instructions: None,
            max_tokens: 256,
        })
        .expect("send should succeed");

        let item = rx.recv().await.expect("recv should yield an item");
        assert_eq!(item.thread_id, "thread_1");
        assert_eq!(item.run_id, "run_1");
        assert_eq!(item.max_tokens, 256);
    }

    #[tokio::test]
    async fn queue_closed_when_sender_dropped() {
        let (tx, mut rx) = new_run_queue();
        drop(tx);
        assert!(
            rx.recv().await.is_none(),
            "closed channel should yield None"
        );
    }

    #[tokio::test]
    async fn queue_multiple_items_in_order() {
        let (tx, mut rx) = new_run_queue();

        for i in 0..5_u32 {
            tx.send(RunWorkItem {
                thread_id: format!("thread_{i}"),
                run_id: format!("run_{i}"),
                model: None,
                instructions: None,
                max_tokens: 128,
            })
            .expect("send");
        }

        for i in 0..5_u32 {
            let item = rx.recv().await.expect("recv");
            assert_eq!(item.thread_id, format!("thread_{i}"));
        }
    }

    #[tokio::test]
    async fn queue_sender_clone_shares_channel() {
        let (tx, mut rx) = new_run_queue();
        let tx2 = tx.clone();

        tx.send(RunWorkItem {
            thread_id: "t1".into(),
            run_id: "r1".into(),
            model: None,
            instructions: None,
            max_tokens: 64,
        })
        .expect("send from tx");

        tx2.send(RunWorkItem {
            thread_id: "t2".into(),
            run_id: "r2".into(),
            model: None,
            instructions: None,
            max_tokens: 64,
        })
        .expect("send from tx2");

        let a = rx.recv().await.expect("first item");
        let b = rx.recv().await.expect("second item");
        assert_eq!(a.thread_id, "t1");
        assert_eq!(b.thread_id, "t2");
    }
}
