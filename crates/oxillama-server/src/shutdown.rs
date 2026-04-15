//! Graceful shutdown handler.
//!
//! Installs signal handlers for SIGTERM and Ctrl-C (SIGINT), returning a
//! future that resolves when either signal is received.  The server's
//! `axum::serve(..).with_graceful_shutdown(shutdown_signal())` pattern
//! ensures in-flight requests are drained before the process exits.

use tokio::signal;
use tracing::info;

/// Wait for a shutdown signal (Ctrl-C / SIGTERM).
///
/// Resolves once when the first signal arrives. The returned future is
/// designed to be passed to `axum::serve(..).with_graceful_shutdown()`.
///
/// On Unix, both SIGINT (Ctrl-C) and SIGTERM are handled.
/// On non-Unix (Windows), only Ctrl-C is available.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("received Ctrl-C, initiating graceful shutdown");
        }
        () = terminate => {
            info!("received SIGTERM, initiating graceful shutdown");
        }
    }
}

/// Create a shutdown signal that can be triggered programmatically.
///
/// Returns a `(trigger, signal)` pair. Calling `trigger.shutdown()` resolves
/// the signal future. Useful for testing.
pub struct ShutdownTrigger {
    sender: tokio::sync::watch::Sender<bool>,
}

impl ShutdownTrigger {
    /// Create a new trigger/signal pair.
    pub fn new() -> (Self, ShutdownSignal) {
        let (sender, receiver) = tokio::sync::watch::channel(false);
        (Self { sender }, ShutdownSignal { receiver })
    }

    /// Trigger the shutdown.
    pub fn shutdown(self) {
        let _ = self.sender.send(true);
    }
}

impl Default for ShutdownTrigger {
    fn default() -> Self {
        Self::new().0
    }
}

/// A future that resolves when the associated [`ShutdownTrigger`] fires.
#[derive(Clone)]
pub struct ShutdownSignal {
    receiver: tokio::sync::watch::Receiver<bool>,
}

impl ShutdownSignal {
    /// Wait for the shutdown signal.
    pub async fn wait(&mut self) {
        while !*self.receiver.borrow() {
            if self.receiver.changed().await.is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trigger_fires_signal() {
        let (trigger, mut signal) = ShutdownTrigger::new();
        let handle = tokio::spawn(async move {
            signal.wait().await;
            true
        });
        trigger.shutdown();
        let result = handle.await.unwrap();
        assert!(result, "signal should resolve after trigger fires");
    }

    #[tokio::test]
    async fn test_signal_is_clone() {
        let (trigger, signal) = ShutdownTrigger::new();
        let mut s1 = signal.clone();
        let mut s2 = signal;
        let h1 = tokio::spawn(async move {
            s1.wait().await;
            true
        });
        let h2 = tokio::spawn(async move {
            s2.wait().await;
            true
        });
        trigger.shutdown();
        assert!(h1.await.unwrap());
        assert!(h2.await.unwrap());
    }
}
