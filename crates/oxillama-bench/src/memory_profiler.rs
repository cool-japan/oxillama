//! Asynchronous RSS memory profiler with event recording.
//!
//! [`AsyncMemoryProfiler`] spawns a Tokio background task that samples the
//! current process's RSS at a configurable interval using the `sysinfo` crate
//! (Pure Rust, no FFI).  Call [`AsyncMemoryProfiler::record_event`] from any
//! thread to annotate the timeline with semantic events such as KV-cache
//! allocations, weight loads, and state changes.
//!
//! After the profiling window, call [`AsyncMemoryProfiler::stop`] to abort the
//! sampler and obtain a [`ProfileReport`] containing all samples and events.
//! The report can be serialised to JSON or summarised as a one-line table.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use tokio::task::JoinHandle;
use tokio::time;

// ── Data types ─────────────────────────────────────────────────────────────

/// A single RSS/virtual-memory sample recorded at an instant in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemSample {
    /// Milliseconds since the profiler was started.
    pub elapsed_ms: u64,
    /// Resident Set Size in kilobytes.
    pub rss_kb: u64,
    /// Virtual memory size in kilobytes.
    pub virtual_kb: u64,
}

/// Kinds of memory lifecycle events that can be recorded manually.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemEventKind {
    /// A KV-cache slab was allocated.
    KvCacheAlloc,
    /// A KV-cache slab was freed.
    KvCacheFree,
    /// Inference state (attention buffers, scratch) was allocated.
    StateAlloc,
    /// Inference state was freed.
    StateFree,
    /// Model weights finished loading from disk.
    WeightsLoaded,
}

/// A manually recorded memory event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemEvent {
    /// Milliseconds since the profiler was started.
    pub elapsed_ms: u64,
    /// Semantic event kind.
    pub kind: MemEventKind,
    /// Approximate number of bytes involved.
    pub bytes: u64,
}

// ── Shared state between the spawned task and the handle ──────────────────

struct ProfilerState {
    samples: Vec<MemSample>,
    events: Vec<MemEvent>,
}

impl ProfilerState {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            events: Vec::new(),
        }
    }
}

// ── AsyncMemoryProfiler ────────────────────────────────────────────────────

/// Async RSS memory profiler.
///
/// Spawns a Tokio background task that polls the current process's RSS at
/// `interval_ms` intervals.  Semantic memory events can be appended at any
/// time via [`record_event`](Self::record_event).
///
/// # Note on naming
/// This type is distinct from [`crate::MemoryProfiler`] (a synchronous
/// rolling-window sampler in `memory.rs`).  `AsyncMemoryProfiler` is the
/// tokio-backed, event-annotated variant required by E1.
pub struct AsyncMemoryProfiler {
    state: Arc<Mutex<ProfilerState>>,
    start: Instant,
    handle: Option<JoinHandle<()>>,
}

impl AsyncMemoryProfiler {
    /// Start sampling RSS at `interval_ms` millisecond intervals.
    ///
    /// Spawns a Tokio task; the caller must be inside a Tokio runtime.
    pub fn start(interval_ms: u64) -> Self {
        let state = Arc::new(Mutex::new(ProfilerState::new()));
        let start = Instant::now();

        let state_clone = Arc::clone(&state);
        let start_clone = start;
        let interval = Duration::from_millis(interval_ms.max(1));

        let handle = tokio::task::spawn(async move {
            // Resolve our own PID once; this never changes for the process.
            let current_pid = std::process::id();
            let pid = Pid::from_u32(current_pid);

            let mut sys = System::new();
            let mut ticker = time::interval(interval);

            loop {
                ticker.tick().await;

                // Refresh only our own process's memory fields.
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    true,
                    ProcessRefreshKind::nothing().with_memory(),
                );

                if let Some(proc) = sys.process(pid) {
                    let rss_bytes = proc.memory();
                    let virt_bytes = proc.virtual_memory();

                    // sysinfo ≥ 0.31: memory() returns bytes.
                    // We divide by 1024 to report KB.
                    let sample = MemSample {
                        elapsed_ms: start_clone.elapsed().as_millis() as u64,
                        rss_kb: rss_bytes / 1024,
                        virtual_kb: virt_bytes / 1024,
                    };

                    if let Ok(mut guard) = state_clone.lock() {
                        guard.samples.push(sample);
                    }
                }
            }
        });

        Self {
            state,
            start,
            handle: Some(handle),
        }
    }

    /// Record a semantic memory lifecycle event.
    ///
    /// Thread-safe; may be called from any thread or async task.
    pub fn record_event(&self, kind: MemEventKind, bytes: u64) {
        let event = MemEvent {
            elapsed_ms: self.start.elapsed().as_millis() as u64,
            kind,
            bytes,
        };
        if let Ok(mut guard) = self.state.lock() {
            guard.events.push(event);
        }
    }

    /// Stop the profiler and collect the recorded data.
    ///
    /// Aborts the background sampler task and returns a [`ProfileReport`].
    pub fn stop(mut self) -> ProfileReport {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
        let duration_ms = self.start.elapsed().as_millis() as u64;

        let (samples, events) = match self.state.lock() {
            Ok(guard) => (guard.samples.clone(), guard.events.clone()),
            Err(_) => (Vec::new(), Vec::new()),
        };

        ProfileReport {
            samples,
            events,
            duration_ms,
        }
    }
}

// ── ProfileReport ──────────────────────────────────────────────────────────

/// Output of a completed profiling session.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Time-ordered RSS samples collected during the session.
    pub samples: Vec<MemSample>,
    /// Time-ordered memory events recorded manually.
    pub events: Vec<MemEvent>,
    /// Total duration of the profiling session in milliseconds.
    pub duration_ms: u64,
}

impl ProfileReport {
    /// Peak RSS across all samples, in kilobytes.
    ///
    /// Returns `0` when no samples were collected.
    pub fn peak_rss_kb(&self) -> u64 {
        self.samples.iter().map(|s| s.rss_kb).max().unwrap_or(0)
    }

    /// Minimum RSS across all samples, in kilobytes.
    ///
    /// Returns `0` when no samples were collected.
    pub fn min_rss_kb(&self) -> u64 {
        self.samples.iter().map(|s| s.rss_kb).min().unwrap_or(0)
    }

    /// Mean RSS across all samples, in kilobytes.
    ///
    /// Returns `0.0` when no samples were collected.
    pub fn mean_rss_kb(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.samples.iter().map(|s| s.rss_kb).sum();
        sum as f64 / self.samples.len() as f64
    }

    /// Serialise the report to a pretty-printed JSON file.
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        let mut f = std::fs::File::create(path)?;
        f.write_all(json.as_bytes())
    }

    /// Print a compact summary to stdout.
    pub fn print_summary(&self) {
        println!("=== AsyncMemoryProfiler Report ===");
        println!("  Duration:    {} ms", self.duration_ms);
        println!("  Samples:     {}", self.samples.len());
        println!("  Events:      {}", self.events.len());
        println!("  Peak RSS:    {} KB", self.peak_rss_kb());
        println!("  Mean RSS:    {:.0} KB", self.mean_rss_kb());

        if !self.events.is_empty() {
            use tabled::{Table, Tabled};

            #[derive(Tabled)]
            struct EventRow {
                #[tabled(rename = "ms")]
                elapsed_ms: u64,
                #[tabled(rename = "kind")]
                kind: String,
                #[tabled(rename = "bytes")]
                bytes: u64,
            }

            let rows: Vec<EventRow> = self
                .events
                .iter()
                .map(|e| EventRow {
                    elapsed_ms: e.elapsed_ms,
                    kind: format!("{:?}", e.kind),
                    bytes: e.bytes,
                })
                .collect();

            println!("\n  Events table:");
            println!("{}", Table::new(rows));
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    /// Start the profiler, allocate a large Vec, wait a bit, stop and verify
    /// that at least one RSS sample was recorded with a non-zero peak.
    #[tokio::test]
    async fn memory_profiler_captures_rss() {
        let profiler = AsyncMemoryProfiler::start(10); // 10ms interval
                                                       // 10 MiB allocation to perturb RSS
        let _big = vec![0u8; 10 * 1024 * 1024];
        tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;
        let report = profiler.stop();
        assert!(
            !report.samples.is_empty(),
            "expected at least one sample after 60ms with 10ms interval"
        );
        assert!(
            report.peak_rss_kb() > 0,
            "peak RSS must be positive (got 0)"
        );
    }

    /// Write the report as JSON and parse it back; the `samples` array must be
    /// non-empty and the top-level `duration_ms` field must be present.
    #[tokio::test]
    async fn memory_profiler_json_output() {
        let profiler = AsyncMemoryProfiler::start(10);
        tokio::time::sleep(tokio::time::Duration::from_millis(40)).await;
        let report = profiler.stop();

        let path = temp_dir().join("test_memory_profile.json");
        report.write_json(&path).expect("write_json must not fail");

        let content = std::fs::read_to_string(&path).expect("read json back");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");

        assert!(
            parsed["samples"].is_array(),
            "JSON must contain a 'samples' array"
        );
        assert!(
            parsed.get("duration_ms").is_some(),
            "JSON must contain 'duration_ms'"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Record exactly 3 events and verify the report contains all 3.
    #[tokio::test]
    async fn memory_event_recording() {
        let profiler = AsyncMemoryProfiler::start(5000); // very slow sampling
        profiler.record_event(MemEventKind::KvCacheAlloc, 1024);
        profiler.record_event(MemEventKind::KvCacheAlloc, 2048);
        profiler.record_event(MemEventKind::KvCacheFree, 1024);
        let report = profiler.stop();
        assert_eq!(report.events.len(), 3, "expected exactly 3 recorded events");
        assert_eq!(report.events[0].kind, MemEventKind::KvCacheAlloc);
        assert_eq!(report.events[1].kind, MemEventKind::KvCacheAlloc);
        assert_eq!(report.events[2].kind, MemEventKind::KvCacheFree);
    }

    /// Verify that `write_json` + `from_str` round-trips the event kinds.
    #[tokio::test]
    async fn memory_profiler_event_roundtrip() {
        let profiler = AsyncMemoryProfiler::start(5000);
        profiler.record_event(MemEventKind::WeightsLoaded, 4_000_000_000);
        profiler.record_event(MemEventKind::StateAlloc, 64 * 1024);
        let report = profiler.stop();

        let path = temp_dir().join("test_memory_events_roundtrip.json");
        report.write_json(&path).expect("write_json");
        let content = std::fs::read_to_string(&path).expect("read back");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("parse json");

        let events = parsed["events"].as_array().expect("events array");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["kind"], "weights_loaded");
        assert_eq!(events[1]["kind"], "state_alloc");
        std::fs::remove_file(&path).ok();
    }
}
