//! Disk-spooled batch job storage.
//!
//! Each job lives in a subdirectory under the configured spool dir:
//!
//! ```text
//! <spool_dir>/
//!   <job_id>/
//!     input.jsonl     — the submitted request lines (immutable after creation)
//!     status.json     — job status (written atomically via tempfile)
//!     output.jsonl    — completed response lines (append-only)
//!     errors.jsonl    — error records (append-only)
//! ```
//!
//! Atomic writes use `tempfile::NamedTempFile::persist` to guarantee that
//! a reader never observes a partial file — this makes `in_progress_scan`
//! safe across server restarts.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

/// Status of a batch job (persisted to `status.json`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum BatchJobStatus {
    /// Submitted, not yet picked up by a worker.
    Pending,
    /// Actively processing.
    InProgress,
    /// All lines processed successfully.
    Completed,
    /// One or more lines failed; see `errors.jsonl`.
    Failed,
    /// Cancelled before completion.
    Cancelled,
}

/// Metadata snapshot written to (and read from) `status.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJobMeta {
    /// Stable job identifier (`batch_<uuid>`).
    pub id: String,
    /// Endpoint the batch targets (e.g. `/v1/chat/completions`).
    pub endpoint: String,
    /// Current status.
    pub status: BatchJobStatus,
    /// Total number of request lines in the input file.
    pub total_lines: u32,
    /// Number of lines processed so far.
    pub completed_lines: u32,
    /// Number of lines that returned an error.
    pub failed_lines: u32,
    /// Unix timestamp (seconds) when the job was created.
    pub created_at: i64,
    /// Unix timestamp when the job was last updated.
    pub updated_at: i64,
    /// Whether a cancel has been requested.
    pub cancel_requested: bool,
}

/// Disk-backed store for batch jobs.
///
/// All methods are synchronous — they are intended to be called from a
/// `tokio::task::spawn_blocking` context or from the background batch worker.
pub struct BatchStore {
    dir: PathBuf,
}

impl BatchStore {
    /// Open (or create) the spool directory.
    ///
    /// Returns an error if the directory cannot be created.
    pub fn new(dir: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Path to a job's subdirectory.
    pub fn job_dir(&self, job_id: &str) -> PathBuf {
        self.dir.join(job_id)
    }

    /// Create a new job on disk.
    ///
    /// Writes `input.jsonl` with the supplied content and writes an initial
    /// `status.json` atomically.
    ///
    /// Returns an error if a job with the same ID already exists.
    pub fn create_job(
        &self,
        job_id: &str,
        input_jsonl: &str,
        endpoint: &str,
        total_lines: u32,
    ) -> std::io::Result<BatchJobMeta> {
        let dir = self.job_dir(job_id);
        fs::create_dir_all(&dir)?;

        // Write input (not atomic — written once before any worker sees it).
        let input_path = dir.join("input.jsonl");
        let mut f = File::create(&input_path)?;
        f.write_all(input_jsonl.as_bytes())?;
        f.flush()?;

        let now = unix_now();
        let meta = BatchJobMeta {
            id: job_id.to_string(),
            endpoint: endpoint.to_string(),
            status: BatchJobStatus::Pending,
            total_lines,
            completed_lines: 0,
            failed_lines: 0,
            created_at: now,
            updated_at: now,
            cancel_requested: false,
        };
        self.write_status_atomic(&dir, &meta)?;
        Ok(meta)
    }

    /// Overwrite `status.json` atomically.
    pub fn update_status(&self, job_id: &str, status: &BatchJobMeta) -> std::io::Result<()> {
        let dir = self.job_dir(job_id);
        self.write_status_atomic(&dir, status)
    }

    /// Read `status.json` for a job.
    pub fn read_status(&self, job_id: &str) -> std::io::Result<BatchJobMeta> {
        let path = self.job_dir(job_id).join("status.json");
        let content = fs::read_to_string(&path)?;
        serde_json::from_str(&content).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("status.json is invalid JSON: {e}"),
            )
        })
    }

    /// Append a single JSONL line to `output.jsonl`.
    ///
    /// The line must NOT contain a trailing newline — one will be added.
    pub fn append_output(&self, job_id: &str, line: &str) -> std::io::Result<()> {
        let path = self.job_dir(job_id).join("output.jsonl");
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{}", line)?;
        Ok(())
    }

    /// Append a single JSONL error record to `errors.jsonl`.
    pub fn append_error(&self, job_id: &str, line: &str) -> std::io::Result<()> {
        let path = self.job_dir(job_id).join("errors.jsonl");
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{}", line)?;
        Ok(())
    }

    /// Read all lines from `input.jsonl`.
    pub fn read_input_lines(&self, job_id: &str) -> std::io::Result<Vec<String>> {
        let path = self.job_dir(job_id).join("input.jsonl");
        let f = File::open(&path)?;
        let reader = BufReader::new(f);
        reader
            .lines()
            .filter(|l| l.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(true))
            .collect()
    }

    /// Read all output lines from `output.jsonl`.
    pub fn read_output_lines(&self, job_id: &str) -> std::io::Result<Vec<String>> {
        let path = self.job_dir(job_id).join("output.jsonl");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let f = File::open(&path)?;
        let reader = BufReader::new(f);
        reader.lines().collect()
    }

    /// List all job IDs under the spool directory.
    pub fn list_jobs(&self) -> std::io::Result<Vec<String>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
        Ok(ids)
    }

    /// Return job IDs whose status is `InProgress` or `Pending`.
    ///
    /// Used at startup to re-enqueue jobs that survived a restart.
    pub fn in_progress_jobs(&self) -> std::io::Result<Vec<String>> {
        let ids = self.list_jobs()?;
        let mut out = Vec::new();
        for id in ids {
            if let Ok(meta) = self.read_status(&id) {
                if matches!(
                    meta.status,
                    BatchJobStatus::InProgress | BatchJobStatus::Pending
                ) {
                    out.push(id);
                }
            }
        }
        Ok(out)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Write `status.json` atomically using a temp file in the same directory.
    fn write_status_atomic(&self, dir: &Path, meta: &BatchJobMeta) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(meta)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        // Create a NamedTempFile in the *same* directory to ensure the
        // rename is atomic (same filesystem). If we used the system temp
        // dir the rename might cross filesystems and fail.
        let mut tmp = NamedTempFile::new_in(dir)?;
        tmp.write_all(json.as_bytes())?;
        tmp.flush()?;

        let status_path = dir.join("status.json");
        tmp.persist(&status_path).map_err(|e| e.error)?;
        Ok(())
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn temp_store(suffix: &str) -> BatchStore {
        // Use a unique directory per test invocation to avoid cross-run pollution.
        let id = uuid::Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_batch_test_{suffix}_{id}"));
        BatchStore::new(dir).expect("should create store")
    }

    /// (a) store_create_and_read_status — create job, read status back; matches.
    #[test]
    fn store_create_and_read_status() {
        let store = temp_store("create_read");
        let job_id = "batch_test_a";

        let meta = store
            .create_job(job_id, "line1\nline2\n", "/v1/chat/completions", 2)
            .expect("create_job should succeed");

        assert_eq!(meta.id, job_id);
        assert_eq!(meta.total_lines, 2);
        assert_eq!(meta.status, BatchJobStatus::Pending);

        let read_back = store
            .read_status(job_id)
            .expect("read_status should succeed");
        assert_eq!(read_back.id, meta.id);
        assert_eq!(read_back.total_lines, meta.total_lines);
        assert_eq!(read_back.status, meta.status);
    }

    /// (b) store_append_output_is_ordered — append 5 lines; read back; order preserved.
    #[test]
    fn store_append_output_is_ordered() {
        let store = temp_store("append_order");
        let job_id = "batch_test_b";
        store
            .create_job(job_id, "", "/v1/chat/completions", 5)
            .expect("create_job");

        for i in 0..5_u32 {
            store
                .append_output(job_id, &format!(r#"{{"index":{i}}}"#))
                .expect("append_output");
        }

        let lines = store.read_output_lines(job_id).expect("read_output_lines");
        assert_eq!(lines.len(), 5, "should have 5 output lines");
        for (i, line) in lines.iter().enumerate() {
            let val: serde_json::Value =
                serde_json::from_str(line).expect("line should be valid JSON");
            assert_eq!(
                val["index"].as_u64(),
                Some(i as u64),
                "line order must be preserved at index {i}"
            );
        }
    }

    /// (c) store_atomic_write_no_partial — status.json is always valid JSON.
    ///
    /// We write, then immediately read back — the atomic rename ensures we
    /// never read a partial file even if a concurrent write was in progress.
    #[test]
    fn store_atomic_write_no_partial() {
        let store = temp_store("atomic");
        let job_id = "batch_test_c";
        let mut meta = store
            .create_job(job_id, "", "/v1/chat/completions", 10)
            .expect("create_job");

        for i in 0..10_u32 {
            meta.completed_lines = i;
            meta.updated_at = unix_now();
            store.update_status(job_id, &meta).expect("update_status");

            // Read back immediately — must always be valid JSON.
            let read_back = store
                .read_status(job_id)
                .expect("status.json should be valid");
            assert_eq!(
                read_back.completed_lines, i,
                "completed_lines mismatch at iteration {i}"
            );
        }
    }

    /// (d) store_in_progress_scan — 3 jobs (1 completed, 2 in-progress);
    ///     `in_progress_jobs()` returns exactly 2.
    #[test]
    fn store_in_progress_scan() {
        let store = temp_store("scan");

        // Create job-1: completed
        let mut m1 = store
            .create_job("scan_job_1", "", "/v1/chat/completions", 0)
            .expect("create job 1");
        m1.status = BatchJobStatus::Completed;
        store.update_status("scan_job_1", &m1).expect("update 1");

        // Create job-2: in_progress
        let mut m2 = store
            .create_job("scan_job_2", "", "/v1/chat/completions", 5)
            .expect("create job 2");
        m2.status = BatchJobStatus::InProgress;
        store.update_status("scan_job_2", &m2).expect("update 2");

        // Create job-3: pending (counts as in-progress for restart)
        store
            .create_job("scan_job_3", "", "/v1/chat/completions", 3)
            .expect("create job 3");

        let in_progress = store
            .in_progress_jobs()
            .expect("in_progress_jobs should succeed");

        assert_eq!(
            in_progress.len(),
            2,
            "should find exactly 2 resumable jobs: {in_progress:?}"
        );
        assert!(
            in_progress.contains(&"scan_job_2".to_string()),
            "scan_job_2 should be in results"
        );
        assert!(
            in_progress.contains(&"scan_job_3".to_string()),
            "scan_job_3 should be in results"
        );
    }

    /// Store directory is created if it does not exist.
    #[test]
    fn store_creates_directory() {
        let dir = temp_dir().join("oxillama_batch_test_dir_creation_xyz");
        let _ = std::fs::remove_dir_all(&dir); // clean up from previous run
        let _store = BatchStore::new(dir.clone()).expect("BatchStore::new should succeed");
        assert!(dir.exists(), "spool directory should be created");
    }
}
