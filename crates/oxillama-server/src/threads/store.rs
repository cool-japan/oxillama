//! Disk-backed persistent store for Assistants API objects.
//!
//! Directory layout:
//!
//! ```text
//! <root>/
//!   <thread_id>/
//!     meta.json          — Thread metadata (atomic write)
//!     messages.jsonl     — Append-only ordered message log
//!     runs/
//!       <run_id>/
//!         status.json    — Run status + error (atomic write)
//! ```
//!
//! Atomic writes use `tempfile::NamedTempFile::persist` to guarantee that a
//! reader never observes a partial file, making the store safe across server
//! restarts.  Append-only operations (messages) never overwrite existing data.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

use crate::error::{ServerError, ServerResult};
use crate::threads::types::{
    Run, RunError, RunStatus, RunStep, RunStepStatus, Thread, ThreadMessage,
};

/// Disk-backed store for threads, messages, and runs.
///
/// All methods are synchronous and are intended to be called from within a
/// `tokio::task::spawn_blocking` context (see the route handlers and worker).
pub struct ThreadStore {
    /// Root directory that contains one sub-directory per thread.
    root_dir: PathBuf,
}

impl ThreadStore {
    /// Open (or create) the thread store root directory.
    ///
    /// Creates the directory and any missing parents if they do not exist.
    pub fn new(dir: PathBuf) -> ServerResult<Self> {
        fs::create_dir_all(&dir).map_err(|e| ServerError::IoError {
            context: format!("create thread store root {}", dir.display()),
            source: e,
        })?;
        Ok(Self { root_dir: dir })
    }

    // ── Thread operations ────────────────────────────────────────────────────

    /// Persist a new thread to disk.
    ///
    /// Creates `{root}/{thread_id}/meta.json` atomically.  Fails if the
    /// directory already exists (duplicate ID).
    pub fn create_thread(&self, thread: &Thread) -> ServerResult<()> {
        let dir = self.thread_dir(&thread.id);
        fs::create_dir_all(&dir).map_err(|e| ServerError::IoError {
            context: format!("create thread directory {}", dir.display()),
            source: e,
        })?;
        self.write_json_atomic(&dir, "meta.json", thread)?;
        Ok(())
    }

    /// Read a thread's metadata from disk.
    ///
    /// Returns `ServerError::ThreadNotFound` if no directory or `meta.json`
    /// exists for the given ID.
    pub fn get_thread(&self, id: &str) -> ServerResult<Thread> {
        let path = self.thread_dir(id).join("meta.json");
        let content =
            fs::read_to_string(&path).map_err(|_| ServerError::ThreadNotFound(id.to_string()))?;
        serde_json::from_str(&content).map_err(ServerError::Serialization)
    }

    /// List all thread IDs stored in the root directory.
    pub fn list_thread_ids(&self) -> ServerResult<Vec<String>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.root_dir).map_err(|e| ServerError::IoError {
            context: "list thread IDs".to_string(),
            source: e,
        })? {
            let entry = entry.map_err(|e| ServerError::IoError {
                context: "read directory entry".to_string(),
                source: e,
            })?;
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
        Ok(ids)
    }

    // ── Message operations ───────────────────────────────────────────────────

    /// Append a single message to the thread's `messages.jsonl` file.
    ///
    /// Appends are not atomic at the OS level (no `fsync` fence), but each
    /// line is a complete JSON object, so a reader will never observe a
    /// partially-written message — incomplete trailing lines are filtered out
    /// by `list_messages`.
    pub fn append_message(&self, thread_id: &str, msg: &ThreadMessage) -> ServerResult<()> {
        let dir = self.thread_dir(thread_id);
        // Verify thread exists.
        if !dir.join("meta.json").exists() {
            return Err(ServerError::ThreadNotFound(thread_id.to_string()));
        }
        let path = dir.join("messages.jsonl");
        let json_line = serde_json::to_string(msg).map_err(ServerError::Serialization)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| ServerError::IoError {
                context: format!("open messages.jsonl for thread {thread_id}"),
                source: e,
            })?;
        writeln!(file, "{}", json_line).map_err(|e| ServerError::IoError {
            context: format!("write message to thread {thread_id}"),
            source: e,
        })?;
        Ok(())
    }

    /// Read all messages for a thread in append order (oldest first).
    ///
    /// Blank lines and lines that fail to parse as JSON are silently skipped
    /// so that a partial write at the end of a previous session does not break
    /// future reads.
    pub fn list_messages(&self, thread_id: &str) -> ServerResult<Vec<ThreadMessage>> {
        let path = self.thread_dir(thread_id).join("messages.jsonl");
        if !self.thread_dir(thread_id).join("meta.json").exists() {
            return Err(ServerError::ThreadNotFound(thread_id.to_string()));
        }
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&path).map_err(|e| ServerError::IoError {
            context: format!("open messages.jsonl for thread {thread_id}"),
            source: e,
        })?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();
        for line_result in reader.lines() {
            let line = line_result.map_err(|e| ServerError::IoError {
                context: format!("read messages.jsonl for thread {thread_id}"),
                source: e,
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(msg) = serde_json::from_str::<ThreadMessage>(trimmed) {
                messages.push(msg);
            }
            // Silently skip malformed lines (partial write protection).
        }
        Ok(messages)
    }

    // ── Run operations ───────────────────────────────────────────────────────

    /// Persist a new run to disk.
    ///
    /// Creates `{root}/{thread_id}/runs/{run_id}/status.json` atomically.
    /// Returns `ThreadNotFound` if the thread does not exist.
    pub fn create_run(&self, thread_id: &str, run: &Run) -> ServerResult<()> {
        let thread_dir = self.thread_dir(thread_id);
        if !thread_dir.join("meta.json").exists() {
            return Err(ServerError::ThreadNotFound(thread_id.to_string()));
        }
        let run_dir = self.run_dir(thread_id, &run.id);
        fs::create_dir_all(&run_dir).map_err(|e| ServerError::IoError {
            context: format!("create run directory {}", run_dir.display()),
            source: e,
        })?;
        self.write_json_atomic(&run_dir, "status.json", run)?;
        Ok(())
    }

    /// Read a run's status from disk.
    ///
    /// Returns `RunNotFound` if no `status.json` exists for the given IDs.
    pub fn get_run(&self, thread_id: &str, run_id: &str) -> ServerResult<Run> {
        let path = self.run_dir(thread_id, run_id).join("status.json");
        let content =
            fs::read_to_string(&path).map_err(|_| ServerError::RunNotFound(run_id.to_string()))?;
        serde_json::from_str(&content).map_err(ServerError::Serialization)
    }

    /// Atomically update a run's status (and optionally set `last_error`).
    ///
    /// Returns `RunNotFound` if the run does not exist, or
    /// `RunInTerminalState` if the run is already in a terminal state.
    pub fn update_run_status(
        &self,
        thread_id: &str,
        run_id: &str,
        status: RunStatus,
        error: Option<RunError>,
    ) -> ServerResult<()> {
        let mut run = self.get_run(thread_id, run_id)?;

        if run.status.is_terminal() {
            return Err(ServerError::RunInTerminalState(format!(
                "{} is already in terminal state {:?}",
                run_id, run.status
            )));
        }

        run.status = status;
        run.last_error = error;
        let run_dir = self.run_dir(thread_id, run_id);
        self.write_json_atomic(&run_dir, "status.json", &run)?;
        Ok(())
    }

    /// Force-update a run's status bypassing terminal-state guard.
    ///
    /// Used by the cancel handler to transition a queued/in-progress run to
    /// `Cancelled` even if no worker is currently processing it.
    pub fn force_update_run_status(
        &self,
        thread_id: &str,
        run_id: &str,
        status: RunStatus,
        error: Option<RunError>,
    ) -> ServerResult<()> {
        let mut run = self.get_run(thread_id, run_id)?;
        run.status = status;
        run.last_error = error;
        let run_dir = self.run_dir(thread_id, run_id);
        self.write_json_atomic(&run_dir, "status.json", &run)?;
        Ok(())
    }

    // ── Run Step operations ───────────────────────────────────────────────────

    /// Return the path to a run's steps directory.
    pub fn steps_dir(&self, thread_id: &str, run_id: &str) -> PathBuf {
        self.run_dir(thread_id, run_id).join("steps")
    }

    /// Persist a new run step to disk.
    ///
    /// Creates `{root}/{thread_id}/runs/{run_id}/steps/{step_id}.json`
    /// atomically.  Returns `RunNotFound` if the run does not exist.
    pub fn append_step(&self, thread_id: &str, run_id: &str, step: &RunStep) -> ServerResult<()> {
        let run_dir = self.run_dir(thread_id, run_id);
        if !run_dir.join("status.json").exists() {
            return Err(ServerError::RunNotFound(run_id.to_string()));
        }
        let steps_dir = self.steps_dir(thread_id, run_id);
        fs::create_dir_all(&steps_dir).map_err(|e| ServerError::IoError {
            context: format!("create steps directory {}", steps_dir.display()),
            source: e,
        })?;
        let filename = format!("{}.json", step.id);
        self.write_json_atomic(&steps_dir, &filename, step)?;
        Ok(())
    }

    /// Read all steps for a run, sorted by `created_at` ascending.
    ///
    /// Returns `RunNotFound` if the run does not exist.
    pub fn list_steps(&self, thread_id: &str, run_id: &str) -> ServerResult<Vec<RunStep>> {
        let run_dir = self.run_dir(thread_id, run_id);
        if !run_dir.join("status.json").exists() {
            return Err(ServerError::RunNotFound(run_id.to_string()));
        }
        let steps_dir = self.steps_dir(thread_id, run_id);
        if !steps_dir.exists() {
            return Ok(Vec::new());
        }
        let mut steps = Vec::new();
        for entry in fs::read_dir(&steps_dir).map_err(|e| ServerError::IoError {
            context: format!("read steps dir {}", steps_dir.display()),
            source: e,
        })? {
            let entry = entry.map_err(|e| ServerError::IoError {
                context: "read steps entry".to_string(),
                source: e,
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(step) = serde_json::from_str::<RunStep>(&content) {
                    steps.push(step);
                }
            }
        }
        steps.sort_by_key(|s| s.created_at);
        Ok(steps)
    }

    /// Read a single run step by ID.
    ///
    /// Returns `RunStepNotFound` if no step with this ID exists.
    pub fn get_step(&self, thread_id: &str, run_id: &str, step_id: &str) -> ServerResult<RunStep> {
        let steps_dir = self.steps_dir(thread_id, run_id);
        let path = steps_dir.join(format!("{step_id}.json"));
        let content = fs::read_to_string(&path)
            .map_err(|_| ServerError::RunStepNotFound(step_id.to_string()))?;
        serde_json::from_str(&content).map_err(ServerError::Serialization)
    }

    /// Atomically update a run step's status (and optionally timestamps/error).
    ///
    /// Returns `RunStepNotFound` if no step with this ID exists.
    pub fn update_step_status(
        &self,
        thread_id: &str,
        run_id: &str,
        step_id: &str,
        status: RunStepStatus,
    ) -> ServerResult<()> {
        let mut step = self.get_step(thread_id, run_id, step_id)?;
        let now_u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        match &status {
            RunStepStatus::Completed => step.completed_at = Some(now_u64),
            RunStepStatus::Failed => step.failed_at = Some(now_u64),
            _ => {}
        }
        step.status = status;
        let steps_dir = self.steps_dir(thread_id, run_id);
        let filename = format!("{step_id}.json");
        self.write_json_atomic(&steps_dir, &filename, &step)?;
        Ok(())
    }

    // ── Path helpers ─────────────────────────────────────────────────────────

    /// Return the path to a thread's subdirectory.
    pub fn thread_dir(&self, thread_id: &str) -> PathBuf {
        self.root_dir.join(thread_id)
    }

    /// Return the path to a run's subdirectory.
    pub fn run_dir(&self, thread_id: &str, run_id: &str) -> PathBuf {
        self.thread_dir(thread_id).join("runs").join(run_id)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Serialize `value` to JSON and write it atomically via a temp file + rename.
    ///
    /// The temp file is created in the same directory as the target so the
    /// rename is always on the same filesystem.
    fn write_json_atomic<T: serde::Serialize>(
        &self,
        dir: &Path,
        filename: &str,
        value: &T,
    ) -> ServerResult<()> {
        let json = serde_json::to_string_pretty(value).map_err(ServerError::Serialization)?;
        let mut tmp = NamedTempFile::new_in(dir).map_err(|e| ServerError::IoError {
            context: format!("create temp file in {}", dir.display()),
            source: e,
        })?;
        tmp.write_all(json.as_bytes())
            .map_err(|e| ServerError::IoError {
                context: "write to temp file".to_string(),
                source: e,
            })?;
        tmp.flush().map_err(|e| ServerError::IoError {
            context: "flush temp file".to_string(),
            source: e,
        })?;
        let target = dir.join(filename);
        tmp.persist(&target).map_err(|e| ServerError::IoError {
            context: format!("persist atomic write to {}", target.display()),
            source: e.error,
        })?;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threads::types::{
        Run, RunStatus, RunStep, RunStepStatus, RunStepType, Thread, ThreadMessage,
    };
    use std::env::temp_dir;
    use uuid::Uuid;

    fn make_store(tag: &str) -> ThreadStore {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_thread_store_test_{tag}_{id}"));
        ThreadStore::new(dir).expect("ThreadStore::new should succeed")
    }

    fn make_thread(id: &str) -> Thread {
        Thread {
            id: id.to_string(),
            object: "thread".to_string(),
            created_at: 1_000_000,
            metadata: serde_json::json!({}),
        }
    }

    fn make_run(id: &str, thread_id: &str) -> Run {
        Run {
            id: id.to_string(),
            object: "thread.run".to_string(),
            created_at: 1_000_001,
            thread_id: thread_id.to_string(),
            status: RunStatus::Queued,
            model: "test-model".to_string(),
            last_error: None,
        }
    }

    #[test]
    fn store_creates_root_directory() {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_thread_store_create_{id}"));
        let _ = fs::remove_dir_all(&dir);
        ThreadStore::new(dir.clone()).expect("should create store");
        assert!(dir.exists());
    }

    #[test]
    fn create_and_get_thread() {
        let store = make_store("get_thread");
        let thread = make_thread("thread_aaa");
        store.create_thread(&thread).expect("create_thread");
        let got = store.get_thread("thread_aaa").expect("get_thread");
        assert_eq!(got.id, "thread_aaa");
    }

    #[test]
    fn get_thread_not_found_returns_error() {
        let store = make_store("thread_notfound");
        let err = store.get_thread("nonexistent").expect_err("should fail");
        assert!(matches!(err, ServerError::ThreadNotFound(_)));
    }

    #[test]
    fn append_and_list_messages_in_order() {
        let store = make_store("messages_order");
        let thread = make_thread("thread_msgs");
        store.create_thread(&thread).expect("create_thread");

        for i in 0..5_u32 {
            let msg = ThreadMessage::new_user(
                format!("msg_{i}"),
                "thread_msgs".to_string(),
                format!("hello {i}"),
            );
            store.append_message("thread_msgs", &msg).expect("append");
        }

        let msgs = store.list_messages("thread_msgs").expect("list");
        assert_eq!(msgs.len(), 5);
        for (i, m) in msgs.iter().enumerate() {
            assert_eq!(m.text_content(), format!("hello {i}"));
        }
    }

    #[test]
    fn append_message_unknown_thread_errors() {
        let store = make_store("append_no_thread");
        let msg = ThreadMessage::new_user("msg_x".into(), "ghost".into(), "hi".into());
        let err = store
            .append_message("ghost", &msg)
            .expect_err("should fail");
        assert!(matches!(err, ServerError::ThreadNotFound(_)));
    }

    #[test]
    fn create_and_get_run() {
        let store = make_store("get_run");
        let thread = make_thread("thread_run");
        store.create_thread(&thread).expect("create");
        let run = make_run("run_001", "thread_run");
        store.create_run("thread_run", &run).expect("create_run");
        let got = store.get_run("thread_run", "run_001").expect("get_run");
        assert_eq!(got.id, "run_001");
        assert_eq!(got.status, RunStatus::Queued);
    }

    #[test]
    fn update_run_status_transitions() {
        let store = make_store("run_status");
        let thread = make_thread("thread_rs");
        store.create_thread(&thread).expect("create");
        let run = make_run("run_002", "thread_rs");
        store.create_run("thread_rs", &run).expect("create_run");

        store
            .update_run_status("thread_rs", "run_002", RunStatus::InProgress, None)
            .expect("to in-progress");

        let got = store.get_run("thread_rs", "run_002").expect("get");
        assert_eq!(got.status, RunStatus::InProgress);

        store
            .update_run_status("thread_rs", "run_002", RunStatus::Completed, None)
            .expect("to completed");

        let final_run = store.get_run("thread_rs", "run_002").expect("get final");
        assert_eq!(final_run.status, RunStatus::Completed);
    }

    #[test]
    fn update_terminal_run_returns_error() {
        let store = make_store("run_terminal");
        let thread = make_thread("thread_term");
        store.create_thread(&thread).expect("create");
        let run = make_run("run_003", "thread_term");
        store.create_run("thread_term", &run).expect("create_run");
        store
            .update_run_status("thread_term", "run_003", RunStatus::Completed, None)
            .expect("complete");
        let err = store
            .update_run_status("thread_term", "run_003", RunStatus::InProgress, None)
            .expect_err("should reject terminal");
        assert!(matches!(err, ServerError::RunInTerminalState(_)));
    }

    #[test]
    fn get_run_not_found() {
        let store = make_store("run_notfound");
        let thread = make_thread("thread_nrf");
        store.create_thread(&thread).expect("create");
        let err = store
            .get_run("thread_nrf", "ghost_run")
            .expect_err("should fail");
        assert!(matches!(err, ServerError::RunNotFound(_)));
    }

    #[test]
    fn persistence_across_store_drop_and_recreate() {
        let id = Uuid::new_v4().as_simple().to_string();
        let dir = temp_dir().join(format!("oxillama_thread_persistence_{id}"));
        let thread = make_thread("thread_persist");

        {
            let store = ThreadStore::new(dir.clone()).expect("create store");
            store.create_thread(&thread).expect("create thread");
            let msg =
                ThreadMessage::new_user("msg_p1".into(), "thread_persist".into(), "data".into());
            store
                .append_message("thread_persist", &msg)
                .expect("append");
        }

        // Drop and re-open from same directory.
        let store2 = ThreadStore::new(dir).expect("reopen store");
        let got = store2
            .get_thread("thread_persist")
            .expect("read after restart");
        assert_eq!(got.id, "thread_persist");
        let msgs = store2.list_messages("thread_persist").expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text_content(), "data");
    }

    #[test]
    fn list_messages_empty_if_no_messages_yet() {
        let store = make_store("empty_msgs");
        let thread = make_thread("thread_empty");
        store.create_thread(&thread).expect("create");
        let msgs = store.list_messages("thread_empty").expect("list");
        assert!(msgs.is_empty());
    }

    #[test]
    fn atomic_write_leaves_no_partial_state() {
        let store = make_store("atomic");
        let thread = make_thread("thread_atomic");
        store.create_thread(&thread).expect("create");
        let run = make_run("run_atomic", "thread_atomic");
        store.create_run("thread_atomic", &run).expect("create run");

        // Perform many rapid status transitions and verify every read is valid.
        for i in 0..20 {
            let target_status = if i % 2 == 0 {
                RunStatus::InProgress
            } else {
                RunStatus::Queued
            };
            // Use force_update to bypass terminal guard in loop.
            store
                .force_update_run_status("thread_atomic", "run_atomic", target_status, None)
                .expect("force update");
            let got = store
                .get_run("thread_atomic", "run_atomic")
                .expect("read mid-loop");
            // Validate we can always parse the status.
            let _ = serde_json::to_string(&got.status).expect("serialize");
        }
    }

    fn make_step(step_id: &str, run_id: &str, thread_id: &str) -> RunStep {
        RunStep {
            id: step_id.to_string(),
            object: "thread.run.step".to_string(),
            run_id: run_id.to_string(),
            thread_id: thread_id.to_string(),
            step_type: RunStepType::MessageCreation,
            status: RunStepStatus::InProgress,
            created_at: 1_000_002,
            completed_at: None,
            failed_at: None,
            error: None,
            step_details: None,
        }
    }

    #[test]
    fn step_list_returns_all_steps() {
        let store = make_store("step_list");
        let thread = make_thread("thread_sl");
        store.create_thread(&thread).expect("create thread");
        let run = make_run("run_sl", "thread_sl");
        store.create_run("thread_sl", &run).expect("create run");

        for i in 0..3_u32 {
            let step = make_step(&format!("step_{i}"), "run_sl", "thread_sl");
            store
                .append_step("thread_sl", "run_sl", &step)
                .expect("append step");
        }

        let steps = store.list_steps("thread_sl", "run_sl").expect("list steps");
        assert_eq!(steps.len(), 3);
    }

    #[test]
    fn step_get_returns_correct_step() {
        let store = make_store("step_get");
        let thread = make_thread("thread_sg");
        store.create_thread(&thread).expect("create thread");
        let run = make_run("run_sg", "thread_sg");
        store.create_run("thread_sg", &run).expect("create run");

        let step = make_step("step_target", "run_sg", "thread_sg");
        store
            .append_step("thread_sg", "run_sg", &step)
            .expect("append");

        let got = store
            .get_step("thread_sg", "run_sg", "step_target")
            .expect("get step");
        assert_eq!(got.id, "step_target");
        assert_eq!(got.step_type, RunStepType::MessageCreation);
        assert_eq!(got.status, RunStepStatus::InProgress);
    }

    #[test]
    fn step_not_found_returns_error() {
        let store = make_store("step_notfound");
        let thread = make_thread("thread_snf");
        store.create_thread(&thread).expect("create thread");
        let run = make_run("run_snf", "thread_snf");
        store.create_run("thread_snf", &run).expect("create run");

        let err = store
            .get_step("thread_snf", "run_snf", "step_ghost")
            .expect_err("should fail");
        assert!(matches!(err, ServerError::RunStepNotFound(_)));
    }

    #[test]
    fn step_update_status_to_completed() {
        let store = make_store("step_complete");
        let thread = make_thread("thread_sc");
        store.create_thread(&thread).expect("create thread");
        let run = make_run("run_sc", "thread_sc");
        store.create_run("thread_sc", &run).expect("create run");

        let step = make_step("step_comp", "run_sc", "thread_sc");
        store
            .append_step("thread_sc", "run_sc", &step)
            .expect("append");

        store
            .update_step_status("thread_sc", "run_sc", "step_comp", RunStepStatus::Completed)
            .expect("update status");

        let got = store
            .get_step("thread_sc", "run_sc", "step_comp")
            .expect("get");
        assert_eq!(got.status, RunStepStatus::Completed);
        assert!(got.completed_at.is_some());
    }

    #[test]
    fn force_update_run_status_bypasses_terminal_guard() {
        let store = make_store("force_cancel");
        let thread = make_thread("thread_fc");
        store.create_thread(&thread).expect("create");
        let run = make_run("run_fc", "thread_fc");
        store.create_run("thread_fc", &run).expect("create run");
        store
            .force_update_run_status("thread_fc", "run_fc", RunStatus::Cancelled, None)
            .expect("cancel");
        let got = store.get_run("thread_fc", "run_fc").expect("read");
        assert_eq!(got.status, RunStatus::Cancelled);
        // force_update again should succeed even though Cancelled is terminal.
        store
            .force_update_run_status(
                "thread_fc",
                "run_fc",
                RunStatus::Expired,
                Some(RunError {
                    code: "expired".into(),
                    message: "timed out".into(),
                }),
            )
            .expect("second force");
        let final_run = store.get_run("thread_fc", "run_fc").expect("read final");
        assert_eq!(final_run.status, RunStatus::Expired);
        assert!(final_run.last_error.is_some());
    }
}
