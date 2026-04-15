//! Memory usage benchmarking.
//!
//! Provides a cross-platform RSS (Resident Set Size) sampler in pure Rust.
//! On Linux the value is read from `/proc/self/status`; on macOS the value
//! is obtained by spawning `ps` (pure Rust, no FFI).

#[cfg(target_os = "linux")]
use std::fs;

/// Configuration for memory benchmarks.
///
/// Currently a unit-like struct; fields may be added in future iterations.
#[derive(Debug, Clone, Default)]
pub struct MemoryConfig;

/// Result of a memory benchmark snapshot.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    /// Peak RSS in bytes since process start.
    ///
    /// On Linux this is `VmPeak` from `/proc/self/status`.
    /// Returns `0` on platforms where reading is not supported.
    pub peak_rss_bytes: usize,
    /// Current RSS in bytes at snapshot time.
    ///
    /// On Linux this is `VmRSS` from `/proc/self/status`.
    /// Returns `0` on platforms where reading is not supported.
    pub current_rss_bytes: usize,
}

impl MemoryResult {
    /// Capture a memory snapshot of the current process.
    pub fn current() -> Self {
        let current = Self::read_current_rss().unwrap_or(0);
        let peak = Self::read_peak_rss().unwrap_or(current);
        Self {
            peak_rss_bytes: peak,
            current_rss_bytes: current,
        }
    }

    /// Compute the estimated model weight memory for a quantized tensor.
    ///
    /// `n_elements` is the total number of weights; `bits_per_weight` is the
    /// effective precision (e.g. 4.5 for Q4_K).
    pub fn model_weight_bytes(n_elements: usize, bits_per_weight: f32) -> usize {
        ((n_elements as f64 * bits_per_weight as f64) / 8.0).ceil() as usize
    }

    /// Estimate KV-cache memory for a given context window.
    ///
    /// `n_layers` × `n_heads` × `head_dim` × `context_len` × 2 (K and V) × 2
    /// bytes (BF16).
    pub fn kv_cache_bytes(
        n_layers: usize,
        n_heads: usize,
        head_dim: usize,
        context_len: usize,
    ) -> usize {
        n_layers * n_heads * head_dim * context_len * 2 * 2
    }

    // ── Platform-specific RSS readers ────────────────────────────────────────

    #[cfg(target_os = "linux")]
    fn read_current_rss() -> Option<usize> {
        Self::parse_proc_status_field("VmRSS:")
    }

    #[cfg(target_os = "linux")]
    fn read_peak_rss() -> Option<usize> {
        Self::parse_proc_status_field("VmPeak:")
    }

    #[cfg(target_os = "linux")]
    fn parse_proc_status_field(field: &str) -> Option<usize> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if line.starts_with(field) {
                // Format: "VmRSS:   12345 kB"
                let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    fn read_current_rss() -> Option<usize> {
        // Use `ps` to query RSS — pure Rust, no FFI needed.
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let text = String::from_utf8(output.stdout).ok()?;
        let kb: usize = text.trim().parse().ok()?;
        Some(kb * 1024)
    }

    #[cfg(target_os = "macos")]
    fn read_peak_rss() -> Option<usize> {
        // macOS doesn't easily expose peak RSS without FFI.
        // Return current RSS as a best-effort approximation.
        Self::read_current_rss()
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_current_rss() -> Option<usize> {
        Some(0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_peak_rss() -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_result_current_does_not_panic() {
        let result = MemoryResult::current();
        // Just verify it returns without panicking.
        let _ = result.current_rss_bytes;
        let _ = result.peak_rss_bytes;
    }

    #[test]
    fn test_model_weight_bytes_q4_k() {
        // 4096×4096 weights at 4.5 bits/weight = 4096*4096*4.5/8 = 9_437_184 bytes
        let expected = ((4096usize * 4096 * 45) as f64 / 80.0).ceil() as usize;
        let got = MemoryResult::model_weight_bytes(4096 * 4096, 4.5);
        assert_eq!(got, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_rss_nonzero() {
        let result = MemoryResult::current();
        assert!(result.current_rss_bytes > 0, "macOS RSS should be > 0");
    }

    #[test]
    fn test_current_rss_bytes_standalone() {
        // On Linux/macOS this should return Some(n) with n > 0.
        // On other platforms it may return None.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let rss = super::current_rss_bytes();
            assert!(rss.is_some(), "RSS should be readable on this platform");
            assert!(rss.expect("checked above") > 0);
        }
        // On unsupported platforms, just make sure it doesn't panic.
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = super::current_rss_bytes();
        }
    }

    #[test]
    fn test_kv_cache_bytes() {
        // LLaMA-7B-like: 32 layers, 32 heads, head_dim=128, ctx=2048
        let bytes = MemoryResult::kv_cache_bytes(32, 32, 128, 2048);
        // 32 * 32 * 128 * 2048 * 2 (K+V) * 2 (bytes/BF16) = 1_073_741_824
        assert_eq!(bytes, 1_073_741_824);
    }
}

/// Memory estimate for model inference resource planning.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Model weight memory in bytes.
    pub model_bytes: u64,
    /// KV cache memory in bytes.
    pub kv_cache_bytes: u64,
    /// Runtime overhead memory in bytes.
    pub overhead_bytes: u64,
    /// Total estimated memory in bytes.
    pub total_bytes: u64,
}

impl MemoryEstimate {
    /// Format the result for human-readable display.
    pub fn display(&self) -> String {
        format!(
            "Memory usage:\n  Model weights: {}\n  KV cache:      {}\n  Overhead:      {}\n  Total:         {}",
            format_bytes(self.model_bytes),
            format_bytes(self.kv_cache_bytes),
            format_bytes(self.overhead_bytes),
            format_bytes(self.total_bytes),
        )
    }
}

/// Compute a memory breakdown estimate for a model.
///
/// # Parameters
/// - `model_file_size`: size of the GGUF file in bytes
/// - `hidden_size`: model hidden dimension
/// - `num_layers`: number of transformer layers
/// - `num_kv_heads`: number of KV attention heads
/// - `head_dim`: dimension per head
/// - `context_size`: max context length for KV cache estimation
pub fn estimate_memory(
    model_file_size: u64,
    hidden_size: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    context_size: usize,
) -> MemoryEstimate {
    let model_bytes = model_file_size;

    // KV cache: 2 (K+V) × layers × kv_heads × head_dim × context × f32 (4 bytes)
    let kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * context_size * 4;

    // Runtime overhead estimate: scratch buffers, embeddings, etc.
    // Roughly: hidden_size × 4 (f32) × ~10 buffers + vocab × hidden × 4
    let scratch_overhead = (hidden_size * 4 * 10) as u64;
    let overhead_bytes = scratch_overhead;

    let total_bytes = model_bytes + kv_cache_bytes as u64 + overhead_bytes;

    MemoryEstimate {
        model_bytes,
        kv_cache_bytes: kv_cache_bytes as u64,
        overhead_bytes,
        total_bytes,
    }
}

/// Read the current process RSS (Resident Set Size).
///
/// On Linux reads from `/proc/self/status`; on macOS spawns `ps`.
/// Returns `None` on unsupported platforms.
pub fn current_rss_bytes() -> Option<u64> {
    current_rss_bytes_impl()
}

#[cfg(target_os = "linux")]
fn current_rss_bytes_impl() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<u64>() {
                    return Some(kb * 1024);
                }
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn current_rss_bytes_impl() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let kb: u64 = text.trim().parse().ok()?;
    Some(kb * 1024)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn current_rss_bytes_impl() -> Option<u64> {
    None
}

/// Track RSS growth across an operation.
pub struct RssTracker {
    baseline: Option<u64>,
}

impl RssTracker {
    /// Start tracking RSS.
    pub fn start() -> Self {
        Self {
            baseline: current_rss_bytes(),
        }
    }

    /// Get the RSS growth since tracking started, in bytes.
    pub fn growth_bytes(&self) -> Option<u64> {
        let current = current_rss_bytes()?;
        let baseline = self.baseline?;
        Some(current.saturating_sub(baseline))
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod estimate_tests {
    use super::*;

    #[test]
    fn test_estimate_memory() {
        let result = estimate_memory(
            4_000_000_000, // 4GB model
            4096,          // hidden_size
            32,            // layers
            8,             // kv_heads
            128,           // head_dim
            2048,          // context
        );

        assert!(result.model_bytes == 4_000_000_000);
        // KV cache: 2 * 32 * 8 * 128 * 2048 * 4 = 536_870_912 bytes = 512 MiB
        assert_eq!(result.kv_cache_bytes, 536_870_912);
        assert!(result.total_bytes > result.model_bytes);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1_048_576), "1.00 MiB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GiB");
    }

    #[test]
    fn test_memory_result_display() {
        let result = MemoryEstimate {
            model_bytes: 1_073_741_824,
            kv_cache_bytes: 536_870_912,
            overhead_bytes: 163_840,
            total_bytes: 1_610_776_576,
        };
        let display = result.display();
        assert!(display.contains("GiB"));
        assert!(display.contains("MiB"));
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn test_current_rss_bytes() {
        let rss = current_rss_bytes();
        assert!(rss.is_some());
        if let Some(val) = rss {
            assert!(val > 0);
        }
    }

    #[test]
    fn test_rss_tracker() {
        let tracker = RssTracker::start();
        // Allocate some memory to cause RSS growth
        let _data: Vec<u8> = vec![1u8; 1024 * 1024];
        // The growth might be zero on some systems depending on page allocation
        let _growth = tracker.growth_bytes();
    }
}
