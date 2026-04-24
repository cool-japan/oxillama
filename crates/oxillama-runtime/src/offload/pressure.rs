//! Memory pressure probe — lightweight OS-level RAM usage monitor.
//!
//! [`MemoryPressureProbe`] exposes a single [`is_high()`] method that returns
//! `true` when the process's RSS exceeds a configurable `high_watermark`
//! fraction of total physical RAM.  When pressure is high the pager should
//! prefer aggressive eviction.
//!
//! Currently implemented for Linux (via `/proc/self/status` + `/proc/meminfo`).
//! macOS is a stub that always returns `false`; patches welcome via the Mach
//! task-info API.
//!
//! [`is_high()`]: MemoryPressureProbe::is_high

/// Returns current host RSS as a fraction of total physical RAM.
///
/// Returns `None` if the platform is unsupported or the OS files cannot be
/// parsed.
pub fn host_memory_pressure() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        linux_rss_fraction()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn linux_rss_fraction() -> Option<f64> {
    // Parse /proc/self/status for VmRSS and /proc/meminfo for MemTotal.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let vm_rss_kb: u64 = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    let vm_rss = vm_rss_kb * 1024; // kB → bytes

    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mem_total_kb: u64 = meminfo
        .lines()
        .find(|l| l.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    let mem_total = mem_total_kb * 1024; // kB → bytes

    if mem_total == 0 {
        return None;
    }
    Some(vm_rss as f64 / mem_total as f64)
}

/// Lightweight memory pressure monitor.
///
/// Call [`is_high()`] before eviction decisions to determine whether to be
/// aggressive.  If the platform is unsupported, [`is_high()`] always returns
/// `false`, meaning the pager relies entirely on the byte-budget to drive
/// eviction.
///
/// # Defaults
///
/// - `high_watermark = 0.90` — trigger aggressive eviction above 90% RSS.
/// - `low_watermark  = 0.75` — stop evicting once below 75% RSS.
///
/// [`is_high()`]: MemoryPressureProbe::is_high
#[derive(Debug, Clone)]
pub struct MemoryPressureProbe {
    /// RSS / total-RAM fraction above which pressure is considered high.
    pub high_watermark: f64,
    /// RSS / total-RAM fraction below which pressure is considered low
    /// (used by callers implementing hysteresis).
    pub low_watermark: f64,
}

impl Default for MemoryPressureProbe {
    fn default() -> Self {
        Self {
            high_watermark: 0.90,
            low_watermark: 0.75,
        }
    }
}

impl MemoryPressureProbe {
    /// Create a new probe with the given watermarks.
    ///
    /// # Panics (debug)
    ///
    /// Panics in debug builds if `low_watermark >= high_watermark`.
    pub fn new(high_watermark: f64, low_watermark: f64) -> Self {
        debug_assert!(
            low_watermark < high_watermark,
            "low_watermark ({low_watermark}) must be less than high_watermark ({high_watermark})"
        );
        Self {
            high_watermark,
            low_watermark,
        }
    }

    /// Returns `true` if the current RSS is at or above the high watermark.
    ///
    /// Returns `false` if the pressure level is unknown (unsupported platform).
    pub fn is_high(&self) -> bool {
        host_memory_pressure()
            .map(|p| p >= self.high_watermark)
            .unwrap_or(false)
    }

    /// Returns `true` if the current RSS is below the low watermark.
    ///
    /// Useful for hysteresis: stop evicting once [`is_low()`] returns `true`.
    ///
    /// Returns `true` if the pressure level is unknown (conservatively assume safe).
    ///
    /// [`is_low()`]: MemoryPressureProbe::is_low
    pub fn is_low(&self) -> bool {
        host_memory_pressure()
            .map(|p| p < self.low_watermark)
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_watermarks() {
        let probe = MemoryPressureProbe::default();
        assert!((probe.high_watermark - 0.90).abs() < 1e-9);
        assert!((probe.low_watermark - 0.75).abs() < 1e-9);
    }

    #[test]
    fn new_probe_stores_watermarks() {
        let probe = MemoryPressureProbe::new(0.85, 0.60);
        assert!((probe.high_watermark - 0.85).abs() < 1e-9);
        assert!((probe.low_watermark - 0.60).abs() < 1e-9);
    }

    #[test]
    fn host_memory_pressure_returns_option() {
        // We don't know the exact value; just verify it doesn't panic and
        // returns a value in [0.0, 1.0] when Some.
        if let Some(p) = host_memory_pressure() {
            assert!(
                (0.0..=1.0).contains(&p),
                "memory pressure {p} must be in [0.0, 1.0]"
            );
        }
    }

    #[test]
    fn is_high_does_not_panic() {
        let probe = MemoryPressureProbe::default();
        // Just ensure it returns without panic; value is OS-dependent.
        let _ = probe.is_high();
    }

    #[test]
    fn is_low_does_not_panic() {
        let probe = MemoryPressureProbe::default();
        let _ = probe.is_low();
    }

    #[test]
    fn probe_clone_is_independent() {
        let original = MemoryPressureProbe::new(0.95, 0.80);
        let cloned = original.clone();
        assert!((cloned.high_watermark - 0.95).abs() < 1e-9);
        assert!((cloned.low_watermark - 0.80).abs() < 1e-9);
    }
}
