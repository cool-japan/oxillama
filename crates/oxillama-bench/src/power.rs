//! Linux RAPL (Running Average Power Limit) power reader.
//!
//! Reads accumulated energy from the Intel RAPL sysfs interface:
//! `/sys/class/powercap/intel-rapl:*` — package-level energy counters in
//! microjoules.  Handles counter wraparound via `max_energy_range_uj`.
//!
//! On non-Linux platforms, or when the RAPL sysfs tree is absent (no Intel CPU,
//! no `intel_rapl` kernel module loaded, or insufficient permissions),
//! `RaplReader::open()` returns `Err(PowerError::NoRapl(...))`.  All callers
//! should treat a `NoRapl` result as a graceful "not available" rather than a
//! hard error.
//!
//! # Linux permissions
//!
//! Reading `/sys/class/powercap/intel-rapl:*/energy_uj` requires either:
//! - root (`uid=0`), or
//! - the `CAP_SYS_RAWIO` capability, or
//! - a kernel built with `CONFIG_POWERCAP_INTEL_RAPL=y` **and** the owning
//!   user has read permissions (some distros allow this via udev rules).
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxillama_bench::power::{RaplReader, measure_tokens_per_joule};
//!
//! match RaplReader::open() {
//!     Ok(rapl) => {
//!         let (tokens, tpj) = measure_tokens_per_joule(&rapl, || {
//!             // run your inference here and return token count
//!             128
//!         }).expect("RAPL read failed");
//!         println!("{tokens} tokens at {tpj:.1} tok/J");
//!     }
//!     Err(e) => eprintln!("RAPL unavailable: {e}"),
//! }
//! ```

use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::time::SystemTime;

// ── Error types ────────────────────────────────────────────────────────────────

/// Errors that can arise when reading RAPL energy counters.
#[derive(Debug)]
pub enum PowerError {
    /// RAPL is unavailable: not Linux, no driver loaded, or permission denied.
    NoRapl(String),
    /// An I/O error occurred while reading an energy file.
    ReadError(std::io::Error),
    /// The energy file contained text that could not be parsed as `u64`.
    ParseError(String),
}

impl std::fmt::Display for PowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PowerError::NoRapl(msg) => write!(f, "RAPL unavailable: {msg}"),
            PowerError::ReadError(e) => write!(f, "RAPL read error: {e}"),
            PowerError::ParseError(msg) => write!(f, "RAPL parse error: {msg}"),
        }
    }
}

impl std::error::Error for PowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PowerError::ReadError(e) => Some(e),
            PowerError::NoRapl(_) | PowerError::ParseError(_) => None,
        }
    }
}

/// Convenience `Result` alias for RAPL power operations.
pub type PowerResult<T> = Result<T, PowerError>;

// ── Data types ─────────────────────────────────────────────────────────────────

/// A single energy snapshot across all available RAPL domains.
///
/// The `package_uj` field is the **sum** of all discovered package-domain
/// `energy_uj` counters.  Sub-domain counters (e.g. `intel-rapl:0:0` for the
/// uncore/DRAM ring) are intentionally excluded so that energy is not
/// double-counted when both the package domain and its children are visible.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnergyReading {
    /// Summed raw `energy_uj` value across all top-level RAPL package domains.
    pub package_uj: u64,
    /// Wall-clock time of this sample, as nanoseconds since the Unix epoch.
    pub timestamp_ns: u64,
}

impl EnergyReading {
    /// Construct a reading from raw fields (used by tests and internal helpers).
    pub fn new(package_uj: u64, timestamp_ns: u64) -> Self {
        Self {
            package_uj,
            timestamp_ns,
        }
    }
}

// ── RaplReader ─────────────────────────────────────────────────────────────────

/// Reads accumulated energy from Linux RAPL sysfs package domains.
///
/// Created via [`RaplReader::open()`]; holds the discovered file paths and the
/// wraparound limit so that [`delta_uj`][RaplReader::delta_uj] can handle
/// counter resets transparently.
pub struct RaplReader {
    /// Paths to each top-level domain's `energy_uj` file.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    domain_paths: Vec<PathBuf>,
    /// Wraparound limit read once from `max_energy_range_uj` on open.
    pub max_energy_uj: u64,
}

impl RaplReader {
    /// Open RAPL readers for all available top-level package domains.
    ///
    /// Scans `/sys/class/powercap/` for directories matching `intel-rapl:<N>`
    /// (top-level packages only; sub-domains like `intel-rapl:0:0` are
    /// skipped).  Reads `max_energy_range_uj` from the first discovered domain
    /// as the shared wraparound limit.
    ///
    /// # Errors
    ///
    /// - `PowerError::NoRapl` if not running on Linux, or the sysfs path does
    ///   not exist, or no domains are found, or permissions are denied.
    /// - `PowerError::ReadError` / `PowerError::ParseError` for I/O issues.
    pub fn open() -> PowerResult<Self> {
        Self::open_impl()
    }

    /// Platform-gated implementation: only compiled on Linux.
    #[cfg(target_os = "linux")]
    fn open_impl() -> PowerResult<Self> {
        use std::fs;

        let powercap_dir = std::path::Path::new("/sys/class/powercap");
        if !powercap_dir.exists() {
            return Err(PowerError::NoRapl(
                "sysfs powercap directory not found; intel_rapl module may not be loaded"
                    .to_string(),
            ));
        }

        let read_dir = fs::read_dir(powercap_dir).map_err(|e| {
            PowerError::NoRapl(format!(
                "cannot read /sys/class/powercap: {e} (check permissions)"
            ))
        })?;

        // Collect top-level `intel-rapl:<N>` domain directories; skip sub-domains
        // (which have the form `intel-rapl:<N>:<M>`).
        let mut domain_paths: Vec<PathBuf> = Vec::new();

        for entry_result in read_dir {
            let entry = entry_result.map_err(PowerError::ReadError)?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Accept only top-level package domains: exactly one colon.
            if name_str.starts_with("intel-rapl:") {
                let colon_count = name_str.chars().filter(|&c| c == ':').count();
                if colon_count == 1 {
                    let energy_path = entry.path().join("energy_uj");
                    // Try to read the file to verify permissions before
                    // storing the path.
                    fs::read_to_string(&energy_path).map_err(|e| {
                        PowerError::NoRapl(format!(
                            "cannot read {}: {e} (try running as root or check udev rules)",
                            energy_path.display()
                        ))
                    })?;
                    domain_paths.push(energy_path);
                }
            }
        }

        if domain_paths.is_empty() {
            return Err(PowerError::NoRapl(
                "no intel-rapl package domains found under /sys/class/powercap/".to_string(),
            ));
        }

        // Sort for stable ordering across runs.
        domain_paths.sort();

        // Read max_energy_range_uj from the first domain.  All Intel package
        // domains share the same 32-bit or 64-bit counter width, so using the
        // first is representative.
        let first_domain_dir = domain_paths[0]
            .parent()
            .ok_or_else(|| PowerError::ParseError("energy_uj path has no parent".to_string()))?;
        let max_path = first_domain_dir.join("max_energy_range_uj");

        let max_energy_uj = if max_path.exists() {
            let raw = fs::read_to_string(&max_path).map_err(PowerError::ReadError)?;
            raw.trim()
                .parse::<u64>()
                .map_err(|e| PowerError::ParseError(format!("max_energy_range_uj: {e}")))?
        } else {
            // Fallback: Intel's RAPL counter is documented as 32-bit (wraps at
            // 2^32 µJ ≈ 4294 J ≈ ~1.2 Wh).  Use u32::MAX as a safe fallback.
            u32::MAX as u64
        };

        Ok(Self {
            domain_paths,
            max_energy_uj,
        })
    }

    /// Non-Linux stub: always returns `NoRapl`.
    #[cfg(not(target_os = "linux"))]
    fn open_impl() -> PowerResult<Self> {
        Err(PowerError::NoRapl(
            "RAPL power reading is only supported on Linux".to_string(),
        ))
    }

    /// Sample the current summed energy across all discovered domains.
    ///
    /// Reads each domain's `energy_uj` file and sums the values.  Captures
    /// `SystemTime::now()` immediately after the last file read for correlation
    /// with wall-clock time.
    pub fn read_energy_uj(&self) -> PowerResult<EnergyReading> {
        self.read_energy_uj_impl()
    }

    /// Platform-gated implementation: only compiled on Linux.
    #[cfg(target_os = "linux")]
    fn read_energy_uj_impl(&self) -> PowerResult<EnergyReading> {
        let mut total_uj: u64 = 0;

        for path in &self.domain_paths {
            let raw = std::fs::read_to_string(path).map_err(PowerError::ReadError)?;
            let val: u64 = raw
                .trim()
                .parse()
                .map_err(|e| PowerError::ParseError(format!("{}: {e}", path.display())))?;
            total_uj = total_uj.saturating_add(val);
        }

        let timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Ok(EnergyReading {
            package_uj: total_uj,
            timestamp_ns,
        })
    }

    /// Non-Linux stub: cannot reach here (open() always fails on non-Linux).
    #[cfg(not(target_os = "linux"))]
    fn read_energy_uj_impl(&self) -> PowerResult<EnergyReading> {
        Err(PowerError::NoRapl(
            "RAPL power reading is only supported on Linux".to_string(),
        ))
    }

    /// Compute the energy delta in microjoules between two readings.
    ///
    /// Handles counter wraparound: when `after.package_uj < before.package_uj`
    /// the counter has wrapped around at `max_uj` and the true delta is
    /// `max_uj - before + after + 1`.
    ///
    /// # Arguments
    ///
    /// - `before` — the earlier energy snapshot.
    /// - `after`  — the later energy snapshot.
    /// - `max_uj` — the wraparound limit (from [`RaplReader::max_energy_uj`]).
    pub fn delta_uj(before: &EnergyReading, after: &EnergyReading, max_uj: u64) -> u64 {
        compute_delta_uj(before.package_uj, after.package_uj, max_uj)
    }
}

/// Core wraparound-safe delta computation (also used in tests via the public
/// helper `delta_uj_raw`).
///
/// Computes `after - before` modulo `max_uj + 1`, handling the case where
/// the counter has wrapped.
pub(crate) fn compute_delta_uj(before_uj: u64, after_uj: u64, max_uj: u64) -> u64 {
    if after_uj >= before_uj {
        after_uj - before_uj
    } else {
        // Wraparound: counter reset to 0 and continued counting.
        // Total energy = energy from `before` to `max_uj` + energy from 0 to `after`.
        // That equals: (max_uj - before_uj) + after_uj + 1
        // The "+1" accounts for the counter ticking from max_uj to 0.
        (max_uj - before_uj)
            .saturating_add(after_uj)
            .saturating_add(1)
    }
}

// ── tokens-per-joule measurement ──────────────────────────────────────────────

/// Measure the tokens-per-joule efficiency of a synchronous benchmark closure.
///
/// Reads the RAPL energy before and after calling `f`, then computes:
///
/// ```text
/// tokens_per_joule = tokens / (delta_uj / 1_000_000)
///                  = tokens * 1_000_000 / delta_uj
/// ```
///
/// Returns `(tokens_produced, tokens_per_joule)`.
///
/// # Errors
///
/// - `PowerError::ReadError` / `PowerError::ParseError` if the RAPL counters
///   cannot be read.
/// - `PowerError::NoRapl` if called on a non-Linux platform (but `open()`
///   would have failed already in that case).
///
/// If `delta_uj` is zero (the closure finished before the counter ticked),
/// `tokens_per_joule` is returned as `f64::INFINITY`.
pub fn measure_tokens_per_joule<F: FnOnce() -> usize>(
    rapl: &RaplReader,
    f: F,
) -> PowerResult<(usize, f64)> {
    let before = rapl.read_energy_uj()?;
    let tokens = f();
    let after = rapl.read_energy_uj()?;

    let delta = RaplReader::delta_uj(&before, &after, rapl.max_energy_uj);
    let tokens_per_joule = compute_tokens_per_joule_from_delta(tokens, delta);

    Ok((tokens, tokens_per_joule))
}

/// Compute tokens/joule given a token count and an energy delta in µJ.
///
/// Exposed as a standalone function so that unit tests can inject pre-computed
/// delta values without needing access to real RAPL hardware.
///
/// Returns `f64::INFINITY` when `delta_uj` is zero.
pub fn compute_tokens_per_joule_from_delta(tokens: usize, delta_uj: u64) -> f64 {
    if delta_uj == 0 {
        return f64::INFINITY;
    }
    // 1 J = 1_000_000 µJ
    tokens as f64 * 1_000_000.0 / delta_uj as f64
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: non-Linux (or missing sysfs) returns NoRapl ───────────────────

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn rapl_unavailable_on_non_linux_returns_no_rapl() {
        let result = RaplReader::open();
        assert!(
            result.is_err(),
            "RaplReader::open() must fail on non-Linux platforms"
        );
        match result {
            Err(PowerError::NoRapl(_)) => {}
            Err(other) => panic!("expected NoRapl, got: {other}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    /// On Linux without RAPL hardware (CI VMs typically lack real RAPL),
    /// we still expect `NoRapl` (sysfs absent or permission denied).
    /// If somehow RAPL *is* available, the test is vacuously satisfied.
    #[test]
    #[cfg(target_os = "linux")]
    fn rapl_unavailable_on_non_linux_returns_no_rapl() {
        // On a Linux machine without RAPL (VM / container), open() should
        // return NoRapl.  On a real machine with RAPL it may succeed — that
        // is also fine; the test just verifies we never panic or return an
        // unexpected error variant.
        match RaplReader::open() {
            Ok(_) => {}                      // RAPL available — test passes vacuously.
            Err(PowerError::NoRapl(_)) => {} // No RAPL — expected in CI.
            Err(PowerError::ReadError(e)) => {
                // Permission denied is acceptable; the real check is no panic.
                eprintln!("RAPL read error (acceptable in CI): {e}");
            }
            Err(PowerError::ParseError(msg)) => {
                panic!("unexpected ParseError: {msg}");
            }
        }
    }

    // ── Test 2: delta without wraparound ──────────────────────────────────────

    #[test]
    fn rapl_delta_no_wraparound() {
        let before = EnergyReading::new(100, 0);
        let after = EnergyReading::new(200, 1);
        let delta = RaplReader::delta_uj(&before, &after, u64::MAX);
        assert_eq!(
            delta, 100,
            "delta without wraparound must be after - before"
        );
    }

    // ── Test 3: delta with wraparound ─────────────────────────────────────────

    #[test]
    fn rapl_delta_with_wraparound() {
        // before = u64::MAX - 5, after = 10, max = u64::MAX
        // Expected delta = (u64::MAX - (u64::MAX - 5)) + 10 + 1 = 5 + 10 + 1 = 16
        let before_uj = u64::MAX - 5;
        let after_uj = 10_u64;
        let max = u64::MAX;
        let delta = compute_delta_uj(before_uj, after_uj, max);
        assert_eq!(delta, 16, "wraparound delta should be 5 + 10 + 1 = 16");
    }

    // ── Test 4: tokens-per-joule formula ──────────────────────────────────────

    /// Verify the tokens/joule formula directly without real RAPL hardware.
    ///
    /// Given: 1_000_000 µJ consumed (= 1 J) and 100 tokens produced,
    /// the expected result is exactly 100.0 tokens/joule.
    #[test]
    fn tokens_per_joule_formula() {
        let delta_uj: u64 = 1_000_000; // exactly 1 joule
        let tokens: usize = 100;
        let tpj = compute_tokens_per_joule_from_delta(tokens, delta_uj);
        assert!(
            (tpj - 100.0).abs() < 1e-9,
            "expected 100.0 tok/J, got {tpj}"
        );
    }

    // ── Additional edge-case coverage ─────────────────────────────────────────

    #[test]
    fn tokens_per_joule_zero_delta_is_infinity() {
        let tpj = compute_tokens_per_joule_from_delta(50, 0);
        assert!(
            tpj.is_infinite(),
            "zero energy delta must yield infinite tok/J"
        );
    }

    #[test]
    fn tokens_per_joule_zero_tokens() {
        let tpj = compute_tokens_per_joule_from_delta(0, 1_000_000);
        assert!(
            (tpj - 0.0).abs() < f64::EPSILON,
            "zero tokens must yield 0.0 tok/J, got {tpj}"
        );
    }

    #[test]
    fn delta_uj_equal_values_is_zero() {
        let before = EnergyReading::new(500_000, 0);
        let after = EnergyReading::new(500_000, 100);
        let delta = RaplReader::delta_uj(&before, &after, u64::MAX);
        assert_eq!(delta, 0, "equal readings should produce zero delta");
    }

    #[test]
    fn delta_uj_wraparound_from_zero() {
        // before = 0, after = 5, no wraparound (normal case)
        let delta = compute_delta_uj(0, 5, 1000);
        assert_eq!(delta, 5);
    }

    #[test]
    fn delta_uj_wraparound_at_exact_max() {
        // before = max, after = 0 → wrapped exactly once
        // (max - max) + 0 + 1 = 1
        let max: u64 = 1000;
        let delta = compute_delta_uj(max, 0, max);
        assert_eq!(delta, 1, "wrapping from exact max to 0 should give 1");
    }

    #[test]
    fn power_error_display_no_rapl() {
        let e = PowerError::NoRapl("test message".to_string());
        let s = e.to_string();
        assert!(
            s.contains("unavailable"),
            "NoRapl display must say unavailable"
        );
        assert!(s.contains("test message"));
    }

    #[test]
    fn power_error_display_parse_error() {
        let e = PowerError::ParseError("bad value".to_string());
        let s = e.to_string();
        assert!(s.contains("parse"), "ParseError display must say parse");
    }

    #[test]
    fn power_error_display_read_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = PowerError::ReadError(io_err);
        let s = e.to_string();
        assert!(s.contains("read"), "ReadError display must say read");
    }
}
