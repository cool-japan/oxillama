//! Configuration schema and loading for OxiLLaMa CLI.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level global configuration for OxiLLaMa.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct OxillamaConfig {
    /// Default model path used when `--model` is not specified.
    pub default_model: Option<PathBuf>,
    /// Default context window size.
    pub default_ctx_size: Option<usize>,
    /// Default thread count.
    pub default_threads: Option<usize>,
    /// Default sampling temperature.
    pub default_temp: Option<f32>,
    /// Log level filter (e.g. `"info"`, `"debug"`).
    pub log_level: Option<String>,
}

/// Per-model sampler profile loaded from `~/.config/oxillama/models/<stem>.toml`.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct ModelProfile {
    /// Context window size override for this model.
    pub ctx_size: Option<usize>,
    /// Thread count override for this model.
    pub threads: Option<usize>,
    /// Sampling temperature override for this model.
    pub temp: Option<f32>,
    /// Top-P nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Top-K sampling limit.
    pub top_k: Option<u32>,
    /// Fixed seed for reproducible output.
    pub seed: Option<u64>,
    /// System prompt prepended to every conversation turn.
    pub system_prompt: Option<String>,
}

/// Resolve the config directory (`~/.config/oxillama/`).
fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("oxillama"))
}

/// Load the global config.
///
/// Priority (highest to lowest):
/// 1. `OXILLAMA_CONFIG` environment variable path
/// 2. `path` argument (from `--config <path>`)
/// 3. `~/.config/oxillama/config.toml`
/// 4. [`OxillamaConfig::default()`]
pub fn load_config(path: Option<PathBuf>) -> Result<OxillamaConfig> {
    // 1. Env var wins.
    if let Ok(env_path) = std::env::var("OXILLAMA_CONFIG") {
        let p = PathBuf::from(&env_path);
        return read_config_toml(&p)
            .with_context(|| format!("loading config from OXILLAMA_CONFIG={env_path}"));
    }

    // 2. Explicit CLI flag.
    if let Some(p) = path {
        return read_config_toml(&p)
            .with_context(|| format!("loading config from --config {}", p.display()));
    }

    // 3. Default location.
    if let Some(default_path) = config_dir().map(|d| d.join("config.toml")) {
        if default_path.exists() {
            return read_config_toml(&default_path)
                .with_context(|| format!("loading config from {}", default_path.display()));
        }
    }

    // 4. Empty defaults.
    Ok(OxillamaConfig::default())
}

/// Load the per-model profile for the given model file stem, if one exists.
///
/// Reads `~/.config/oxillama/models/<model_stem>.toml`.
pub fn load_model_profile(model_stem: &str) -> Result<Option<ModelProfile>> {
    let profile_path = match config_dir() {
        Some(d) => d.join("models").join(format!("{model_stem}.toml")),
        None => return Ok(None),
    };

    if !profile_path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(&profile_path)
        .with_context(|| format!("reading model profile {}", profile_path.display()))?;
    let profile: ModelProfile = toml::from_str(&raw)
        .with_context(|| format!("parsing model profile {}", profile_path.display()))?;
    Ok(Some(profile))
}

fn read_config_toml(path: &std::path::Path) -> Result<OxillamaConfig> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parsing TOML from {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_config_default_when_no_file() {
        // Unset env and pass no path; should return default.
        std::env::remove_var("OXILLAMA_CONFIG");
        let cfg = load_config(None).expect("load_config should not fail");
        assert!(cfg.default_model.is_none());
        assert!(cfg.default_ctx_size.is_none());
    }

    #[test]
    fn test_load_config_from_explicit_path() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_test_config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "default_ctx_size = 8192").unwrap();
        writeln!(f, "default_threads = 8").unwrap();

        std::env::remove_var("OXILLAMA_CONFIG");
        let cfg = load_config(Some(path.clone())).expect("should parse");
        assert_eq!(cfg.default_ctx_size, Some(8192));
        assert_eq!(cfg.default_threads, Some(8));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_config_from_env_var() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxillama_test_env_config.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "log_level = \"debug\"").unwrap();

        std::env::set_var("OXILLAMA_CONFIG", path.to_str().unwrap());
        let cfg = load_config(None).expect("should parse from env");
        assert_eq!(cfg.log_level.as_deref(), Some("debug"));
        std::env::remove_var("OXILLAMA_CONFIG");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_model_profile_missing() {
        let result = load_model_profile("nonexistent-model-xyz").expect("should not error");
        assert!(result.is_none());
    }

    #[test]
    fn test_load_model_profile_exists() {
        // Write a profile into temp dir and monkey-patch via a direct call.
        let dir = std::env::temp_dir();
        let profile_path = dir.join("test_profile_oxillama.toml");
        let mut f = std::fs::File::create(&profile_path).unwrap();
        writeln!(f, "temp = 0.5").unwrap();
        writeln!(f, "top_k = 20").unwrap();

        // Read directly (bypassing load_model_profile path resolution).
        let raw = std::fs::read_to_string(&profile_path).unwrap();
        let profile: ModelProfile = toml::from_str(&raw).unwrap();
        assert_eq!(profile.temp, Some(0.5_f32));
        assert_eq!(profile.top_k, Some(20));
        std::fs::remove_file(&profile_path).ok();
    }
}
