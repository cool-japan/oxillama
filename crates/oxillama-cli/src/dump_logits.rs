// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `oxillama run --dump-logits <DIR>` — per-step raw logit dumps.
//!
//! This is a **debug / verification** facility, not a user-facing feature. It
//! exists so that OxiLLaMa's output can be compared numerically against another
//! implementation (llama.cpp being the one that matters) instead of against
//! OxiLLaMa's own scalar reference — a self-comparison proves nothing.
//!
//! # What is written
//!
//! For a run of `N` steps, `DIR` receives
//!
//! ```text
//! DIR/step0.logits.f32.bin   ... DIR/step<N-1>.logits.f32.bin
//! DIR/manifest.json
//! ```
//!
//! Each `step<k>.logits.f32.bin` is **headerless**: exactly `vocab_size`
//! consecutive IEEE-754 `binary32` values in little-endian order, index ==
//! token id, so the file is always `4 * vocab_size` bytes. The values are the
//! **raw** logits the forward pass produced — pre-softmax, before temperature,
//! repetition penalty, logit bias, grammar masking or any other sampler stage.
//! This is exactly the format `llama_get_logits_ith()` hands back in llama.cpp,
//! so the two can be diffed byte-position by byte-position.
//!
//! # Step semantics
//!
//! * `step0` holds the logits at position `n_prompt - 1`; its argmax is the
//!   first generated token.
//! * `step k` holds the logits at position `n_prompt - 1 + k`, produced after
//!   feeding generated token `k - 1`.
//!
//! # Deliberate deviations from the normal `run` loop
//!
//! * **End-of-generation does not stop the loop.** The full `--max-tokens`
//!   steps always run, and the first EOG token is recorded in the manifest as
//!   `eog_at_step`. Stopping early would renumber nothing but would leave the
//!   dump short, and a reference dump produced with the same protocol would no
//!   longer line up step-for-step.
//! * **Stop sequences are not applied.** They are a text-level concern and
//!   would truncate the dump for the same reason.
//!
//! # Teacher forcing
//!
//! With `--force-tokens`, the token fed into step `k + 1` is taken from the
//! supplied list instead of from the sampler. That makes step `k` of this dump
//! directly comparable with step `k` of a reference run even when the two
//! implementations would have chosen different tokens: both models then see
//! the identical prefix. Without it, a single divergent token makes every
//! later step incomparable.

use anyhow::{bail, Context, Result};
use oxillama_runtime::{InferenceEngine, Sampler, SamplerConfig};
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Schema version of the emitted `manifest.json`.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Where the prompt token ids came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromptSource {
    /// The model's tokenizer encoded a prompt string (BOS handling applied).
    Tokenizer,
    /// The caller supplied explicit token ids via `--prompt-tokens`.
    ExplicitTokenIds,
}

/// Description of the on-disk logit layout, copied into the manifest so that a
/// consumer never has to guess.
#[derive(Debug, Clone, Serialize)]
pub struct LogitsFormat {
    /// Element type.
    pub dtype: &'static str,
    /// Byte order of each element.
    pub endianness: &'static str,
    /// How elements map onto token ids.
    pub layout: &'static str,
    /// Number of elements per file.
    pub count: usize,
    /// Size of each file in bytes (`4 * count`).
    pub file_bytes: usize,
    /// Header description (there is none).
    pub header: &'static str,
    /// Reminder that no sampler stage has touched these numbers.
    pub note: &'static str,
}

impl LogitsFormat {
    /// Build the format block for a vocabulary of `vocab_size` entries.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            dtype: "float32",
            endianness: "little",
            layout: "flat, one value per vocab id, index == token id",
            count: vocab_size,
            file_bytes: vocab_size * 4,
            header: "none",
            note: "logits are RAW (pre-softmax, no temperature, no penalties, no grammar mask)",
        }
    }
}

/// One decode step's record inside the manifest.
#[derive(Debug, Clone, Serialize)]
pub struct StepRecord {
    /// Step index, `0`-based.
    pub step: usize,
    /// Sequence position these logits belong to (`n_prompt - 1 + step`).
    pub position: usize,
    /// File name, relative to the dump directory.
    pub file: String,
    /// Argmax over the dumped values, lowest index winning ties.
    pub argmax_token_id: u32,
    /// The argmax's logit value.
    pub max_logit: f32,
    /// Number of vocabulary entries tied at `max_logit`.
    pub n_argmax_ties: usize,
    /// The token actually emitted (differs from `argmax_token_id` when
    /// sampling is stochastic or when this step was teacher-forced).
    pub emitted_token_id: u32,
    /// `true` when `emitted_token_id` came from `--force-tokens`.
    pub forced: bool,
    /// `true` when `emitted_token_id` ends generation for this model.
    pub is_eog: bool,
}

/// The `manifest.json` written alongside the logit files.
#[derive(Debug, Clone, Serialize)]
pub struct DumpManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Producing binary and version.
    pub producer: String,
    /// GGUF file the logits came from.
    pub model_path: String,
    /// Architecture reported by the loaded model.
    pub architecture: String,
    /// Vocabulary size == number of values per logit file.
    pub vocab_size: usize,
    /// The prompt string, when one was tokenized (absent for
    /// `--prompt-tokens`).
    pub prompt: Option<String>,
    /// Where `prompt_token_ids` came from.
    pub prompt_source: PromptSource,
    /// Whether a BOS token was prepended to `prompt_token_ids`.
    ///
    /// This is observed from the produced ids, not merely copied from the
    /// tokenizer flag: it is `true` exactly when the model declares a BOS id
    /// and the first prompt token is it.
    pub add_bos_applied: bool,
    /// The model's BOS id, if it declares one.
    pub bos_token_id: Option<u32>,
    /// The model's primary EOS id, if it declares one.
    pub eos_token_id: Option<u32>,
    /// Number of prompt tokens.
    pub n_prompt_tokens: usize,
    /// The exact prompt token ids fed to the model.
    pub prompt_token_ids: Vec<u32>,
    /// Token ids emitted, one per step, in order.
    pub emitted_token_ids: Vec<u32>,
    /// Detokenized `emitted_token_ids`, for eyeballing.
    pub continuation: String,
    /// `true` when at least one step was teacher-forced.
    pub teacher_forced: bool,
    /// Index of the first emitted end-of-generation token, or `-1` if none.
    ///
    /// Recorded rather than obeyed — see the module docs.
    pub eog_at_step: i64,
    /// On-disk logit layout.
    pub logits_format: LogitsFormat,
    /// Per-step records.
    pub steps: Vec<StepRecord>,
}

/// Everything the dump loop needs that is not the token stream itself.
#[derive(Debug, Clone)]
pub struct DumpOptions {
    /// Directory to create and fill.
    pub out_dir: PathBuf,
    /// Number of decode steps to run.
    pub max_steps: usize,
    /// Sampler used for the steps that are not teacher-forced.
    pub sampler: SamplerConfig,
    /// Tokens to force, one per step, starting at step 0.
    pub forced_tokens: Vec<u32>,
    /// The prompt string, when one was tokenized.
    pub prompt: Option<String>,
    /// GGUF path, recorded in the manifest.
    pub model_path: String,
}

/// The forward-pass surface the dump loop needs.
///
/// Abstracting it keeps the loop's step/position/teacher-forcing semantics
/// testable without a multi-gigabyte GGUF on disk; [`EngineSource`] is the
/// only production implementation.
pub(crate) trait LogitSource {
    /// Vocabulary size, i.e. the expected logit-vector length.
    fn vocab_size(&self) -> usize;
    /// Reset any sequence state so the dump starts from position 0.
    fn reset(&mut self);
    /// Run the prompt through the model, returning the last position's logits.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    /// Run one decode step for `token` at sequence position `pos`.
    fn decode(&mut self, token: u32, pos: usize) -> Result<Vec<f32>>;
    /// `true` when `token` ends generation for this model.
    fn is_eog(&self, token: u32) -> bool;
}

/// [`LogitSource`] backed by a loaded [`InferenceEngine`].
pub(crate) struct EngineSource<'a> {
    engine: &'a mut InferenceEngine,
    vocab_size: usize,
    prefill_chunk_size: usize,
}

impl<'a> EngineSource<'a> {
    /// Wrap a loaded engine.
    ///
    /// # Errors
    ///
    /// Fails when no model has been loaded (there is no vocabulary yet).
    pub(crate) fn new(engine: &'a mut InferenceEngine) -> Result<Self> {
        let model_config = engine
            .model_config()
            .context("--dump-logits requires a loaded model")?;
        let vocab_size = model_config.vocab_size;
        let prefill_chunk_size = engine.config().prefill_chunk_size;
        Ok(Self {
            engine,
            vocab_size,
            prefill_chunk_size,
        })
    }
}

impl LogitSource for EngineSource<'_> {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn reset(&mut self) {
        self.engine.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Mirror the engine's own chunked prefill so that the dumped numbers
        // come from the same batching the normal `run` path would use: a
        // 512-token batch and a 1-token batch do not sum floats in the same
        // order, and this facility exists to measure exactly that kind of
        // difference elsewhere, not to introduce it here.
        let chunk = match self.prefill_chunk_size {
            0 => tokens.len(),
            n => n.min(tokens.len()),
        };
        let mut logits = Vec::new();
        let mut pos = 0usize;
        for batch in tokens.chunks(chunk) {
            logits = self
                .engine
                .forward_prefill(batch, pos)
                .with_context(|| format!("prefill forward pass at position {pos}"))?;
            pos += batch.len();
        }
        Ok(logits)
    }

    fn decode(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        self.engine
            .forward_decode(token, pos)
            .with_context(|| format!("decode forward pass for token {token} at position {pos}"))
    }

    fn is_eog(&self, token: u32) -> bool {
        self.engine.is_eos(token)
    }
}

/// Parse a comma-separated list of token ids.
///
/// Accepts surrounding and interior whitespace and an optional trailing comma,
/// so `"1, 2, 3"` and `"1,2,3,"` both parse. An empty (or whitespace-only)
/// string yields an empty vector rather than an error, which lets a caller
/// pass `--force-tokens ""` to mean "force nothing".
///
/// # Errors
///
/// Fails on any element that is not a base-10 `u32`.
pub fn parse_token_ids(spec: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for (index, field) in spec.split(',').enumerate() {
        let trimmed = field.trim();
        if trimmed.is_empty() {
            // Tolerate a trailing comma and an all-whitespace input; reject an
            // empty field wedged between two populated ones, which is a typo.
            if index == 0 || index + 1 == spec.split(',').count() {
                continue;
            }
            bail!("empty token id at position {index} in '{spec}'");
        }
        let id: u32 = trimmed
            .parse()
            .with_context(|| format!("token id '{trimmed}' at position {index} is not a u32"))?;
        ids.push(id);
    }
    Ok(ids)
}

/// Argmax over `values`, lowest index winning ties.
///
/// Returns the index, the value, and how many entries tie at that value.
/// Non-finite entries are skipped: a `NaN` compares false against everything
/// and would otherwise capture the argmax through a naive `>` scan. Returns
/// `None` when `values` is empty or entirely non-finite.
pub fn argmax_lowest_index(values: &[f32]) -> Option<(usize, f32, usize)> {
    let mut best_index = usize::MAX;
    let mut best_value = f32::NEG_INFINITY;
    let mut ties = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if best_index == usize::MAX || value > best_value {
            best_index = index;
            best_value = value;
            ties = 1;
        } else if value == best_value {
            ties += 1;
        }
    }
    if best_index == usize::MAX {
        None
    } else {
        Some((best_index, best_value, ties))
    }
}

/// Write one logit vector as headerless little-endian `f32`.
///
/// # Errors
///
/// Propagates any I/O failure, naming the file.
pub fn write_logits_file(path: &Path, logits: &[f32]) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating logit dump '{}'", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for value in logits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    writer
        .write_all(&bytes)
        .with_context(|| format!("writing logit dump '{}'", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing logit dump '{}'", path.display()))?;
    Ok(())
}

/// Run the dump loop against `source`, writing one file per step.
///
/// Returns the per-step records and the emitted token ids. The caller builds
/// the manifest around them (it owns the tokenizer needed for detokenization).
pub(crate) fn dump_steps(
    source: &mut dyn LogitSource,
    prompt_tokens: &[u32],
    out_dir: &Path,
    max_steps: usize,
    forced_tokens: &[u32],
    sampler_config: SamplerConfig,
) -> Result<(Vec<StepRecord>, Vec<u32>, i64)> {
    if prompt_tokens.is_empty() {
        bail!("--dump-logits needs a non-empty prompt");
    }
    if max_steps == 0 {
        bail!("--dump-logits needs at least one step (--max-tokens 0 was given)");
    }
    if forced_tokens.len() > max_steps {
        bail!(
            "--force-tokens has {} ids but only {max_steps} steps will run; \
             raise --max-tokens or shorten the list",
            forced_tokens.len()
        );
    }
    let vocab_size = source.vocab_size();
    for (index, &token) in prompt_tokens.iter().enumerate() {
        if token as usize >= vocab_size {
            bail!("prompt token id {token} at position {index} is outside the vocabulary (size {vocab_size})");
        }
    }
    for (index, &token) in forced_tokens.iter().enumerate() {
        if token as usize >= vocab_size {
            bail!("forced token id {token} for step {index} is outside the vocabulary (size {vocab_size})");
        }
    }

    let mut sampler = Sampler::new(sampler_config);
    let mut steps = Vec::with_capacity(max_steps);
    let mut emitted = Vec::with_capacity(max_steps);
    let mut recent = prompt_tokens.to_vec();
    let mut eog_at_step: i64 = -1;

    source.reset();
    let mut logits = source.prefill(prompt_tokens)?;
    let mut position = prompt_tokens.len() - 1;

    for step in 0..max_steps {
        if logits.len() != vocab_size {
            bail!(
                "step {step}: forward pass returned {} logits, expected {vocab_size}",
                logits.len()
            );
        }
        let file_name = format!("step{step}.logits.f32.bin");
        write_logits_file(&out_dir.join(&file_name), &logits)?;

        let (argmax_token_id, max_logit, n_argmax_ties) = argmax_lowest_index(&logits)
            .with_context(|| format!("step {step}: every logit is non-finite"))?;

        let forced = forced_tokens.get(step).copied();
        let emitted_token_id = match forced {
            Some(token) => token,
            None => sampler
                .try_sample(&logits, &recent)
                .with_context(|| format!("step {step}: sampling failed"))?,
        };
        let is_eog = source.is_eog(emitted_token_id);
        if is_eog && eog_at_step < 0 {
            eog_at_step = step as i64;
        }

        steps.push(StepRecord {
            step,
            position,
            file: file_name,
            argmax_token_id: argmax_token_id as u32,
            max_logit,
            n_argmax_ties,
            emitted_token_id,
            forced: forced.is_some(),
            is_eog,
        });
        emitted.push(emitted_token_id);
        recent.push(emitted_token_id);

        if step + 1 == max_steps {
            break;
        }
        position += 1;
        logits = source.decode(emitted_token_id, position)?;
    }

    Ok((steps, emitted, eog_at_step))
}

/// Run a full logit dump for `prompt_tokens` against a loaded engine.
///
/// Creates `options.out_dir` (including parents), writes one
/// `step<k>.logits.f32.bin` per step and a `manifest.json` describing them.
///
/// # Errors
///
/// Fails when no model is loaded, when the directory cannot be created, on any
/// forward-pass or I/O failure, or when an argument is out of range (empty
/// prompt, zero steps, more forced tokens than steps, token id ≥ vocab size).
pub fn run(
    engine: &mut InferenceEngine,
    prompt_tokens: &[u32],
    options: &DumpOptions,
) -> Result<DumpManifest> {
    std::fs::create_dir_all(&options.out_dir).with_context(|| {
        format!(
            "creating logit dump directory '{}'",
            options.out_dir.display()
        )
    })?;

    let architecture = engine
        .model_config()
        .map(|c| c.architecture.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let bos_token_id = engine.tokenizer().and_then(|t| t.bos_token_id());
    let eos_token_id = engine.eos_token_id();
    // Observed, not asserted: `add_bos_applied` describes the ids that were
    // actually fed to the model. With `--prompt-tokens` the tokenizer flag is
    // irrelevant, and a prompt that legitimately starts with the BOS piece is
    // indistinguishable from a prepended one — which is exactly what a
    // consumer of the dump cares about.
    let add_bos_applied = match (bos_token_id, prompt_tokens.first()) {
        (Some(bos), Some(&first)) => bos == first,
        _ => false,
    };

    let mut source = EngineSource::new(engine)?;
    let vocab_size = source.vocab_size();
    let (steps, emitted_token_ids, eog_at_step) = dump_steps(
        &mut source,
        prompt_tokens,
        &options.out_dir,
        options.max_steps,
        &options.forced_tokens,
        options.sampler.clone(),
    )?;

    // Special tokens are rendered rather than dropped: an EOG in the middle of
    // the continuation is a fact the reader needs to see.
    let continuation = match engine.tokenizer() {
        Some(tokenizer) => tokenizer
            .decode_with(&emitted_token_ids, false)
            .context("detokenizing the emitted tokens for the manifest")?,
        None => String::new(),
    };

    let manifest = DumpManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        producer: format!("oxillama {}", env!("CARGO_PKG_VERSION")),
        model_path: options.model_path.clone(),
        architecture,
        vocab_size,
        prompt: options.prompt.clone(),
        prompt_source: match options.prompt {
            Some(_) => PromptSource::Tokenizer,
            None => PromptSource::ExplicitTokenIds,
        },
        add_bos_applied,
        bos_token_id,
        eos_token_id,
        n_prompt_tokens: prompt_tokens.len(),
        prompt_token_ids: prompt_tokens.to_vec(),
        emitted_token_ids,
        continuation,
        teacher_forced: !options.forced_tokens.is_empty(),
        eog_at_step,
        logits_format: LogitsFormat::new(vocab_size),
        steps,
    };

    let manifest_path = options.out_dir.join("manifest.json");
    let json =
        serde_json::to_string_pretty(&manifest).context("serializing the logit dump manifest")?;
    std::fs::write(&manifest_path, json)
        .with_context(|| format!("writing '{}'", manifest_path.display()))?;

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic stand-in for the forward pass.
    ///
    /// Logits are a pure function of the token that produced them, so a test
    /// can assert precisely which token was fed at which step.
    struct FakeSource {
        vocab_size: usize,
        /// Tokens handed to `decode`, in order.
        decoded: Vec<(u32, usize)>,
        /// Tokens handed to `prefill`.
        prefilled: Vec<u32>,
        /// Number of `reset` calls.
        resets: usize,
        /// Ids treated as end-of-generation.
        eog: Vec<u32>,
    }

    impl FakeSource {
        fn new(vocab_size: usize) -> Self {
            Self {
                vocab_size,
                decoded: Vec::new(),
                prefilled: Vec::new(),
                resets: 0,
                eog: Vec::new(),
            }
        }

        /// Logits whose argmax is `(seed * 7 + 3) % vocab_size`.
        fn logits_for(&self, seed: u32) -> Vec<f32> {
            let mut logits = vec![0.0f32; self.vocab_size];
            let winner = ((seed as usize) * 7 + 3) % self.vocab_size;
            logits[winner] = 10.0;
            logits
        }
    }

    impl LogitSource for FakeSource {
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
        fn reset(&mut self) {
            self.resets += 1;
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            self.prefilled = tokens.to_vec();
            Ok(self.logits_for(tokens[tokens.len() - 1]))
        }
        fn decode(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
            self.decoded.push((token, pos));
            Ok(self.logits_for(token))
        }
        fn is_eog(&self, token: u32) -> bool {
            self.eog.contains(&token)
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oxillama_dump_logits_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn parse_token_ids_accepts_plain_and_spaced_lists() {
        assert_eq!(parse_token_ids("1,2,3").expect("parse"), vec![1, 2, 3]);
        assert_eq!(parse_token_ids(" 1, 2 ,3 ").expect("parse"), vec![1, 2, 3]);
        assert_eq!(parse_token_ids("1,2,3,").expect("parse"), vec![1, 2, 3]);
        assert_eq!(parse_token_ids("").expect("parse"), Vec::<u32>::new());
        assert_eq!(parse_token_ids("   ").expect("parse"), Vec::<u32>::new());
        assert_eq!(parse_token_ids("128000").expect("parse"), vec![128_000]);
    }

    #[test]
    fn parse_token_ids_rejects_junk() {
        assert!(parse_token_ids("1,x,3").is_err());
        assert!(parse_token_ids("-1").is_err());
        assert!(parse_token_ids("1,,3").is_err());
        assert!(parse_token_ids("4294967296").is_err(), "u32 overflow");
    }

    #[test]
    fn argmax_breaks_ties_toward_the_lowest_index() {
        // llama.cpp's greedy sampler keeps the first maximum it sees, so a tie
        // must resolve to the lowest id or the two implementations disagree on
        // a step where the logits are actually identical.
        let (index, value, ties) =
            argmax_lowest_index(&[1.0, 5.0, 5.0, 2.0]).expect("finite input");
        assert_eq!(index, 1);
        assert_eq!(value, 5.0);
        assert_eq!(ties, 2);
    }

    #[test]
    fn argmax_skips_non_finite_entries() {
        let (index, value, ties) = argmax_lowest_index(&[f32::NAN, 1.0, f32::INFINITY, 2.0])
            .expect("finite entries exist");
        assert_eq!(index, 3);
        assert_eq!(value, 2.0);
        assert_eq!(ties, 1);
        assert!(argmax_lowest_index(&[]).is_none());
        assert!(argmax_lowest_index(&[f32::NAN, f32::NAN]).is_none());
    }

    #[test]
    fn write_logits_file_is_headerless_little_endian_f32() {
        let dir = temp_dir("write_format");
        let path = dir.join("step0.logits.f32.bin");
        let logits = vec![1.0f32, -2.5, 0.0, f32::MIN_POSITIVE];
        write_logits_file(&path, &logits).expect("write");
        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(bytes.len(), logits.len() * 4, "exactly 4 bytes per value");
        for (index, expected) in logits.iter().enumerate() {
            let mut quad = [0u8; 4];
            quad.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
            assert_eq!(f32::from_le_bytes(quad), *expected);
        }
        // First value's low byte must be the file's first byte: little endian.
        assert_eq!(bytes[0], 1.0f32.to_le_bytes()[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dump_steps_writes_one_file_per_step_with_correct_positions() {
        let dir = temp_dir("step_positions");
        let mut source = FakeSource::new(64);
        let prompt = [5u32, 6, 7];
        let (steps, emitted, eog) =
            dump_steps(&mut source, &prompt, &dir, 4, &[], SamplerConfig::greedy()).expect("dump");

        assert_eq!(source.resets, 1, "sequence state reset exactly once");
        assert_eq!(source.prefilled, prompt.to_vec());
        assert_eq!(steps.len(), 4);
        assert_eq!(emitted.len(), 4);
        assert_eq!(eog, -1, "no EOG token was emitted");

        // step0 sits at n_prompt - 1 and each later step advances by one.
        for (index, record) in steps.iter().enumerate() {
            assert_eq!(record.step, index);
            assert_eq!(record.position, prompt.len() - 1 + index);
            assert_eq!(record.file, format!("step{index}.logits.f32.bin"));
            let path = dir.join(&record.file);
            let bytes = std::fs::read(&path).expect("step file exists");
            assert_eq!(bytes.len(), source.vocab_size * 4);
        }

        // Greedy sampling means the emitted token is the argmax, and decode
        // must be fed exactly that token at the following position.
        for record in &steps {
            assert_eq!(record.emitted_token_id, record.argmax_token_id);
            assert!(!record.forced);
        }
        let expected_decodes: Vec<(u32, usize)> = emitted[..3]
            .iter()
            .enumerate()
            .map(|(index, &token)| (token, prompt.len() + index))
            .collect();
        assert_eq!(source.decoded, expected_decodes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dump_steps_teacher_forces_the_supplied_tokens() {
        let dir = temp_dir("teacher_forcing");
        let mut source = FakeSource::new(64);
        let prompt = [1u32, 2];
        let forced = [40u32, 41, 42];
        let (steps, emitted, _) = dump_steps(
            &mut source,
            &prompt,
            &dir,
            3,
            &forced,
            SamplerConfig::greedy(),
        )
        .expect("dump");

        assert_eq!(emitted, forced.to_vec());
        assert!(steps.iter().all(|s| s.forced));
        // The argmax is recorded even though it was overridden — that is the
        // whole point: it is what a free-running run would have picked.
        assert!(
            steps
                .iter()
                .any(|s| s.argmax_token_id != s.emitted_token_id),
            "the fake's argmax should differ from the forced ids"
        );
        // Forcing must reach the model: decode sees the forced tokens.
        assert_eq!(
            source.decoded,
            vec![(forced[0], prompt.len()), (forced[1], prompt.len() + 1)]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dump_steps_records_eog_without_stopping() {
        let dir = temp_dir("eog_recorded");
        let mut source = FakeSource::new(64);
        source.eog = vec![9];
        let prompt = [1u32];
        let (steps, emitted, eog) = dump_steps(
            &mut source,
            &prompt,
            &dir,
            4,
            &[3, 9, 5, 6],
            SamplerConfig::greedy(),
        )
        .expect("dump");
        assert_eq!(eog, 1, "EOG was emitted at step 1");
        assert_eq!(emitted.len(), 4, "the loop runs all 4 steps regardless");
        assert!(steps[1].is_eog);
        assert!(!steps[2].is_eog);
        assert!(dir.join("step3.logits.f32.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dump_steps_rejects_out_of_range_and_oversized_inputs() {
        let dir = temp_dir("rejects");
        let mut source = FakeSource::new(16);
        assert!(
            dump_steps(&mut source, &[], &dir, 2, &[], SamplerConfig::greedy()).is_err(),
            "empty prompt"
        );
        assert!(
            dump_steps(&mut source, &[1], &dir, 0, &[], SamplerConfig::greedy()).is_err(),
            "zero steps"
        );
        assert!(
            dump_steps(&mut source, &[1], &dir, 1, &[1, 2], SamplerConfig::greedy()).is_err(),
            "more forced tokens than steps"
        );
        assert!(
            dump_steps(&mut source, &[99], &dir, 1, &[], SamplerConfig::greedy()).is_err(),
            "prompt token beyond the vocabulary"
        );
        assert!(
            dump_steps(&mut source, &[1], &dir, 1, &[99], SamplerConfig::greedy()).is_err(),
            "forced token beyond the vocabulary"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_serializes_the_fields_a_comparison_tool_needs() {
        let manifest = DumpManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            producer: "oxillama test".to_string(),
            model_path: "model.gguf".to_string(),
            architecture: "llama".to_string(),
            vocab_size: 4,
            prompt: Some("hi".to_string()),
            prompt_source: PromptSource::Tokenizer,
            add_bos_applied: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            n_prompt_tokens: 2,
            prompt_token_ids: vec![1, 3],
            emitted_token_ids: vec![2],
            continuation: "!".to_string(),
            teacher_forced: false,
            eog_at_step: 0,
            logits_format: LogitsFormat::new(4),
            steps: vec![StepRecord {
                step: 0,
                position: 1,
                file: "step0.logits.f32.bin".to_string(),
                argmax_token_id: 2,
                max_logit: 1.5,
                n_argmax_ties: 1,
                emitted_token_id: 2,
                forced: false,
                is_eog: true,
            }],
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse back");
        assert_eq!(parsed["prompt_token_ids"], serde_json::json!([1, 3]));
        assert_eq!(parsed["add_bos_applied"], serde_json::json!(true));
        assert_eq!(parsed["emitted_token_ids"], serde_json::json!([2]));
        assert_eq!(parsed["vocab_size"], serde_json::json!(4));
        assert_eq!(parsed["prompt_source"], serde_json::json!("tokenizer"));
        assert_eq!(parsed["logits_format"]["file_bytes"], serde_json::json!(16));
        assert_eq!(parsed["steps"][0]["position"], serde_json::json!(1));
    }
}
