//! Crate-wide error type.
//!
//! All `pub` fallible APIs in `scrapbox` return [`Result<T>`] where the error
//! variant is [`ScrapboxError`]. Per `notes/discipline/CONVENTIONS.md`,
//! variants carry enough information to diagnose the failure without
//! re-running.

use std::path::PathBuf;

/// Convenience `Result` alias for crate-wide use.
pub type Result<T, E = ScrapboxError> = std::result::Result<T, E>;

/// Crate-wide error type for the scrapbox library and CLI.
#[derive(Debug, thiserror::Error)]
pub enum ScrapboxError {
    /// A `config.toml` file could not be read from disk.
    #[error("failed to read config file {path}: {source}")]
    ConfigRead {
        /// The path the loader attempted to read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// A `config.toml` file failed to deserialize against the schema.
    ///
    /// Carries the original TOML error so the caller can surface line/column
    /// information to the user (see `notes/discipline/CONFIG.md` §schema-rules).
    #[error("failed to parse config file: {source}")]
    ConfigParse {
        /// The underlying `toml` deserialization error.
        #[source]
        source: toml::de::Error,
    },

    /// The config's `schema_version` does not match what this build supports.
    #[error("unsupported schema_version {found:?} (this build supports {supported:?})")]
    SchemaVersionMismatch {
        /// The value read from the config.
        found: String,
        /// The version this build targets (compile-time constant).
        supported: &'static str,
    },

    /// A semantic rule of the schema was violated
    /// (e.g. `num_electrons_per_spin > num_sites`).
    #[error("config validation failed: {message}")]
    ConfigValidation {
        /// Human-readable description of which rule was violated.
        message: String,
    },

    /// The SCF loop ran out of iterations without meeting the tolerance.
    #[error("SCF did not converge: {iterations} iterations, last residual {last_residual:e}")]
    ScfDivergence {
        /// Iteration count when the loop gave up.
        iterations: usize,
        /// Final density-change residual in the loop.
        last_residual: f64,
    },

    /// A physical-law identity (per `notes/discipline/ACCEPTANCE.md` §1.2)
    /// was violated by the converged state.
    #[error("physical-law identity violated: {identity} (residual {residual:e}, tolerance {tolerance:e})")]
    PhysicalLawViolation {
        /// Symbolic name of the broken identity (e.g. `pratt_sum_rule`).
        identity: &'static str,
        /// Observed residual.
        residual: f64,
        /// Declared tolerance.
        tolerance: f64,
    },

    /// A required external file (reference dataset, warm-start state, ...)
    /// could not be located or parsed.
    #[error("external artifact error at {path}: {message}")]
    Artifact {
        /// The path the runtime tried to access.
        path: PathBuf,
        /// Human-readable description.
        message: String,
    },

    /// A wrapper around generic IO failures from `output/` writers.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization error from `output/` writers.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
