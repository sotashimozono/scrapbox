//! Strict TOML deserialization of `config.toml` per
//! `notes/discipline/CONFIG.md`.
//!
//! Entry points:
//!
//! - [`Config::from_file`]     — load + validate a config from disk.
//! - [`Config::from_toml_str`] — load + validate from an in-memory string.
//!
//! Unknown keys are a hard error (`deny_unknown_fields`), per the
//! discipline invariant in `notes/discipline/CONFIG.md` §schema-level rules.

use crate::error::{Result, ScrapboxError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// `schema_version` value this build supports.
pub const SUPPORTED_SCHEMA_VERSION: &str = "0.2";

/// Top-level `config.toml` deserialization target.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Schema version, required.
    pub schema_version: String,
    /// Run identity (name, description, tags). Never affects physics.
    pub meta: Meta,
    /// The physical model — the only Layer 0 input.
    pub hamiltonian: Hamiltonian,
    /// Exchange-correlation functional choice (Layer 1).
    pub xc_functional: XcFunctional,
    /// Spectrum source (Layer 3).
    pub spectrum_source: SpectrumSource,
    /// Canonical density evaluator (Layer 4).
    pub density_evaluator: DensityEvaluator,
    /// SCF loop control (Layer 5).
    pub scf: Scf,
    /// Sudden-quench protocol — optional.
    #[serde(default)]
    pub quench: Option<Quench>,
    /// Observables to compute and report (Layer 6).
    pub observables: Observables,
    /// Where results go on disk.
    pub output: Output,
    /// Reference-dataset comparison — optional, triggers `scrapbox validate`.
    #[serde(default)]
    pub validation: Option<Validation>,
    /// Parameter grid — optional, drives `scrapbox sweep`.
    #[serde(default)]
    pub sweep: Option<Sweep>,
}

impl Config {
    /// Load a config from a file on disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|source| ScrapboxError::ConfigRead {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&raw)
    }

    /// Parse a config from a TOML string.
    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let config: Self =
            toml::from_str(raw).map_err(|source| ScrapboxError::ConfigParse { source })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(ScrapboxError::SchemaVersionMismatch {
                found: self.schema_version.clone(),
                supported: SUPPORTED_SCHEMA_VERSION,
            });
        }
        self.hamiltonian.validate()?;
        Ok(())
    }
}

// ─── Meta ──────────────────────────────────────────────────────────────

/// Run identity. Free-form except `name` (must be filesystem-safe).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    /// Filesystem-safe identifier for the run.
    pub name: String,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional creation date (free-form string, typically ISO 8601).
    #[serde(default)]
    pub created: Option<String>,
    /// Optional tags.
    #[serde(default)]
    pub tags: Vec<String>,
}

// ─── Hamiltonian ───────────────────────────────────────────────────────

/// The physical model — Layer 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hamiltonian {
    /// Lattice model identifier. v0.1: `"hubbard_1d_inhomogeneous"`.
    pub model: String,
    /// Number of lattice sites `L`.
    pub num_sites: usize,
    /// Hopping parameter `J`.
    pub hopping_j: f64,
    /// On-site interaction `U`.
    pub on_site_interaction: f64,
    /// Spinful? (v0.1 always `true`.)
    pub spinful: bool,
    /// `N_↑ = N_↓` per spin sector.
    pub num_electrons_per_spin: usize,
    /// Inverse temperature `β`.
    pub beta: f64,
    /// Unit convention: `"j_units"` (default) or `"absolute"`.
    #[serde(default = "default_units")]
    pub units: String,
    /// External potential.
    pub external_potential: ExternalPotential,
}

fn default_units() -> String {
    "j_units".to_string()
}

impl Hamiltonian {
    fn validate(&self) -> Result<()> {
        if self.num_sites == 0 {
            return Err(ScrapboxError::ConfigValidation {
                message: "hamiltonian.num_sites must be > 0".into(),
            });
        }
        if self.num_electrons_per_spin > self.num_sites {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "hamiltonian.num_electrons_per_spin ({}) must be <= num_sites ({})",
                    self.num_electrons_per_spin, self.num_sites
                ),
            });
        }
        if self.beta <= 0.0 || !self.beta.is_finite() {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "hamiltonian.beta must be positive and finite (got {})",
                    self.beta
                ),
            });
        }
        if !(self.units == "j_units" || self.units == "absolute") {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "hamiltonian.units must be \"j_units\" or \"absolute\" (got {:?})",
                    self.units
                ),
            });
        }
        self.external_potential.validate(self.num_sites)?;
        if self.model != "hubbard_1d_inhomogeneous" {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "hamiltonian.model = {:?} is not supported in this build (v0.1 supports only \"hubbard_1d_inhomogeneous\")",
                    self.model
                ),
            });
        }
        Ok(())
    }
}

/// External-potential specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum ExternalPotential {
    /// `V_i = amplitude` for all `i`.
    Uniform {
        /// Constant value.
        amplitude: f64,
    },
    /// `V_i = amplitude * (-1)^i`.
    Comb {
        /// Per-site amplitude; sign alternates.
        amplitude: f64,
    },
    /// Explicit per-site values.
    Explicit {
        /// Length must equal `num_sites`.
        values: Vec<f64>,
    },
}

impl ExternalPotential {
    fn validate(&self, num_sites: usize) -> Result<()> {
        if let Self::Explicit { values } = self {
            if values.len() != num_sites {
                return Err(ScrapboxError::ConfigValidation {
                    message: format!(
                        "hamiltonian.external_potential.values has length {} but num_sites is {}",
                        values.len(),
                        num_sites
                    ),
                });
            }
        }
        Ok(())
    }

    /// Materialize into a length-`num_sites` vector.
    #[must_use]
    pub fn to_site_values(&self, num_sites: usize) -> Vec<f64> {
        match self {
            Self::Uniform { amplitude } => vec![*amplitude; num_sites],
            Self::Comb { amplitude } => (0..num_sites)
                .map(|i| if i % 2 == 0 { *amplitude } else { -*amplitude })
                .collect(),
            Self::Explicit { values } => values.clone(),
        }
    }
}

// ─── XcFunctional ──────────────────────────────────────────────────────

/// Exchange-correlation choice (Layer 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum XcFunctional {
    /// Single-site analytical LDA for Hubbard (per ARCHITECTURE Sec V).
    HubbardLda {
        /// Numerical parameters (clamp etc.).
        #[serde(default)]
        params: HubbardLdaParams,
    },
    /// Set `λ^{h-xc} = 0` everywhere.
    NonInteracting,
}

/// Numerical parameters for the Hubbard LDA xc functional.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HubbardLdaParams {
    /// Density clamp `η` near `n_i → 0` or `n_i → 2`.
    #[serde(default = "default_clamp_eta")]
    pub clamp_eta: f64,
}

impl Default for HubbardLdaParams {
    fn default() -> Self {
        Self {
            clamp_eta: default_clamp_eta(),
        }
    }
}

const fn default_clamp_eta() -> f64 {
    1.0e-14
}

// ─── SpectrumSource ────────────────────────────────────────────────────

/// How to obtain the eigendecomposition of `H^KS` (Layer 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum SpectrumSource {
    /// Dense LAPACK-style eigendecomposition via `faer`.
    DenseDiag,
}

// ─── DensityEvaluator ──────────────────────────────────────────────────

/// How to assemble `{n_i^β}` and `Z_N` (Layer 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum DensityEvaluator {
    /// Pratt-Borrmann-Franke recursion (exact for non-interacting fermions).
    PrattRecursion {
        /// Tunable knobs.
        #[serde(default)]
        params: PrattParams,
    },
}

/// Numerical parameters for Pratt recursion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrattParams {
    /// Subtract `ε_min` before the recursion to avoid overflow.
    #[serde(default = "default_true")]
    pub spectrum_shift: bool,
}

impl Default for PrattParams {
    fn default() -> Self {
        Self {
            spectrum_shift: true,
        }
    }
}

const fn default_true() -> bool {
    true
}

// ─── Scf ───────────────────────────────────────────────────────────────

/// Self-consistent loop control (Layer 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scf {
    /// Maximum number of SCF iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on `‖n^(k+1) − n^(k)‖_∞`.
    pub tolerance: f64,
    /// Density mixing strategy.
    pub mixing: Mixing,
    /// How to initialize the density at iteration 0.
    pub initial_density: InitialDensity,
}

/// Density mixing strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum Mixing {
    /// Simple linear `α n_new + (1-α) n_old`.
    Linear {
        /// Mixing fraction `α ∈ (0, 1]`.
        alpha: f64,
    },
    /// Pulay / DIIS extrapolation over the last `history_depth` density
    /// residuals. Falls back to linear mixing until the history is filled.
    /// See `notes/Zettelkasten/PermanentNote/canonical-ks-construction.md`
    /// "Failure modes / where typicality enters" §2 for context.
    Pulay {
        /// Mixing fraction applied to the residual when extrapolating.
        alpha: f64,
        /// Number of past (density, residual) pairs to retain. Typical: 6-8.
        #[serde(default = "default_pulay_history_depth")]
        history_depth: usize,
    },
}

const fn default_pulay_history_depth() -> usize {
    8
}

/// Initial-density specification for the SCF loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum InitialDensity {
    /// Uniform `n_i = N / L` (half-filling target).
    Uniform,
    /// Explicit starting density vector.
    Explicit {
        /// Length must equal `hamiltonian.num_sites`.
        values: Vec<f64>,
    },
}

// ─── Quench ────────────────────────────────────────────────────────────

/// Quench protocol — currently only "sudden" supported.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum Quench {
    /// Instantaneous switch of the external potential.
    Sudden {
        /// Post-quench external potential.
        final_external_potential: ExternalPotential,
    },
}

// ─── Observables ───────────────────────────────────────────────────────

/// Which observables to compute and dump.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observables {
    /// Compute `<W>` (requires `[quench]`).
    #[serde(default)]
    pub mean_work: bool,
    /// Compute `<S_irr> = β(<W> − ΔF)` (requires `[quench]`).
    #[serde(default)]
    pub irreversible_entropy: bool,
    /// Compute the second moment `<W²>` and variance `σ_w²` of the
    /// sudden-quench work distribution (Palamara 2024 eq 28).
    /// Requires `[quench]`.
    #[serde(default)]
    pub work_variance: bool,
    /// Quantum correction `Θ_2` (Palamara 2024 eq 30) needed when the
    /// pre- and post-quench Hamiltonians do not commute.
    #[serde(default)]
    pub theta_2: Theta2Spec,
    /// Dump `F = -β⁻¹ ln Z_N`.
    #[serde(default = "default_true")]
    pub free_energy: bool,
    /// Dump the per-spin canonical partition function.
    #[serde(default = "default_true")]
    pub partition_function: bool,
}

/// How to estimate `Θ_2` per Palamara 2024 §IV.1.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Theta2Spec {
    /// `zero` (default — neglect) or `lda` (requires a homogeneous-
    /// system reference; lands fully in v0.3 once BALDA is wired).
    #[serde(default = "default_theta_2_method")]
    pub method: String,
}

fn default_theta_2_method() -> String {
    "zero".to_string()
}

// ─── Output ────────────────────────────────────────────────────────────

/// Where to write results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    /// Directory path, may contain `{meta.name}` interpolation.
    pub directory: String,
    /// Output format.
    pub format: OutputFormat,
    /// Dump the converged density vector.
    #[serde(default = "default_true")]
    pub dump_density: bool,
    /// Dump the KS spectrum (eigenvalues).
    #[serde(default = "default_true")]
    pub dump_spectrum: bool,
    /// Dump KS eigenvectors (large; off by default).
    #[serde(default)]
    pub dump_eigvecs: bool,
    /// Dump `Z_1(mβ), Z_k`.
    #[serde(default = "default_true")]
    pub dump_partition_function: bool,
    /// Error if directory exists (vs append).
    #[serde(default)]
    pub overwrite: bool,
}

/// Serialization format for output files.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum OutputFormat {
    /// JSON (v0.1 default).
    Json,
}

// ─── Validation ────────────────────────────────────────────────────────

/// Reference-dataset comparison directives.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Validation {
    /// Path to the JSON reference dataset.
    pub reference_path: PathBuf,
    /// Per-observable tolerances.
    pub tolerances: ValidationTolerances,
    /// Should mismatches block (exit nonzero)?
    #[serde(default = "default_true")]
    pub fail_on_mismatch: bool,
}

/// Per-observable tolerances for validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationTolerances {
    /// Tolerance on per-site density values.
    #[serde(default = "default_tol_density")]
    pub density: f64,
    /// Tolerance on `F`.
    #[serde(default = "default_tol_energy")]
    pub free_energy: f64,
    /// Tolerance on `<W>`.
    #[serde(default = "default_tol_energy")]
    pub mean_work: f64,
}

const fn default_tol_density() -> f64 {
    1.0e-5
}

const fn default_tol_energy() -> f64 {
    1.0e-6
}

// ─── Sweep ─────────────────────────────────────────────────────────────

/// Parameter-grid sweep specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    /// Axes (cartesian product).
    pub axes: Vec<SweepAxis>,
    /// Subdirectory template for each cell. `{key}` interpolation.
    pub subdir_template: String,
    /// Number of parallel workers.
    #[serde(default = "default_sweep_parallelism")]
    pub parallelism: usize,
}

/// Single axis of a sweep — `key` is a dotted config path, `values` are the override list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepAxis {
    /// Dotted config key (e.g. `"hamiltonian.on_site_interaction"`).
    pub key: String,
    /// Values to substitute in turn.
    pub values: Vec<f64>,
}

const fn default_sweep_parallelism() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_DIMER_TOML: &str = r#"
schema_version = "0.2"

[meta]
name = "dimer_smoke"

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 2
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 1
beta = 2.0
external_potential.kind = "uniform"
external_potential.amplitude = 0.0

[xc_functional]
kind = "hubbard_lda"

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 200
tolerance = 1e-10
mixing.kind = "linear"
mixing.alpha = 0.5
initial_density.kind = "uniform"

[observables]
mean_work = false
irreversible_entropy = false
free_energy = true

[output]
directory = "runs/dimer_smoke"
format = "json"
"#;

    #[test]
    fn parses_full_dimer_config() {
        let cfg = Config::from_toml_str(FULL_DIMER_TOML).expect("should parse");
        assert_eq!(cfg.schema_version, "0.2");
        assert_eq!(cfg.meta.name, "dimer_smoke");
        assert!(matches!(cfg.xc_functional, XcFunctional::HubbardLda { .. }));
        assert!(matches!(cfg.spectrum_source, SpectrumSource::DenseDiag));
        assert!(matches!(
            cfg.density_evaluator,
            DensityEvaluator::PrattRecursion { .. }
        ));
        assert!(matches!(cfg.scf.mixing, Mixing::Linear { .. }));
        assert!(matches!(cfg.scf.initial_density, InitialDensity::Uniform));
        assert!(matches!(cfg.output.format, OutputFormat::Json));
        assert!(cfg.quench.is_none());
    }

    #[test]
    fn parses_sudden_quench_section() {
        let raw = format!(
            "{FULL_DIMER_TOML}\n[quench]\nkind = \"sudden\"\n\
             final_external_potential.kind = \"comb\"\n\
             final_external_potential.amplitude = 4.5\n"
        );
        let cfg = Config::from_toml_str(&raw).expect("should parse");
        let quench = cfg.quench.expect("quench should be present");
        match quench {
            Quench::Sudden {
                final_external_potential,
            } => {
                assert!(matches!(
                    final_external_potential,
                    ExternalPotential::Comb { .. }
                ));
            }
        }
    }

    #[test]
    fn rejects_unknown_top_level_key() {
        let raw = format!("{FULL_DIMER_TOML}\n[unexpected_section]\nfoo = 1\n");
        let err = Config::from_toml_str(&raw).expect_err("must reject unknown key");
        assert!(matches!(err, ScrapboxError::ConfigParse { .. }));
    }

    #[test]
    fn rejects_schema_version_mismatch() {
        let raw = FULL_DIMER_TOML.replace("\"0.2\"", "\"0.99\"");
        let err = Config::from_toml_str(&raw).expect_err("must reject");
        assert!(matches!(err, ScrapboxError::SchemaVersionMismatch { .. }));
    }

    #[test]
    fn rejects_more_electrons_than_sites() {
        let raw =
            FULL_DIMER_TOML.replace("num_electrons_per_spin = 1", "num_electrons_per_spin = 3");
        let err = Config::from_toml_str(&raw).expect_err("must reject");
        assert!(matches!(err, ScrapboxError::ConfigValidation { .. }));
    }

    #[test]
    fn rejects_non_positive_beta() {
        let raw = FULL_DIMER_TOML.replace("beta = 2.0", "beta = -1.0");
        let err = Config::from_toml_str(&raw).expect_err("must reject");
        assert!(matches!(err, ScrapboxError::ConfigValidation { .. }));
    }

    #[test]
    fn comb_external_potential_materializes_alternating() {
        let pot = ExternalPotential::Comb { amplitude: 0.5 };
        assert_eq!(pot.to_site_values(4), vec![0.5, -0.5, 0.5, -0.5]);
    }

    #[test]
    fn explicit_external_potential_length_must_match() {
        let raw = FULL_DIMER_TOML.replace(
            "external_potential.kind = \"uniform\"\nexternal_potential.amplitude = 0.0",
            "external_potential = { kind = \"explicit\", values = [0.5, -0.5, 0.5] }",
        );
        let err = Config::from_toml_str(&raw).expect_err("3 != 2");
        assert!(matches!(err, ScrapboxError::ConfigValidation { .. }));
    }
}
