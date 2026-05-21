//! Reference-dataset loader and comparator for `scrapbox validate`.
//!
//! Compares the runner's outputs against a checked-in JSON reference,
//! per `notes/discipline/HARNESS.md` §validation flow.

use crate::config::Validation as ValidationConfig;
use crate::error::{Result, ScrapboxError};
use crate::observables::ObservableReport;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Reference dataset shape on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceDataset {
    /// Mirrors the runner's `schema_version`.
    pub schema_version: String,
    /// Source of the reference (e.g. `"exact_diagonalization"`).
    pub produced_by: String,
    /// Reference observables (any may be `None` — comparison is skipped).
    #[serde(default)]
    pub observables: ObservableReport,
    /// Reference density vector (length == `num_sites`). `None` skips comparison.
    #[serde(default)]
    pub site_density: Option<Vec<f64>>,
}

impl ReferenceDataset {
    /// Load a reference dataset from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|source| ScrapboxError::Artifact {
            path: path.to_path_buf(),
            message: format!("failed to read reference dataset: {source}"),
        })?;
        let ds: Self = serde_json::from_str(&raw)?;
        Ok(ds)
    }
}

/// Result of one validation comparison.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    /// Per-observable residuals.
    pub residuals: Vec<ObservableResidual>,
    /// Did every requested observable pass its tolerance?
    pub all_passed: bool,
}

/// Per-observable comparison record.
#[derive(Debug, Clone, Serialize)]
pub struct ObservableResidual {
    /// Symbolic name (`"mean_work"`, `"free_energy"`, `"density"`, ...).
    pub name: &'static str,
    /// Worst element-wise absolute difference between value and reference.
    pub residual: f64,
    /// Tolerance applied.
    pub tolerance: f64,
    /// Did the comparison pass (or skip because reference is absent)?
    pub passed: bool,
}

/// Compare the runner's outputs against the reference dataset.
pub fn compare(
    observed: &ObservableReport,
    site_density: &[f64],
    reference: &ReferenceDataset,
    cfg: &ValidationConfig,
) -> Result<ValidationReport> {
    let tol = &cfg.tolerances;
    let residuals = vec![
        compare_scalar(
            "free_energy",
            observed.free_energy,
            reference.observables.free_energy,
            tol.free_energy,
        ),
        compare_scalar(
            "mean_work",
            observed.mean_work,
            reference.observables.mean_work,
            tol.mean_work,
        ),
        compare_scalar(
            "irreversible_entropy",
            observed.irreversible_entropy,
            reference.observables.irreversible_entropy,
            tol.mean_work,
        ),
        compare_density(
            "density",
            site_density,
            reference.site_density.as_deref(),
            tol.density,
        ),
    ];
    let all_passed = residuals.iter().all(|r| r.passed);
    Ok(ValidationReport {
        residuals,
        all_passed,
    })
}

fn compare_scalar(
    name: &'static str,
    observed: Option<f64>,
    reference: Option<f64>,
    tolerance: f64,
) -> ObservableResidual {
    match (observed, reference) {
        (Some(o), Some(r)) => {
            let residual = (o - r).abs();
            ObservableResidual {
                name,
                residual,
                tolerance,
                passed: residual <= tolerance,
            }
        }
        _ => ObservableResidual {
            name,
            residual: f64::NAN,
            tolerance,
            passed: true, // skip when reference is absent
        },
    }
}

fn compare_density(
    name: &'static str,
    observed: &[f64],
    reference: Option<&[f64]>,
    tolerance: f64,
) -> ObservableResidual {
    let Some(reference) = reference else {
        return ObservableResidual {
            name,
            residual: f64::NAN,
            tolerance,
            passed: true,
        };
    };
    if observed.len() != reference.len() {
        return ObservableResidual {
            name,
            residual: f64::INFINITY,
            tolerance,
            passed: false,
        };
    }
    let residual = observed
        .iter()
        .zip(reference.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0_f64, f64::max);
    ObservableResidual {
        name,
        residual,
        tolerance,
        passed: residual <= tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ValidationTolerances;

    fn cfg() -> ValidationConfig {
        ValidationConfig {
            reference_path: "ignored".into(),
            tolerances: ValidationTolerances {
                density: 1e-5,
                free_energy: 1e-6,
                mean_work: 1e-5,
            },
            fail_on_mismatch: true,
        }
    }

    fn ref_dataset(free_energy: Option<f64>, density: Option<Vec<f64>>) -> ReferenceDataset {
        ReferenceDataset {
            schema_version: "0.1".into(),
            produced_by: "test".into(),
            observables: ObservableReport {
                free_energy,
                ..ObservableReport::default()
            },
            site_density: density,
        }
    }

    #[test]
    fn matching_values_pass() {
        let observed = ObservableReport {
            free_energy: Some(-2.018),
            ..ObservableReport::default()
        };
        let reference = ref_dataset(Some(-2.018), Some(vec![1.0, 1.0]));
        let report = compare(&observed, &[1.0, 1.0], &reference, &cfg()).unwrap();
        assert!(report.all_passed, "report = {report:?}");
    }

    #[test]
    fn tampered_density_fails() {
        let observed = ObservableReport {
            free_energy: Some(-2.018),
            ..ObservableReport::default()
        };
        let reference = ref_dataset(Some(-2.018), Some(vec![0.5, 1.5]));
        let report = compare(&observed, &[1.0, 1.0], &reference, &cfg()).unwrap();
        assert!(!report.all_passed);
        let density_row = report
            .residuals
            .iter()
            .find(|r| r.name == "density")
            .unwrap();
        assert!(!density_row.passed);
        assert!((density_row.residual - 0.5).abs() < 1e-12);
    }

    #[test]
    fn missing_reference_skips_observable() {
        let observed = ObservableReport {
            free_energy: Some(-2.018),
            ..ObservableReport::default()
        };
        let reference = ref_dataset(None, None);
        let report = compare(&observed, &[1.0, 1.0], &reference, &cfg()).unwrap();
        assert!(report.all_passed);
    }

    #[test]
    fn length_mismatch_in_density_fails_hard() {
        let observed = ObservableReport::default();
        let reference = ref_dataset(None, Some(vec![1.0, 1.0, 1.0]));
        let report = compare(&observed, &[1.0, 1.0], &reference, &cfg()).unwrap();
        let density_row = report
            .residuals
            .iter()
            .find(|r| r.name == "density")
            .unwrap();
        assert!(!density_row.passed);
        assert!(density_row.residual.is_infinite());
    }
}
