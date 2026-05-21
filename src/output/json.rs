//! JSON output writer — produces one file per concern per
//! `notes/discipline/HARNESS.md` §output directory layout.

use crate::config::{Config, Output as OutputConfig};
use crate::error::{Result, ScrapboxError};
use crate::observables::ObservableReport;
use crate::scf::KsState;
use serde::Serialize;
use std::path::Path;

/// On-disk shape of `density.json`.
#[derive(Debug, Serialize)]
struct DensityDump<'a> {
    num_sites: usize,
    site_density: &'a [f64],
}

/// On-disk shape of `spectrum.json`.
#[derive(Debug, Serialize)]
struct SpectrumDump<'a> {
    num_states: usize,
    eigenvalues: &'a [f64],
    /// Per-state eigenvector column. Present only if `output.dump_eigvecs`
    /// is `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    eigenvectors: Option<Vec<Vec<f64>>>,
}

/// On-disk shape of `partition_function.json`.
#[derive(Debug, Serialize)]
struct PartitionFunctionDump {
    z_per_spin: f64,
    z_total: f64,
    free_energy: f64,
    iterations: usize,
    residual: f64,
}

/// Write the per-run JSON files into `output_dir`.
///
/// Produces `density.json`, `spectrum.json`, `partition_function.json`, and
/// `observables.json` (the last is always written, even when empty).
/// Use [`write_with_config`] to additionally dump the resolved `run.toml`.
pub fn write(
    output_dir: &Path,
    state: &KsState,
    observables: &ObservableReport,
    cfg: &OutputConfig,
) -> Result<()> {
    write_impl(output_dir, state, observables, cfg, None)
}

/// Write the standard outputs plus `run.toml` for full reproducibility.
pub fn write_with_config(
    output_dir: &Path,
    state: &KsState,
    observables: &ObservableReport,
    out_cfg: &OutputConfig,
    full_cfg: &Config,
) -> Result<()> {
    write_impl(output_dir, state, observables, out_cfg, Some(full_cfg))
}

fn write_impl(
    output_dir: &Path,
    state: &KsState,
    observables: &ObservableReport,
    cfg: &OutputConfig,
    full_cfg: Option<&Config>,
) -> Result<()> {
    if output_dir.exists() {
        if !cfg.overwrite {
            return Err(ScrapboxError::Artifact {
                path: output_dir.to_path_buf(),
                message: "output directory already exists; set [output].overwrite=true to replace"
                    .into(),
            });
        }
    } else {
        std::fs::create_dir_all(output_dir)?;
    }

    if cfg.dump_density {
        let dump = DensityDump {
            num_sites: state.densities.len(),
            site_density: &state.densities,
        };
        write_json(&output_dir.join("density.json"), &dump)?;
    }

    if cfg.dump_spectrum {
        let eigenvectors = if cfg.dump_eigvecs {
            let n = state.eigen.eigenvalues.len();
            let mut cols = Vec::with_capacity(n);
            for k in 0..n {
                let mut col = Vec::with_capacity(n);
                for i in 0..n {
                    col.push(state.eigen.eigenvectors[(i, k)]);
                }
                cols.push(col);
            }
            Some(cols)
        } else {
            None
        };
        let dump = SpectrumDump {
            num_states: state.eigen.eigenvalues.len(),
            eigenvalues: &state.eigen.eigenvalues,
            eigenvectors,
        };
        write_json(&output_dir.join("spectrum.json"), &dump)?;
    }

    if cfg.dump_partition_function {
        let dump = PartitionFunctionDump {
            z_per_spin: state.partition_function_per_spin,
            z_total: state.partition_function,
            free_energy: state.free_energy,
            iterations: state.iterations,
            residual: state.residual,
        };
        write_json(&output_dir.join("partition_function.json"), &dump)?;
    }

    write_json(&output_dir.join("observables.json"), observables)?;

    if let Some(full) = full_cfg {
        let toml_text = toml::to_string_pretty(full).map_err(|e| ScrapboxError::Artifact {
            path: output_dir.join("run.toml"),
            message: format!("failed to serialize resolved config: {e}"),
        })?;
        std::fs::write(output_dir.join("run.toml"), toml_text)?;
    }

    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|source| ScrapboxError::Artifact {
        path: path.to_path_buf(),
        message: format!("failed to create output file: {source}"),
    })?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::Eigendecomposition;
    use faer::Mat;
    use serde_json::Value;

    fn dummy_state() -> KsState {
        let mut eigvecs = Mat::<f64>::zeros(2, 2);
        eigvecs[(0, 0)] = 1.0;
        eigvecs[(1, 1)] = 1.0;
        KsState {
            densities: vec![1.0, 1.0],
            eigen: Eigendecomposition {
                eigenvalues: vec![-1.0, 1.0],
                eigenvectors: eigvecs,
            },
            partition_function_per_spin: 0.5,
            partition_function: 0.25,
            free_energy: -0.123,
            hxc_potential: vec![0.0, 0.0],
            iterations: 7,
            residual: 1e-12,
        }
    }

    fn default_output_cfg(directory: &str) -> OutputConfig {
        OutputConfig {
            directory: directory.into(),
            format: crate::config::OutputFormat::Json,
            dump_density: true,
            dump_spectrum: true,
            dump_eigvecs: false,
            dump_partition_function: true,
            overwrite: false,
        }
    }

    #[test]
    fn writes_all_dumps_and_observables() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("scratch");
        let state = dummy_state();
        let obs = ObservableReport {
            free_energy: Some(-0.123),
            partition_function: Some(0.25),
            mean_work: None,
            irreversible_entropy: None,
        };
        let cfg = default_output_cfg(dir.to_str().unwrap());
        write(&dir, &state, &obs, &cfg).unwrap();

        for name in [
            "density.json",
            "spectrum.json",
            "partition_function.json",
            "observables.json",
        ] {
            let path = dir.join(name);
            assert!(path.exists(), "missing {name}");
            let raw = std::fs::read_to_string(&path).unwrap();
            let parsed: Value = serde_json::from_str(&raw).unwrap();
            assert!(parsed.is_object(), "{name} is not a JSON object");
        }
    }

    #[test]
    fn refuses_overwrite_unless_opted_in() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("scratch");
        std::fs::create_dir_all(&dir).unwrap();
        let state = dummy_state();
        let obs = ObservableReport::default();
        let cfg = default_output_cfg(dir.to_str().unwrap());
        let err = write(&dir, &state, &obs, &cfg).expect_err("should refuse existing dir");
        assert!(matches!(err, ScrapboxError::Artifact { .. }));
    }

    #[test]
    fn skip_eigvecs_when_not_requested() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("scratch");
        let cfg = default_output_cfg(dir.to_str().unwrap());
        write(&dir, &dummy_state(), &ObservableReport::default(), &cfg).unwrap();
        let raw = std::fs::read_to_string(dir.join("spectrum.json")).unwrap();
        assert!(
            !raw.contains("eigenvectors"),
            "eigenvectors must be absent when dump_eigvecs=false"
        );
    }
}
