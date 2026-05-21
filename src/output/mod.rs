//! Output writers — JSON in v0.1, parquet planned for v1.0.

pub mod json;

use crate::config::{Config, Output as OutputConfig, OutputFormat};
use crate::error::Result;
use crate::observables::ObservableReport;
use crate::scf::KsState;
use std::path::Path;

/// Write a converged [`KsState`] + observable report.
///
/// Use [`write_run_outputs_with_config`] to additionally dump the resolved
/// `run.toml` (recommended for the CLI; required for `scrapbox validate`
/// reproducibility).
pub fn write_run_outputs(
    output_dir: &Path,
    state: &KsState,
    observables: &ObservableReport,
    cfg: &OutputConfig,
) -> Result<()> {
    match cfg.format {
        OutputFormat::Json => json::write(output_dir, state, observables, cfg),
    }
}

/// Same as [`write_run_outputs`] but also writes a `run.toml` (the
/// fully-resolved config) into the output directory.
pub fn write_run_outputs_with_config(
    output_dir: &Path,
    state: &KsState,
    observables: &ObservableReport,
    out_cfg: &OutputConfig,
    full_cfg: &Config,
) -> Result<()> {
    match out_cfg.format {
        OutputFormat::Json => {
            json::write_with_config(output_dir, state, observables, out_cfg, full_cfg)
        }
    }
}
