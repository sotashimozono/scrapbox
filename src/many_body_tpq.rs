#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    missing_docs
)]
//! TPQ analysis runner exposed via the `scrapbox tpq` CLI subcommand.
//!
//! Four modes selected by `[tpq]` (kind x source):
//! density via ed / matrix_free; work_statistics via ed / matrix_free.
//! Matrix-free paths consume `JwHubbard` via the LinearOperator trait
//! so Hilbert dimensions too large for ED still work. work_statistics
//! requires `[quench]` for the final Hamiltonian (only Sudden today).

use crate::config::{Config, Quench, TpqKind, TpqSource, TpqSpec};
use crate::error::{Result, ScrapboxError};
use crate::reference::{ed, tpq};
use crate::spectrum::hubbard_jw::JwHubbard;
use crate::spectrum::linear_operator::LinearOperator;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TpqRunOutput {
    Density {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        seed: u64,
        beta: f64,
        density: Vec<f64>,
        wall_time_ms: u128,
    },
    WorkStatistics {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        seed: u64,
        beta: f64,
        mean_w: f64,
        work_variance: f64,
        mean_w_stderr: f64,
        wall_time_ms: u128,
    },
}

pub fn run(cfg: &Config) -> Result<TpqRunOutput> {
    let tpq_cfg = cfg
        .tpq
        .as_ref()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: "[tpq] section required for `scrapbox tpq` (set kind, source, n_samples, seed; optional beta, krylov_m)".into(),
        })?;
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let beta = tpq_cfg.beta.unwrap_or(cfg.hamiltonian.beta);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;

    let out = match (tpq_cfg.kind, tpq_cfg.source) {
        (TpqKind::Density, TpqSource::Ed) => run_density_ed(cfg, &v_ext, n_up, n_dn, beta, tpq_cfg),
        (TpqKind::Density, TpqSource::MatrixFree) => {
            run_density_mf(cfg, &v_ext, n_up, n_dn, beta, tpq_cfg)
        }
        (TpqKind::WorkStatistics, TpqSource::Ed) => {
            run_work_ed(cfg, &v_ext, n_up, n_dn, beta, tpq_cfg)?
        }
        (TpqKind::WorkStatistics, TpqSource::MatrixFree) => {
            run_work_mf(cfg, &v_ext, n_up, n_dn, beta, tpq_cfg)?
        }
    };

    let out_dir = crate::bin_support::resolve_output_dir(cfg);
    std::fs::create_dir_all(&out_dir).map_err(|source| ScrapboxError::Artifact {
        path: out_dir.clone(),
        message: format!("failed to create output dir: {source}"),
    })?;
    let out_path = out_dir.join("tpq_report.json");
    write_output_json(&out_path, &out)?;
    Ok(out)
}

fn run_density_ed(
    cfg: &Config,
    v_ext: &[f64],
    n_up: usize,
    n_dn: usize,
    beta: f64,
    tpq_cfg: &TpqSpec,
) -> TpqRunOutput {
    let ed_result = ed::canonical_thermal(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_ext,
    );
    let dim = ed_result.eigenvalues.len();
    let start = Instant::now();
    let density = tpq::tpq_density(&ed_result, beta, tpq_cfg.n_samples, tpq_cfg.seed);
    let elapsed = start.elapsed().as_millis();
    TpqRunOutput::Density {
        source: "ed",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        beta,
        density,
        wall_time_ms: elapsed,
    }
}

fn run_density_mf(
    cfg: &Config,
    v_ext: &[f64],
    n_up: usize,
    n_dn: usize,
    beta: f64,
    tpq_cfg: &TpqSpec,
) -> TpqRunOutput {
    let jw = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_ext,
    );
    let dim = jw.dim();
    let m = tpq_cfg.krylov_m.unwrap_or(30);
    let start = Instant::now();
    let density = tpq::tpq_density_matrix_free(&jw, beta, tpq_cfg.n_samples, tpq_cfg.seed, m);
    let elapsed = start.elapsed().as_millis();
    TpqRunOutput::Density {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        beta,
        density,
        wall_time_ms: elapsed,
    }
}

fn resolve_v_final(cfg: &Config) -> Result<Vec<f64>> {
    let q = cfg.quench.as_ref().ok_or_else(|| ScrapboxError::ConfigValidation {
        message: "[tpq] kind = \"work_statistics\" requires a [quench] section to define the final Hamiltonian".into(),
    })?;
    let Quench::Sudden {
        final_external_potential,
    } = q;
    Ok(final_external_potential.to_site_values(cfg.hamiltonian.num_sites))
}

fn run_work_ed(
    cfg: &Config,
    v_init: &[f64],
    n_up: usize,
    n_dn: usize,
    beta: f64,
    tpq_cfg: &TpqSpec,
) -> Result<TpqRunOutput> {
    let v_final = resolve_v_final(cfg)?;
    let ed_init = ed::canonical_thermal(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_init,
    );
    let ed_final = ed::canonical_thermal(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_final,
    );
    let dim = ed_init.eigenvalues.len();
    let start = Instant::now();
    let stats =
        tpq::tpq_work_statistics(&ed_init, &ed_final, beta, tpq_cfg.n_samples, tpq_cfg.seed);
    let elapsed = start.elapsed().as_millis();
    Ok(TpqRunOutput::WorkStatistics {
        source: "ed",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        beta,
        mean_w: stats.mean_w,
        work_variance: stats.work_variance,
        mean_w_stderr: stats.mean_w_stderr,
        wall_time_ms: elapsed,
    })
}

fn run_work_mf(
    cfg: &Config,
    v_init: &[f64],
    n_up: usize,
    n_dn: usize,
    beta: f64,
    tpq_cfg: &TpqSpec,
) -> Result<TpqRunOutput> {
    let v_final = resolve_v_final(cfg)?;
    let jw_init = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_init,
    );
    let jw_final = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_final,
    );
    let dim = jw_init.dim();
    let m = tpq_cfg.krylov_m.unwrap_or(30);
    let start = Instant::now();
    let stats = tpq::tpq_work_statistics_matrix_free(
        &jw_init,
        &jw_final,
        beta,
        tpq_cfg.n_samples,
        tpq_cfg.seed,
        m,
    );
    let elapsed = start.elapsed().as_millis();
    Ok(TpqRunOutput::WorkStatistics {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        beta,
        mean_w: stats.mean_w,
        work_variance: stats.work_variance,
        mean_w_stderr: stats.mean_w_stderr,
        wall_time_ms: elapsed,
    })
}

fn write_output_json(path: &Path, out: &TpqRunOutput) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|source| ScrapboxError::Artifact {
        path: path.to_path_buf(),
        message: format!("failed to write tpq_report.json: {source}"),
    })?;
    serde_json::to_writer_pretty(file, out)?;
    Ok(())
}
