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
use crate::spectrum::krylov::{KrylovSpec, KrylovStats};
use crate::spectrum::linear_operator::LinearOperator;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct KrylovStatsJson {
    pub min_m: usize,
    pub max_m: usize,
    pub mean_m: f64,
}

impl From<KrylovStats> for KrylovStatsJson {
    fn from(s: KrylovStats) -> Self {
        Self {
            min_m: s.min_m,
            max_m: s.max_m,
            mean_m: s.mean_m,
        }
    }
}

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
        #[serde(skip_serializing_if = "Option::is_none")]
        krylov_stats: Option<KrylovStatsJson>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        krylov_stats: Option<KrylovStatsJson>,
    },
    Theta2 {
        source: &'static str,
        dim: usize,
        beta: f64,
        theta_2: f64,
        wall_time_ms: u128,
        #[serde(skip_serializing_if = "Option::is_none")]
        krylov_stats: Option<KrylovStatsJson>,
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
        (TpqKind::Theta2, TpqSource::Ed) => run_theta_2_ed(cfg, &v_ext, n_up, n_dn, beta)?,
        (TpqKind::Theta2, TpqSource::MatrixFree) => {
            run_theta_2_mf(cfg, &v_ext, n_up, n_dn, beta, tpq_cfg)?
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
        krylov_stats: None,
    }
}

fn krylov_spec_from(tpq_cfg: &TpqSpec) -> KrylovSpec {
    tpq_cfg.krylov_tol.map_or_else(
        || KrylovSpec::Fixed {
            m: tpq_cfg.krylov_m.unwrap_or(30),
        },
        |tol| KrylovSpec::Adaptive {
            tol,
            max_m: tpq_cfg.krylov_m.unwrap_or(60),
        },
    )
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
    let spec = krylov_spec_from(tpq_cfg);
    let start = Instant::now();
    let (density, krylov_stats) =
        tpq::tpq_density_matrix_free(&jw, beta, tpq_cfg.n_samples, tpq_cfg.seed, spec);
    let elapsed = start.elapsed().as_millis();
    TpqRunOutput::Density {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        beta,
        density,
        wall_time_ms: elapsed,
        krylov_stats: Some(KrylovStatsJson::from(krylov_stats)),
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
        krylov_stats: None,
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
    let spec = krylov_spec_from(tpq_cfg);
    let start = Instant::now();
    let (stats, krylov_stats) = tpq::tpq_work_statistics_matrix_free(
        &jw_init,
        &jw_final,
        beta,
        tpq_cfg.n_samples,
        tpq_cfg.seed,
        spec,
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
        krylov_stats: Some(KrylovStatsJson::from(krylov_stats)),
    })
}

fn run_theta_2_ed(
    cfg: &Config,
    v_init: &[f64],
    n_up: usize,
    n_dn: usize,
    beta: f64,
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
    let theta_2 = ed::exact_theta_2(&ed_init, &ed_final, beta);
    let elapsed = start.elapsed().as_millis();
    Ok(TpqRunOutput::Theta2 {
        source: "ed",
        dim,
        beta,
        theta_2,
        wall_time_ms: elapsed,
        krylov_stats: None,
    })
}

fn run_theta_2_mf(
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
    let k_states = tpq_cfg.theta_2_k_states.unwrap_or(16).min(dim);
    let krylov_m = tpq_cfg
        .krylov_m
        .unwrap_or(k_states * 2)
        .max(k_states)
        .min(dim);
    let start = Instant::now();
    let (theta_2, effective_m) =
        ed::exact_theta_2_matrix_free(&jw_init, &jw_final, beta, k_states, krylov_m);
    let elapsed = start.elapsed().as_millis();
    let krylov_stats = KrylovStatsJson {
        min_m: effective_m,
        max_m: effective_m,
        #[allow(clippy::cast_precision_loss)]
        mean_m: effective_m as f64,
    };
    Ok(TpqRunOutput::Theta2 {
        source: "matrix_free",
        dim,
        beta,
        theta_2,
        wall_time_ms: elapsed,
        krylov_stats: Some(krylov_stats),
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
