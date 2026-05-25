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

/// One row of a beta-sweep density result.
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepBetaRow {
    /// Inverse temperature this row was evaluated at.
    pub beta: f64,
    /// Per-site canonical thermal density at this beta.
    pub density: Vec<f64>,
}

/// One row of a krylov_tol-sweep density result (v0.14 gamma).
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepKrylovTolRow {
    /// Adaptive-Krylov tolerance this row was evaluated at.
    pub krylov_tol: f64,
    /// Per-site canonical thermal density at this tol.
    pub density: Vec<f64>,
    /// Per-row Krylov diagnostics (varies between rows: tighter tol
    /// drives larger `effective_m`).
    pub krylov_stats: KrylovStatsJson,
}

/// One row of a beta-sweep work-statistics result (v0.14 gamma).
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepBetaWorkRow {
    /// Inverse temperature this row was evaluated at.
    pub beta: f64,
    /// Per-row TPQ mean work.
    pub mean_w: f64,
    /// Per-row TPQ work variance.
    pub work_variance: f64,
    /// Standard error of `mean_w` across TPQ samples.
    pub mean_w_stderr: f64,
}

/// One row of a 2-axis cartesian density sweep (v0.15 beta).
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepCartesianDensityRow {
    /// Primary-axis label (matches `axis` at the top of the output).
    pub axis_primary: &'static str,
    /// Primary-axis value at this cell.
    pub value_primary: f64,
    /// Secondary-axis label.
    pub axis_secondary: &'static str,
    /// Secondary-axis value at this cell.
    pub value_secondary: f64,
    /// Per-site canonical thermal density at this cell.
    pub density: Vec<f64>,
}

/// One row of a seed-sweep work-statistics result (v0.15 gamma).
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepSeedWorkRow {
    /// RNG seed this row was evaluated at.
    pub seed: u64,
    /// Per-row TPQ mean work at this seed.
    pub mean_w: f64,
    /// Per-row TPQ work variance at this seed.
    pub work_variance: f64,
    /// Standard error of `mean_w` across the TPQ samples WITHIN this
    /// seed run (intra-seed). For inter-seed dispersion see the
    /// `ensemble_summary` block.
    pub mean_w_stderr: f64,
}

/// Scalar ensemble summary of work statistics across the seed sweep
/// (v0.15 gamma): mean of per-seed `mean_w` plus its stderr across
/// seeds.
#[derive(Debug, Clone, Serialize)]
pub struct TpqWorkEnsembleSummary {
    /// Mean of per-seed `mean_w` across the seed list.
    pub mean_w_mean: f64,
    /// Standard error of `mean_w_mean` across seeds (inter-seed
    /// dispersion of the within-seed mean estimate). Uses sample
    /// stddev with N-1 denominator; zero for single-seed runs.
    pub mean_w_stderr_across_seeds: f64,
}

/// One row of a beta-sweep theta_2 result (v0.15 alpha).
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepBetaTheta2Row {
    /// Inverse temperature this row was evaluated at.
    pub beta: f64,
    /// Off-diagonal work-variance contribution
    /// (Palamara 2024 III.3 `Theta_2`) at this beta.
    pub theta_2: f64,
}

/// One row of a seed-sweep density result (v0.14 delta).
#[derive(Debug, Clone, Serialize)]
pub struct TpqSweepSeedRow {
    /// RNG seed this row was evaluated at.
    pub seed: u64,
    /// Per-site canonical thermal density at this seed.
    pub density: Vec<f64>,
}

/// Per-site mean and standard error of TPQ observables across the
/// seed-sweep ensemble (v0.14 delta).
#[derive(Debug, Clone, Serialize)]
pub struct TpqEnsembleSummary {
    /// Per-site mean density across all seeds.
    pub mean_density: Vec<f64>,
    /// Per-site standard error of the mean across seeds.
    pub stderr_density: Vec<f64>,
}

/// Output payload emitted by `scrapbox tpq` when `[tpq.sweep]` is set.
/// Written to `tpq_sweep_report.json` instead of `tpq_report.json`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TpqSweepRunOutput {
    /// Beta-sweep density rows (axis = "beta").
    Density {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        seed: u64,
        axis: &'static str,
        rows: Vec<TpqSweepBetaRow>,
        wall_time_ms: u128,
        krylov_stats: KrylovStatsJson,
    },
    /// Krylov-tol-sweep density rows (v0.14 gamma).
    DensityKrylovTol {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        seed: u64,
        axis: &'static str,
        rows: Vec<TpqSweepKrylovTolRow>,
        wall_time_ms: u128,
    },
    /// Beta-sweep work-statistics rows (v0.14 gamma).
    WorkStatistics {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        seed: u64,
        axis: &'static str,
        rows: Vec<TpqSweepBetaWorkRow>,
        wall_time_ms: u128,
        krylov_stats: KrylovStatsJson,
    },
    /// Seed-sweep work-statistics rows + ensemble summary (v0.15 gamma).
    WorkStatisticsSeedSweep {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        axis: &'static str,
        rows: Vec<TpqSweepSeedWorkRow>,
        ensemble_summary: TpqWorkEnsembleSummary,
        wall_time_ms: u128,
        krylov_stats: KrylovStatsJson,
    },
    /// 2-axis cartesian density sweep (v0.15 beta).
    CartesianDensity {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        axis_primary: &'static str,
        axis_secondary: &'static str,
        rows: Vec<TpqSweepCartesianDensityRow>,
        wall_time_ms: u128,
        krylov_stats: KrylovStatsJson,
    },
    /// Beta-sweep theta_2 rows (v0.15 alpha).
    #[serde(rename = "theta_2")]
    Theta2 {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        /// Echoed from [tpq].seed for schema parity with the density
        /// and work_statistics variants. theta_2 evaluators (ed and
        /// matrix-free) are deterministic given (jw_init, jw_final,
        /// beta), so this field has no effect on the output rows.
        seed: u64,
        axis: &'static str,
        rows: Vec<TpqSweepBetaTheta2Row>,
        wall_time_ms: u128,
        krylov_stats: KrylovStatsJson,
    },
    /// Seed-sweep density rows + ensemble summary (v0.14 delta).
    DensitySeedSweep {
        source: &'static str,
        dim: usize,
        n_samples: usize,
        axis: &'static str,
        rows: Vec<TpqSweepSeedRow>,
        ensemble_summary: TpqEnsembleSummary,
        wall_time_ms: u128,
        krylov_stats: KrylovStatsJson,
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
    let (theta_2, effective_m) = tpq_cfg.theta_2_lanczos_tol.map_or_else(
        || ed::exact_theta_2_matrix_free(&jw_init, &jw_final, beta, k_states, krylov_m),
        |tol| {
            let max_m = tpq_cfg.krylov_m.unwrap_or(80).max(k_states).min(dim);
            ed::exact_theta_2_matrix_free_adaptive(&jw_init, &jw_final, beta, k_states, tol, max_m)
        },
    );
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

/// Dispatch a `[tpq.sweep]` run.
///
/// v0.14 gamma supported `(axis, kind, source)` combinations:
///
/// - `(beta, density, matrix_free)` — v0.13 alpha; subspace reuse.
/// - `(beta, density, ed)` — v0.14 gamma; per-beta ED dispatch.
/// - `(beta, work_statistics, matrix_free)` — v0.14 gamma.
/// - `(krylov_tol, density, matrix_free)` — v0.14 gamma; per-tol
///   adaptive Lanczos.
///
/// Any other combination returns [`ScrapboxError::ConfigValidation`].
#[allow(clippy::too_many_lines)]
pub fn run_sweep(cfg: &Config) -> Result<TpqSweepRunOutput> {
    let tpq_cfg = cfg
        .tpq
        .as_ref()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: "[tpq] section required for `scrapbox tpq` sweep dispatch".into(),
        })?;
    let sweep_cfg = tpq_cfg
        .sweep
        .as_ref()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: "[tpq.sweep] section required for sweep dispatch (set axis and values)".into(),
        })?;
    if sweep_cfg.values.is_empty() {
        return Err(ScrapboxError::ConfigValidation {
            message: "[tpq.sweep].values must be non-empty".into(),
        });
    }

    if sweep_cfg.second_axis.is_some() || sweep_cfg.second_values.is_some() {
        let secondary_axis =
            sweep_cfg
                .second_axis
                .ok_or_else(|| ScrapboxError::ConfigValidation {
                    message: "[tpq.sweep].second_axis required when second_values is set".into(),
                })?;
        let secondary_values =
            sweep_cfg
                .second_values
                .as_ref()
                .ok_or_else(|| ScrapboxError::ConfigValidation {
                    message: "[tpq.sweep].second_values required when second_axis is set".into(),
                })?;
        if secondary_values.is_empty() {
            return Err(ScrapboxError::ConfigValidation {
                message: "[tpq.sweep].second_values must be non-empty".into(),
            });
        }
        if sweep_cfg.axis == secondary_axis {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "[tpq.sweep] cartesian axes must differ: axis = second_axis = {:?}",
                    sweep_cfg.axis
                ),
            });
        }
        let out = match (sweep_cfg.axis, secondary_axis, tpq_cfg.kind, tpq_cfg.source) {
            (
                crate::config::TpqSweepAxis::Beta,
                crate::config::TpqSweepAxis::Seed,
                TpqKind::Density,
                TpqSource::MatrixFree,
            )
            | (
                crate::config::TpqSweepAxis::Seed,
                crate::config::TpqSweepAxis::Beta,
                TpqKind::Density,
                TpqSource::MatrixFree,
            ) => run_sweep_cartesian_beta_seed_density_mf(
                cfg,
                tpq_cfg,
                sweep_cfg,
                secondary_axis,
                secondary_values,
            ),
            (axis, second_axis, kind, source) => {
                return Err(ScrapboxError::ConfigValidation {
                    message: format!(
                        "[tpq.sweep] cartesian (axis = {axis:?}, second_axis = {second_axis:?},                          kind = {kind:?}, source = {source:?}) unsupported; v0.15 beta supports                          only (beta, seed) and (seed, beta) cartesian for density + matrix_free"
                    ),
                });
            }
        };
        let out_dir = crate::bin_support::resolve_output_dir(cfg);
        std::fs::create_dir_all(&out_dir).map_err(|source| ScrapboxError::Artifact {
            path: out_dir.clone(),
            message: format!("failed to create output dir: {source}"),
        })?;
        let out_path = out_dir.join("tpq_sweep_report.json");
        write_sweep_output_json(&out_path, &out)?;
        return Ok(out);
    }

    let out = match (sweep_cfg.axis, tpq_cfg.kind, tpq_cfg.source) {
        (crate::config::TpqSweepAxis::Beta, TpqKind::Density, TpqSource::MatrixFree) => {
            run_sweep_beta_density_mf(cfg, tpq_cfg, sweep_cfg)
        }
        (crate::config::TpqSweepAxis::Beta, TpqKind::Density, TpqSource::Ed) => {
            run_sweep_beta_density_ed(cfg, tpq_cfg, sweep_cfg)
        }
        (crate::config::TpqSweepAxis::Beta, TpqKind::WorkStatistics, TpqSource::MatrixFree) => {
            run_sweep_beta_work_mf(cfg, tpq_cfg, sweep_cfg)?
        }
        (crate::config::TpqSweepAxis::KrylovTol, TpqKind::Density, TpqSource::MatrixFree) => {
            run_sweep_krylov_tol_density_mf(cfg, tpq_cfg, sweep_cfg)
        }
        (crate::config::TpqSweepAxis::Seed, TpqKind::Density, TpqSource::MatrixFree) => {
            run_sweep_seed_density_mf(cfg, tpq_cfg, sweep_cfg)
        }
        (crate::config::TpqSweepAxis::Beta, TpqKind::Theta2, TpqSource::Ed) => {
            run_sweep_beta_theta_2_ed(cfg, tpq_cfg, sweep_cfg)?
        }
        (crate::config::TpqSweepAxis::Beta, TpqKind::Theta2, TpqSource::MatrixFree) => {
            run_sweep_beta_theta_2_mf(cfg, tpq_cfg, sweep_cfg)?
        }
        (crate::config::TpqSweepAxis::Seed, TpqKind::WorkStatistics, TpqSource::MatrixFree) => {
            run_sweep_seed_work_mf(cfg, tpq_cfg, sweep_cfg)?
        }
        (axis, kind, source) => {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "[tpq.sweep] unsupported combination (axis = {axis:?}, kind = {kind:?}, source = {source:?}); v0.15 alpha supports (beta, density, matrix_free), (beta, density, ed), (beta, work_statistics, matrix_free), (krylov_tol, density, matrix_free), (seed, density, matrix_free), (beta, theta_2, ed), (beta, theta_2, matrix_free), (seed, work_statistics, matrix_free)"
                ),
            });
        }
    };

    let out_dir = crate::bin_support::resolve_output_dir(cfg);
    std::fs::create_dir_all(&out_dir).map_err(|source| ScrapboxError::Artifact {
        path: out_dir.clone(),
        message: format!("failed to create output dir: {source}"),
    })?;
    let out_path = out_dir.join("tpq_sweep_report.json");
    write_sweep_output_json(&out_path, &out)?;
    Ok(out)
}

fn run_sweep_beta_density_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> TpqSweepRunOutput {
    use crate::spectrum::linear_operator::LinearOperator;
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_ext,
    );
    let dim = jw.dim();
    let m = tpq_cfg.krylov_m.unwrap_or(30);
    let betas: Vec<f64> = sweep_cfg.values.clone();
    let start = Instant::now();
    let (rows, krylov_stats) =
        tpq::tpq_density_matrix_free_beta_sweep(&jw, &betas, tpq_cfg.n_samples, tpq_cfg.seed, m);
    let elapsed = start.elapsed().as_millis();
    let rows_payload: Vec<TpqSweepBetaRow> = betas
        .iter()
        .zip(rows)
        .map(|(&beta, density)| TpqSweepBetaRow { beta, density })
        .collect();
    TpqSweepRunOutput::Density {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        axis: "beta",
        rows: rows_payload,
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson::from(krylov_stats),
    }
}

fn run_sweep_beta_density_ed(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> TpqSweepRunOutput {
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let betas: Vec<f64> = sweep_cfg.values.clone();
    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepBetaRow> = Vec::with_capacity(betas.len());
    let mut dim = 0_usize;
    for &beta in &betas {
        let ed_result = ed::canonical_thermal(
            cfg.hamiltonian.num_sites,
            n_up,
            n_dn,
            cfg.hamiltonian.hopping_j,
            cfg.hamiltonian.on_site_interaction,
            &v_ext,
        );
        dim = ed_result.eigenvalues.len();
        let density = tpq::tpq_density(&ed_result, beta, tpq_cfg.n_samples, tpq_cfg.seed);
        rows_payload.push(TpqSweepBetaRow { beta, density });
    }
    let elapsed = start.elapsed().as_millis();
    TpqSweepRunOutput::Density {
        source: "ed",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        axis: "beta",
        rows: rows_payload,
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson {
            min_m: 0,
            max_m: 0,
            mean_m: 0.0,
        },
    }
}

fn run_sweep_beta_work_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> Result<TpqSweepRunOutput> {
    use crate::spectrum::linear_operator::LinearOperator;
    let v_init = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let v_final = resolve_v_final(cfg)?;
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw_init = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_init,
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
    let betas: Vec<f64> = sweep_cfg.values.clone();
    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepBetaWorkRow> = Vec::with_capacity(betas.len());
    let mut m_log: Vec<usize> = Vec::new();
    for &beta in &betas {
        let (stats, krylov_stats) = tpq::tpq_work_statistics_matrix_free(
            &jw_init,
            &jw_final,
            beta,
            tpq_cfg.n_samples,
            tpq_cfg.seed,
            spec,
        );
        m_log.push(krylov_stats.min_m);
        m_log.push(krylov_stats.max_m);
        rows_payload.push(TpqSweepBetaWorkRow {
            beta,
            mean_w: stats.mean_w,
            work_variance: stats.work_variance,
            mean_w_stderr: stats.mean_w_stderr,
        });
    }
    let elapsed = start.elapsed().as_millis();
    let agg_stats = KrylovStats::from_samples(&m_log);
    Ok(TpqSweepRunOutput::WorkStatistics {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        axis: "beta",
        rows: rows_payload,
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson::from(agg_stats),
    })
}

fn run_sweep_krylov_tol_density_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> TpqSweepRunOutput {
    use crate::spectrum::krylov::KrylovSpec;
    use crate::spectrum::linear_operator::LinearOperator;
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_ext,
    );
    let dim = jw.dim();
    let beta = tpq_cfg.beta.unwrap_or(cfg.hamiltonian.beta);
    let max_m = tpq_cfg.krylov_m.unwrap_or(60).min(dim);
    let tols: Vec<f64> = sweep_cfg.values.clone();
    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepKrylovTolRow> = Vec::with_capacity(tols.len());
    for &tol in &tols {
        let spec = KrylovSpec::Adaptive { tol, max_m };
        let (density, krylov_stats) =
            tpq::tpq_density_matrix_free(&jw, beta, tpq_cfg.n_samples, tpq_cfg.seed, spec);
        rows_payload.push(TpqSweepKrylovTolRow {
            krylov_tol: tol,
            density,
            krylov_stats: KrylovStatsJson::from(krylov_stats),
        });
    }
    let elapsed = start.elapsed().as_millis();
    TpqSweepRunOutput::DensityKrylovTol {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        axis: "krylov_tol",
        rows: rows_payload,
        wall_time_ms: elapsed,
    }
}

fn run_sweep_seed_density_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> TpqSweepRunOutput {
    use crate::spectrum::linear_operator::LinearOperator;
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_ext,
    );
    let dim = jw.dim();
    let beta = tpq_cfg.beta.unwrap_or(cfg.hamiltonian.beta);
    let spec = krylov_spec_from(tpq_cfg);
    let num_sites = cfg.hamiltonian.num_sites;
    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepSeedRow> = Vec::with_capacity(sweep_cfg.values.len());
    let mut m_log: Vec<usize> = Vec::new();
    for &seed_f in &sweep_cfg.values {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let seed = seed_f as u64;
        let (density, krylov_stats) =
            tpq::tpq_density_matrix_free(&jw, beta, tpq_cfg.n_samples, seed, spec);
        m_log.push(krylov_stats.min_m);
        m_log.push(krylov_stats.max_m);
        rows_payload.push(TpqSweepSeedRow { seed, density });
    }
    let elapsed = start.elapsed().as_millis();
    let agg_stats = KrylovStats::from_samples(&m_log);

    let n_seeds = rows_payload.len();
    let mut mean_density = vec![0.0_f64; num_sites];
    for row in &rows_payload {
        for (i, &x) in row.density.iter().enumerate() {
            mean_density[i] += x;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n_f = n_seeds as f64;
    for x in &mut mean_density {
        *x /= n_f;
    }
    let mut variance = vec![0.0_f64; num_sites];
    for row in &rows_payload {
        for (i, &x) in row.density.iter().enumerate() {
            let d = x - mean_density[i];
            variance[i] += d * d;
        }
    }
    let denom = if n_seeds > 1 {
        #[allow(clippy::cast_precision_loss)]
        let nm1 = (n_seeds - 1) as f64;
        nm1 * n_f
    } else {
        1.0
    };
    let stderr_density: Vec<f64> = variance.iter().map(|v| (v / denom).sqrt()).collect();

    TpqSweepRunOutput::DensitySeedSweep {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        axis: "seed",
        rows: rows_payload,
        ensemble_summary: TpqEnsembleSummary {
            mean_density,
            stderr_density,
        },
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson::from(agg_stats),
    }
}

fn run_sweep_beta_theta_2_ed(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> Result<TpqSweepRunOutput> {
    let v_init = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let v_final = resolve_v_final(cfg)?;
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let ed_init = ed::canonical_thermal(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_init,
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
    let betas: Vec<f64> = sweep_cfg.values.clone();
    let start = Instant::now();
    let rows_payload: Vec<TpqSweepBetaTheta2Row> = betas
        .iter()
        .map(|&beta| TpqSweepBetaTheta2Row {
            beta,
            theta_2: ed::exact_theta_2(&ed_init, &ed_final, beta),
        })
        .collect();
    let elapsed = start.elapsed().as_millis();
    Ok(TpqSweepRunOutput::Theta2 {
        source: "ed",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        axis: "beta",
        rows: rows_payload,
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson {
            min_m: 0,
            max_m: 0,
            mean_m: 0.0,
        },
    })
}

fn run_sweep_beta_theta_2_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> Result<TpqSweepRunOutput> {
    use crate::spectrum::linear_operator::LinearOperator;
    let v_init = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let v_final = resolve_v_final(cfg)?;
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw_init = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_init,
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
    let betas: Vec<f64> = sweep_cfg.values.clone();
    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepBetaTheta2Row> = Vec::with_capacity(betas.len());
    let mut m_log: Vec<usize> = Vec::new();
    for &beta in &betas {
        let (theta_2, effective_m) = tpq_cfg.theta_2_lanczos_tol.map_or_else(
            || ed::exact_theta_2_matrix_free(&jw_init, &jw_final, beta, k_states, krylov_m),
            |tol| {
                let max_m = tpq_cfg.krylov_m.unwrap_or(80).max(k_states).min(dim);
                ed::exact_theta_2_matrix_free_adaptive(
                    &jw_init, &jw_final, beta, k_states, tol, max_m,
                )
            },
        );
        m_log.push(effective_m);
        rows_payload.push(TpqSweepBetaTheta2Row { beta, theta_2 });
    }
    let elapsed = start.elapsed().as_millis();
    let agg_stats = KrylovStats::from_samples(&m_log);
    Ok(TpqSweepRunOutput::Theta2 {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        seed: tpq_cfg.seed,
        axis: "beta",
        rows: rows_payload,
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson::from(agg_stats),
    })
}

fn run_sweep_cartesian_beta_seed_density_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
    secondary_axis: crate::config::TpqSweepAxis,
    secondary_values: &[f64],
) -> TpqSweepRunOutput {
    use crate::spectrum::linear_operator::LinearOperator;
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_ext,
    );
    let dim = jw.dim();
    let spec = krylov_spec_from(tpq_cfg);

    let (betas_owned, seeds_owned, primary_is_beta): (Vec<f64>, Vec<u64>, bool) =
        if matches!(sweep_cfg.axis, crate::config::TpqSweepAxis::Beta) {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let seeds: Vec<u64> = secondary_values.iter().map(|&v| v as u64).collect();
            (sweep_cfg.values.clone(), seeds, true)
        } else {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let seeds: Vec<u64> = sweep_cfg.values.iter().map(|&v| v as u64).collect();
            (secondary_values.to_vec(), seeds, false)
        };
    let _ = secondary_axis;

    let (axis_primary, axis_secondary): (&'static str, &'static str) = if primary_is_beta {
        ("beta", "seed")
    } else {
        ("seed", "beta")
    };

    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepCartesianDensityRow> =
        Vec::with_capacity(betas_owned.len() * seeds_owned.len());
    let mut m_log: Vec<usize> = Vec::new();
    if primary_is_beta {
        for &beta in &betas_owned {
            for &seed in &seeds_owned {
                let (density, krylov_stats) =
                    tpq::tpq_density_matrix_free(&jw, beta, tpq_cfg.n_samples, seed, spec);
                m_log.push(krylov_stats.min_m);
                m_log.push(krylov_stats.max_m);
                #[allow(clippy::cast_precision_loss)]
                let seed_f = seed as f64;
                rows_payload.push(TpqSweepCartesianDensityRow {
                    axis_primary,
                    value_primary: beta,
                    axis_secondary,
                    value_secondary: seed_f,
                    density,
                });
            }
        }
    } else {
        for &seed in &seeds_owned {
            for &beta in &betas_owned {
                let (density, krylov_stats) =
                    tpq::tpq_density_matrix_free(&jw, beta, tpq_cfg.n_samples, seed, spec);
                m_log.push(krylov_stats.min_m);
                m_log.push(krylov_stats.max_m);
                #[allow(clippy::cast_precision_loss)]
                let seed_f = seed as f64;
                rows_payload.push(TpqSweepCartesianDensityRow {
                    axis_primary,
                    value_primary: seed_f,
                    axis_secondary,
                    value_secondary: beta,
                    density,
                });
            }
        }
    }
    let elapsed = start.elapsed().as_millis();
    let agg_stats = KrylovStats::from_samples(&m_log);

    TpqSweepRunOutput::CartesianDensity {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        axis_primary,
        axis_secondary,
        rows: rows_payload,
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson::from(agg_stats),
    }
}

fn run_sweep_seed_work_mf(
    cfg: &Config,
    tpq_cfg: &TpqSpec,
    sweep_cfg: &crate::config::TpqSweep,
) -> Result<TpqSweepRunOutput> {
    use crate::spectrum::linear_operator::LinearOperator;
    let v_init = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let v_final = resolve_v_final(cfg)?;
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;
    let jw_init = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        &v_init,
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
    let beta = tpq_cfg.beta.unwrap_or(cfg.hamiltonian.beta);
    let start = Instant::now();
    let mut rows_payload: Vec<TpqSweepSeedWorkRow> = Vec::with_capacity(sweep_cfg.values.len());
    let mut m_log: Vec<usize> = Vec::new();
    for &seed_f in &sweep_cfg.values {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let seed = seed_f as u64;
        let (stats, krylov_stats) = tpq::tpq_work_statistics_matrix_free(
            &jw_init,
            &jw_final,
            beta,
            tpq_cfg.n_samples,
            seed,
            spec,
        );
        m_log.push(krylov_stats.min_m);
        m_log.push(krylov_stats.max_m);
        rows_payload.push(TpqSweepSeedWorkRow {
            seed,
            mean_w: stats.mean_w,
            work_variance: stats.work_variance,
            mean_w_stderr: stats.mean_w_stderr,
        });
    }
    let elapsed = start.elapsed().as_millis();
    let agg_stats = KrylovStats::from_samples(&m_log);

    let n_seeds = rows_payload.len();
    #[allow(clippy::cast_precision_loss)]
    let n_f = n_seeds as f64;
    let mean_w_mean = if n_seeds > 0 {
        rows_payload.iter().map(|r| r.mean_w).sum::<f64>() / n_f
    } else {
        0.0
    };
    let variance = rows_payload
        .iter()
        .map(|r| {
            let d = r.mean_w - mean_w_mean;
            d * d
        })
        .sum::<f64>();
    let mean_w_stderr_across_seeds = if n_seeds > 1 {
        #[allow(clippy::cast_precision_loss)]
        let nm1 = (n_seeds - 1) as f64;
        (variance / (nm1 * n_f)).sqrt()
    } else {
        0.0
    };

    Ok(TpqSweepRunOutput::WorkStatisticsSeedSweep {
        source: "matrix_free",
        dim,
        n_samples: tpq_cfg.n_samples,
        axis: "seed",
        rows: rows_payload,
        ensemble_summary: TpqWorkEnsembleSummary {
            mean_w_mean,
            mean_w_stderr_across_seeds,
        },
        wall_time_ms: elapsed,
        krylov_stats: KrylovStatsJson::from(agg_stats),
    })
}

fn write_sweep_output_json(path: &Path, out: &TpqSweepRunOutput) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|source| ScrapboxError::Artifact {
        path: path.to_path_buf(),
        message: format!("failed to write tpq_sweep_report.json: {source}"),
    })?;
    serde_json::to_writer_pretty(file, out)?;
    Ok(())
}

fn write_output_json(path: &Path, out: &TpqRunOutput) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|source| ScrapboxError::Artifact {
        path: path.to_path_buf(),
        message: format!("failed to write tpq_report.json: {source}"),
    })?;
    serde_json::to_writer_pretty(file, out)?;
    Ok(())
}
