//! `scrapbox` CLI entry point.
//!
//! Subcommands (per `notes/discipline/HARNESS.md`):
//!
//! - `run <config>`      single SCF calculation + dump
//! - `validate <config>` `run` + reference-dataset comparison
//! - `sweep <config>`    cartesian-product over `[sweep].axes`      (Batch 15)
//! - `bench <config>`    `run` + wall-clock timing                  (later)
//! - `doctor <config>`   parse + dispatch only                      (later)

use scrapbox::config::{Config, Quench};
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::observables::ObservableReport;
use scrapbox::output;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::validation::{compare as validate_compare, ReferenceDataset};
use scrapbox::xc::ExchangeCorrelation;
use scrapbox::{Result, ScrapboxError};
use std::env;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "\
scrapbox — canonical-ensemble finite-T DFT runner

USAGE:
    scrapbox <subcommand> <config.toml>

SUBCOMMANDS:
    run         Single calculation (SCF + observables + dump)
    validate    `run` + reference-dataset comparison
    sweep       Cartesian-product parameter grid (v0.2)
    bench       `run` + wall-clock timing (later)
    doctor      Parse + dispatch only (later)

See notes/discipline/HARNESS.md for the full design.
";

fn main() -> ExitCode {
    tracing_subscriber::fmt::try_init().ok();
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let subcommand = &args[1];
    let config_path = &args[2];

    match subcommand.as_str() {
        "run" => dispatch(run_subcommand(config_path)),
        "validate" => match validate_subcommand(config_path) {
            Ok(true) => ExitCode::from(0),
            Ok(false) => ExitCode::from(1),
            Err(e) => {
                eprintln!("scrapbox validate: {e}");
                ExitCode::from(1)
            }
        },
        "sweep" => dispatch(sweep_subcommand(config_path)),
        "bench" | "doctor" => match Config::from_file(config_path) {
            Ok(cfg) => {
                eprintln!(
                    "scrapbox {subcommand}: parsed config '{name}' \
                     (subcommand body not implemented yet).",
                    name = cfg.meta.name
                );
                ExitCode::from(2)
            }
            Err(e) => {
                eprintln!("scrapbox {subcommand}: {e}");
                ExitCode::from(1)
            }
        },
        _ => {
            eprintln!("scrapbox: unknown subcommand '{subcommand}'\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(r: Result<()>) -> ExitCode {
    match r {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("scrapbox: {e}");
            ExitCode::from(1)
        }
    }
}

fn run_subcommand(config_path: &str) -> Result<()> {
    let cfg = Config::from_file(config_path)?;
    let ks_state = run_initial_solve(&cfg)?;
    let observables = compute_observables(&cfg, &ks_state)?;
    let out_dir = resolve_output_dir(&cfg);
    output::write_run_outputs_with_config(&out_dir, &ks_state, &observables, &cfg.output, &cfg)?;
    eprintln!(
        "scrapbox run: solved '{name}' in {iters} iterations (residual {res:e}) → {dir}",
        name = cfg.meta.name,
        iters = ks_state.iterations,
        res = ks_state.residual,
        dir = out_dir.display(),
    );
    Ok(())
}

fn run_initial_solve(cfg: &Config) -> Result<scrapbox::scf::KsState> {
    let h = KohnShamHamiltonian::from_config(&cfg.hamiltonian)?;
    let xc = ExchangeCorrelation::from_config(&cfg.xc_functional, &h)?;
    let spectrum = SpectrumSource::from_config(&cfg.spectrum_source)?;
    let density = CanonicalDensityEvaluator::from_config(&cfg.density_evaluator)?;
    let solver = CanonicalThermalDFTSolver::new(h, xc, spectrum, density, cfg.scf.clone());
    solver.solve()
}

/// Finite-difference susceptibility `χ_ij = ∂n_i / ∂V_j` at the supplied
/// base potential, using central differences with step `epsilon`.
/// Returns a `num_sites × num_sites` row-major matrix.
fn susceptibility_finite_difference(
    cfg: &Config,
    base_potential: &[f64],
    epsilon: f64,
) -> Result<Vec<Vec<f64>>> {
    use scrapbox::config::ExternalPotential;
    let n = cfg.hamiltonian.num_sites;
    let mut chi = vec![vec![0.0_f64; n]; n];
    for j in 0..n {
        let mut v_plus = base_potential.to_vec();
        v_plus[j] += epsilon;
        let mut v_minus = base_potential.to_vec();
        v_minus[j] -= epsilon;
        let mut cfg_plus = cfg.clone();
        cfg_plus.hamiltonian.external_potential = ExternalPotential::Explicit { values: v_plus };
        cfg_plus.quench = None;
        let mut cfg_minus = cfg.clone();
        cfg_minus.hamiltonian.external_potential = ExternalPotential::Explicit { values: v_minus };
        cfg_minus.quench = None;
        let ks_plus = run_initial_solve(&cfg_plus)?;
        let ks_minus = run_initial_solve(&cfg_minus)?;
        for i in 0..n {
            chi[i][j] = (ks_plus.densities[i] - ks_minus.densities[i]) / (2.0 * epsilon);
        }
    }
    Ok(chi)
}

fn compute_observables(
    cfg: &Config,
    ks_state: &scrapbox::scf::KsState,
) -> Result<ObservableReport> {
    let mut report = ObservableReport::default();
    if cfg.observables.free_energy {
        report.free_energy = Some(ks_state.free_energy);
    }
    if cfg.observables.partition_function {
        report.partition_function = Some(ks_state.partition_function);
    }

    let want_quench_obs = cfg.observables.mean_work
        || cfg.observables.irreversible_entropy
        || cfg.observables.work_variance;
    if !want_quench_obs {
        return Ok(report);
    }

    let Some(Quench::Sudden {
        final_external_potential,
    }) = cfg.quench.clone()
    else {
        return Err(ScrapboxError::ConfigValidation {
            message:
                "[observables] requested mean_work / irreversible_entropy / work_variance but \
                 no [quench] section is present"
                    .into(),
        });
    };

    let mut cfg_final = cfg.clone();
    cfg_final.hamiltonian.external_potential = final_external_potential;
    cfg_final.quench = None;
    cfg_final.observables.mean_work = false;
    cfg_final.observables.irreversible_entropy = false;
    cfg_final.observables.work_variance = false;
    let ks_final = run_initial_solve(&cfg_final)?;

    let initial_potential = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let final_potential = cfg_final
        .hamiltonian
        .external_potential
        .to_site_values(cfg_final.hamiltonian.num_sites);
    let beta = cfg.hamiltonian.beta;
    let evaluator = scrapbox::quench::SuddenQuenchEvaluator::new(
        initial_potential.clone(),
        final_potential.clone(),
        beta,
    );
    let mean_work = evaluator.mean_work(&ks_state.densities);
    let s_irr =
        evaluator.irreversible_entropy(mean_work, ks_state.free_energy, ks_final.free_energy);

    if cfg.observables.mean_work {
        report.mean_work = Some(mean_work);
    }
    if cfg.observables.irreversible_entropy {
        report.irreversible_entropy = Some(s_irr);
    }

    if cfg.observables.work_variance {
        let epsilon = 1e-4;
        let chi = susceptibility_finite_difference(cfg, &initial_potential, epsilon)?;

        let delta: Vec<f64> = initial_potential
            .iter()
            .zip(final_potential.iter())
            .map(|(v0, vf)| vf - v0)
            .collect();

        // <W²>_c = <W>² - (1/β) Σ_ij δλ_i δλ_j χ_ij
        let mut quad = 0.0_f64;
        for i in 0..delta.len() {
            for j in 0..delta.len() {
                quad += delta[i] * chi[i][j] * delta[j];
            }
        }
        let mean_w_sq_c = mean_work.mul_add(mean_work, -quad / beta);

        let theta_2 = match cfg.observables.theta_2.method.as_str() {
            "zero" => 0.0,
            "lda" => {
                return Err(ScrapboxError::ConfigValidation {
                    message: "[observables].theta_2.method = \"lda\" requires a Bethe-ansatz \
                              homogeneous reference and is gated to v0.3+ (see PHASES.md). \
                              Use \"zero\" until BALDA lands."
                        .into(),
                });
            }
            other => {
                return Err(ScrapboxError::ConfigValidation {
                    message: format!(
                        "[observables].theta_2.method = {other:?} is not a recognized value \
                         (supported: \"zero\")"
                    ),
                });
            }
        };

        let mean_w_sq = mean_w_sq_c + theta_2;
        let sigma_w_sq = mean_work.mul_add(-mean_work, mean_w_sq);
        let fdr_pred = (0.5 * beta * beta).mul_add(sigma_w_sq - theta_2, 0.0);
        let fdr_residual = s_irr - fdr_pred;

        report.mean_work_sq = Some(mean_w_sq);
        report.work_variance = Some(sigma_w_sq);
        report.theta_2 = Some(theta_2);
        report.fdr_residual = Some(fdr_residual);
    }

    Ok(report)
}

fn resolve_output_dir(cfg: &Config) -> std::path::PathBuf {
    let mut s = cfg.output.directory.clone();
    s = s.replace("{meta.name}", &cfg.meta.name);
    Path::new(&s).to_path_buf()
}

fn validate_subcommand(config_path: &str) -> Result<bool> {
    let cfg = Config::from_file(config_path)?;
    let ks_state = run_initial_solve(&cfg)?;
    let observables = compute_observables(&cfg, &ks_state)?;

    let validation_cfg =
        cfg.validation
            .as_ref()
            .ok_or_else(|| ScrapboxError::ConfigValidation {
                message: "[validation] section required for `scrapbox validate` (set tolerances + \
                      reference_path or use `scrapbox run`)"
                    .into(),
            })?;
    let reference = ReferenceDataset::from_file(&validation_cfg.reference_path)?;

    let out_dir = resolve_output_dir(&cfg);
    output::write_run_outputs_with_config(&out_dir, &ks_state, &observables, &cfg.output, &cfg)?;

    let report = validate_compare(
        &observables,
        &ks_state.densities,
        &reference,
        validation_cfg,
    )?;

    let report_path = out_dir.join("validation_report.json");
    let file = std::fs::File::create(&report_path).map_err(|source| ScrapboxError::Artifact {
        path: report_path.clone(),
        message: format!("failed to write validation_report.json: {source}"),
    })?;
    serde_json::to_writer_pretty(file, &report)?;

    for row in &report.residuals {
        let status = if row.passed { "PASS" } else { "FAIL" };
        eprintln!(
            "  {name:<22} {status:<4}  residual = {residual:e}  tolerance = {tolerance:e}",
            name = row.name,
            status = status,
            residual = row.residual,
            tolerance = row.tolerance,
        );
    }

    if !report.all_passed && validation_cfg.fail_on_mismatch {
        eprintln!("scrapbox validate: FAIL → {dir}", dir = out_dir.display());
        return Ok(false);
    }

    eprintln!("scrapbox validate: PASS → {dir}", dir = out_dir.display());
    Ok(true)
}

fn sweep_subcommand(config_path: &str) -> Result<()> {
    let cfg = Config::from_file(config_path)?;
    scrapbox::sweep::run(&cfg)
}
