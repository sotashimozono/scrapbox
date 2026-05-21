//! `scrapbox` CLI entry point.
//!
//! Subcommands (per `notes/discipline/HARNESS.md`):
//!
//! - `run <config>`      single SCF calculation + dump
//! - `validate <config>` `run` + reference-dataset comparison       (Batch 8)
//! - `sweep <config>`    cartesian-product over `[sweep].axes`      (Batch 8+)
//! - `bench <config>`    `run` + wall-clock timing                  (Batch 8+)
//! - `doctor <config>`   parse + dispatch only                      (Batch 8+)

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
    validate    `run` + reference-dataset comparison (Batch 8)
    sweep       Cartesian-product parameter grid (Batch 8+)
    bench       `run` + wall-clock timing (Batch 8+)
    doctor      Parse + dispatch only (Batch 8+)

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
        "sweep" | "bench" | "doctor" => match Config::from_file(config_path) {
            Ok(cfg) => {
                eprintln!(
                    "scrapbox {subcommand}: parsed config '{name}' \
                     (subcommand body lands in Batch 8 — see PHASES.md).",
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

    if cfg.observables.mean_work || cfg.observables.irreversible_entropy {
        let Some(Quench::Sudden {
            final_external_potential,
        }) = cfg.quench.clone()
        else {
            return Err(ScrapboxError::ConfigValidation {
                message: "[observables] requested mean_work / irreversible_entropy but \
                          no [quench] section is present"
                    .into(),
            });
        };

        // Solve the post-quench Hamiltonian (same model, swapped external potential).
        let mut cfg_final = cfg.clone();
        cfg_final.hamiltonian.external_potential = final_external_potential;
        cfg_final.quench = None;
        cfg_final.observables.mean_work = false;
        cfg_final.observables.irreversible_entropy = false;
        let ks_final = run_initial_solve(&cfg_final)?;

        let initial_potential = cfg
            .hamiltonian
            .external_potential
            .to_site_values(cfg.hamiltonian.num_sites);
        let final_potential = cfg_final
            .hamiltonian
            .external_potential
            .to_site_values(cfg_final.hamiltonian.num_sites);
        let evaluator = scrapbox::quench::SuddenQuenchEvaluator::new(
            initial_potential,
            final_potential,
            cfg.hamiltonian.beta,
        );
        let mean_work = evaluator.mean_work(&ks_state.densities);
        if cfg.observables.mean_work {
            report.mean_work = Some(mean_work);
        }
        if cfg.observables.irreversible_entropy {
            let s_irr = evaluator.irreversible_entropy(
                mean_work,
                ks_state.free_energy,
                ks_final.free_energy,
            );
            report.irreversible_entropy = Some(s_irr);
        }
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

    // Dump the per-observable report into the output dir.
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
