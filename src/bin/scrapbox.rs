//! `scrapbox` CLI entry point.
//!
//! Subcommands (per `notes/discipline/HARNESS.md`):
//!
//! - `run <config>`      single SCF calculation + dump
//! - `validate <config>` `run` + reference-dataset comparison
//! - `sweep <config>`    cartesian-product over `[sweep].axes`      (Batch 15)
//! - `bench <config>`    `run` + wall-clock timing                  (later)
//! - `doctor <config>`   parse + dispatch only                      (later)

use scrapbox::config::Config;
use scrapbox::output;
use scrapbox::validation::{compare as validate_compare, ReferenceDataset};
use scrapbox::{Result, ScrapboxError};
use std::env;
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
    ed          Many-body Hubbard ED dispatcher (dense / matrix-free Lanczos)
    tpq         TPQ analysis dispatcher (density / work statistics; ed / matrix-free)

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
        "bench" => dispatch(bench_subcommand(config_path)),
        "doctor" => dispatch(doctor_subcommand(config_path)),
        "ed" => dispatch(ed_subcommand(config_path)),
        "tpq" => dispatch(tpq_subcommand(config_path)),
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
    scrapbox::bin_support::solve_and_write(&cfg)
}

fn validate_subcommand(config_path: &str) -> Result<bool> {
    let cfg = Config::from_file(config_path)?;
    let ks_state = scrapbox::bin_support::run_initial_solve(&cfg)?;
    let observables = scrapbox::bin_support::compute_observables(&cfg, &ks_state)?;

    let validation_cfg =
        cfg.validation
            .as_ref()
            .ok_or_else(|| ScrapboxError::ConfigValidation {
                message: "[validation] section required for `scrapbox validate` (set tolerances + \
                      reference_path or use `scrapbox run`)"
                    .into(),
            })?;
    let reference = ReferenceDataset::from_file(&validation_cfg.reference_path)?;

    let out_dir = scrapbox::bin_support::resolve_output_dir(&cfg);
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
    scrapbox::sweep::run(std::path::Path::new(config_path))
}

fn bench_subcommand(config_path: &str) -> Result<()> {
    scrapbox::bench::run(std::path::Path::new(config_path))
}

fn doctor_subcommand(config_path: &str) -> Result<()> {
    scrapbox::doctor::run(std::path::Path::new(config_path))
}

fn ed_subcommand(config_path: &str) -> Result<()> {
    let cfg = Config::from_file(config_path)?;
    let out = scrapbox::many_body_ed::run(&cfg)?;
    let resolved = scrapbox::bin_support::resolve_output_dir(&cfg);
    eprintln!(
        "scrapbox ed: solver = {solver}, dim = {dim}, returned = {k} (wall {ms} ms) -> {dir}",
        solver = out.solver,
        dim = out.dim,
        k = out.num_eigenvalues_returned,
        ms = out.wall_time_ms,
        dir = resolved.display()
    );
    Ok(())
}

fn tpq_subcommand(config_path: &str) -> Result<()> {
    use scrapbox::many_body_tpq::TpqRunOutput;
    let cfg = Config::from_file(config_path)?;
    let out = scrapbox::many_body_tpq::run(&cfg)?;
    let resolved = scrapbox::bin_support::resolve_output_dir(&cfg);
    match &out {
        TpqRunOutput::Density {
            source,
            dim,
            n_samples,
            wall_time_ms,
            krylov_stats,
            ..
        } => {
            let krylov = krylov_stats
                .map(|k| {
                    format!(
                        " [krylov m: min={}, max={}, mean={:.1}]",
                        k.min_m, k.max_m, k.mean_m
                    )
                })
                .unwrap_or_default();
            eprintln!(
                "scrapbox tpq: kind = density, source = {source}, dim = {dim}, samples = {n_samples}{krylov} (wall {wall_time_ms} ms) -> {dir}",
                dir = resolved.display()
            );
        }
        TpqRunOutput::WorkStatistics {
            source,
            dim,
            n_samples,
            mean_w,
            work_variance,
            mean_w_stderr,
            wall_time_ms,
            krylov_stats,
            ..
        } => {
            let krylov = krylov_stats
                .map(|k| {
                    format!(
                        " [krylov m: min={}, max={}, mean={:.1}]",
                        k.min_m, k.max_m, k.mean_m
                    )
                })
                .unwrap_or_default();
            eprintln!(
                "scrapbox tpq: kind = work_statistics, source = {source}, dim = {dim}, samples = {n_samples}, <W> = {mean_w:.6} (+/- {mean_w_stderr:.4}), sigma_W^2 = {work_variance:.6}{krylov} (wall {wall_time_ms} ms) -> {dir}",
                dir = resolved.display()
            );
        }
        TpqRunOutput::Theta2 {
            source,
            dim,
            beta,
            theta_2,
            wall_time_ms,
            krylov_stats,
            ..
        } => {
            let krylov = krylov_stats
                .map(|k| {
                    format!(
                        " [krylov m: min={}, max={}, mean={:.1}]",
                        k.min_m, k.max_m, k.mean_m
                    )
                })
                .unwrap_or_default();
            eprintln!(
                "scrapbox tpq: kind = theta_2, source = {source}, dim = {dim}, beta = {beta}, theta_2 = {theta_2:.6}{krylov} (wall {wall_time_ms} ms) -> {dir}",
                dir = resolved.display()
            );
        }
    }
    Ok(())
}
