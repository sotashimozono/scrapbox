//! Layer 7 - `scrapbox doctor`: parse a config, construct every layer
//! without solving the SCF, and report what each layer dispatches to.

use std::path::Path;

use crate::config::{Config, Quench};
use crate::density::CanonicalDensityEvaluator;
use crate::error::{Result, ScrapboxError};
use crate::hamiltonian::KohnShamHamiltonian;
use crate::spectrum::SpectrumSource;
use crate::xc::ExchangeCorrelation;

/// Run the doctor pass on the config at `config_path`.
pub fn run(config_path: &Path) -> Result<()> {
    let cfg = Config::from_file(config_path)?;
    let mut report = Vec::<String>::new();

    report.push(format!("config: {}", config_path.display()));
    report.push(format!("schema_version: {}", cfg.schema_version));
    report.push(format!("meta.name: {}", cfg.meta.name));
    report.push(format!(
        "hamiltonian: L={}, J={}, U={}, beta={}, N_per_spin={}",
        cfg.hamiltonian.num_sites,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        cfg.hamiltonian.beta,
        cfg.hamiltonian.num_electrons_per_spin,
    ));

    let ham = KohnShamHamiltonian::from_config(&cfg.hamiltonian).map_err(|e| {
        ScrapboxError::ConfigValidation {
            message: format!("hamiltonian construction failed: {e}"),
        }
    })?;
    report.push(format!(
        "  external_potential length = {}",
        ham.external_potential.len()
    ));

    let xc = ExchangeCorrelation::from_config(&cfg.xc_functional, &ham)?;
    report.push(format!("xc: {}", xc_kind_name(&xc)));

    let spectrum = SpectrumSource::from_config(&cfg.spectrum_source)?;
    report.push(format!("spectrum: {}", spectrum_kind_name(&spectrum)));

    let density = CanonicalDensityEvaluator::from_config(&cfg.density_evaluator)?;
    report.push(format!("density: {}", density_kind_name(&density)));

    report.push(format!(
        "scf: max_iter={}, tol={}, mixing={}",
        cfg.scf.max_iterations,
        cfg.scf.tolerance,
        mixing_kind_name(&cfg.scf.mixing),
    ));

    if let Some(Quench::Sudden {
        final_external_potential,
    }) = &cfg.quench
    {
        let lengths = final_external_potential.to_site_values(cfg.hamiltonian.num_sites);
        report.push(format!(
            "quench: sudden, final_V length = {}",
            lengths.len()
        ));
    } else {
        report.push("quench: none".into());
    }

    let obs = &cfg.observables;
    report.push(format!(
        "observables: free_energy={}, partition_function={}, mean_work={}, S_irr={}, work_variance={}, theta_2.method={}",
        obs.free_energy, obs.partition_function, obs.mean_work,
        obs.irreversible_entropy, obs.work_variance, obs.theta_2.method,
    ));

    report.push(format!(
        "output: directory={}, format={:?}, overwrite={}",
        cfg.output.directory, cfg.output.format, cfg.output.overwrite,
    ));

    if cfg.validation.is_some() {
        report.push("validation: present".into());
    }
    if let Some(sweep) = &cfg.sweep {
        report.push(format!(
            "sweep: present ({} axes, parallelism = {})",
            sweep.axes.len(),
            sweep.parallelism,
        ));
    }
    if let Some(bench) = &cfg.bench {
        report.push(format!(
            "bench: warmup={}, measured={}",
            bench.warmup, bench.measured
        ));
    }

    for line in &report {
        eprintln!("  {line}");
    }

    let report_json = serde_json::json!({
        "config_path": config_path.display().to_string(),
        "lines": report,
        "status": "ok",
    });
    let out_dir = std::path::PathBuf::from(&cfg.output.directory);
    std::fs::create_dir_all(&out_dir).map_err(|source| ScrapboxError::Artifact {
        path: out_dir.clone(),
        message: format!("create doctor output dir: {source}"),
    })?;
    let report_path = out_dir.join("doctor_report.json");
    let file = std::fs::File::create(&report_path).map_err(|source| ScrapboxError::Artifact {
        path: report_path.clone(),
        message: format!("create doctor_report.json: {source}"),
    })?;
    serde_json::to_writer_pretty(file, &report_json)?;
    eprintln!("scrapbox doctor: report -> {}", report_path.display());
    Ok(())
}

fn xc_kind_name(xc: &ExchangeCorrelation) -> &'static str {
    match xc {
        ExchangeCorrelation::HubbardLda(_) => "hubbard_lda",
        ExchangeCorrelation::Balda(_) => "balda",
        ExchangeCorrelation::NonInteracting => "non_interacting",
    }
}

fn spectrum_kind_name(s: &SpectrumSource) -> &'static str {
    match s {
        SpectrumSource::DenseDiag => "dense_diag",
        SpectrumSource::Lanczos(_) => "lanczos",
    }
}

fn density_kind_name(d: &CanonicalDensityEvaluator) -> &'static str {
    match d {
        CanonicalDensityEvaluator::PrattRecursion(_) => "pratt_recursion",
        CanonicalDensityEvaluator::GcePlusProjection(_) => "gce_plus_projection",
    }
}

fn mixing_kind_name(m: &crate::config::Mixing) -> &'static str {
    match m {
        crate::config::Mixing::Linear { .. } => "linear",
        crate::config::Mixing::Pulay { .. } => "pulay",
    }
}
