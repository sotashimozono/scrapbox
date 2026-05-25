//! Layer 7 - `scrapbox doctor`: parse a config, construct every layer
//! without solving the SCF, and report what each layer dispatches to.

use std::path::Path;

use crate::config::{Config, Mixing, Quench};
use crate::density::CanonicalDensityEvaluator;
use crate::error::{Result, ScrapboxError};
use crate::hamiltonian::KohnShamHamiltonian;
use crate::spectrum::SpectrumSource;
use crate::validation::ReferenceDataset;
use crate::xc::ExchangeCorrelation;

/// Run the doctor pass on the config at `config_path`.
pub fn run(config_path: &Path) -> Result<()> {
    let cfg = Config::from_file(config_path)?;
    let mut report = Vec::<String>::new();

    // Canonicalize so CI artifacts pulled to a different machine can still
    // be traced back to the source config. Falls back to the original if
    // the path cant be resolved (deleted between parse and now).
    let resolved_config = config_path
        .canonicalize()
        .unwrap_or_else(|_| config_path.to_path_buf());
    report.push(format!("config: {}", resolved_config.display()));
    report.push(format!("schema_version: {}", cfg.schema_version));
    report.push(format!("meta.name: {}", cfg.meta.name));
    report.push(format!(
        "hamiltonian: L={}, J={:.3}, U={:.3}, beta={:.3}, N_per_spin={}",
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
        "scf: max_iter={}, tol={:e}, mixing={}",
        cfg.scf.max_iterations,
        cfg.scf.tolerance,
        mixing_kind_name(&cfg.scf.mixing),
    ));

    if let Some(Quench::Sudden { .. }) = &cfg.quench {
        report.push(format!(
            "quench: sudden, expected final_V length = {}",
            cfg.hamiltonian.num_sites
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

    if let Some(validation_cfg) = &cfg.validation {
        // Doctor is the layer that should catch missing reference files
        // before SCF burns minutes only to fail at validate time.
        match ReferenceDataset::from_file(&validation_cfg.reference_path) {
            Ok(_) => report.push(format!(
                "validation: present, reference_path={} (readable)",
                validation_cfg.reference_path.display()
            )),
            Err(e) => {
                report.push(format!(
                    "validation: present, reference_path={} (UNREADABLE: {e})",
                    validation_cfg.reference_path.display()
                ));
                return Err(e);
            }
        }
    } else {
        report.push("validation: none".into());
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

    // Drop "status" - presence of the file is the success signal.
    // Absence could mean either "doctor never ran" or "doctor failed
    // before write"; a hardcoded "ok" would mislead either way.
    let report_json = serde_json::json!({
        "config_path": resolved_config.display().to_string(),
        "lines": report,
    });
    // Write next to the config (not into output.directory) so doctor does
    // not side-effect the production runs/ tree just to surface a report.
    let report_path = config_path.with_extension("doctor_report.json");
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
        ExchangeCorrelation::BaldaFiniteT(_) => "balda_finite_t",
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

fn mixing_kind_name(m: &Mixing) -> &'static str {
    match m {
        Mixing::Linear { .. } => "linear",
        Mixing::Pulay { .. } => "pulay",
    }
}
