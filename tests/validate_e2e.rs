//! End-to-end validation: exercise `validation::compare` on the dimer
//! reference dataset. Verifies that the live runner reproduces the
//! checked-in numbers (regression guard per ACCEPTANCE.md §1.1).

use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::observables::ObservableReport;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::validation::{compare, ReferenceDataset};
use scrapbox::xc::ExchangeCorrelation;

fn run_dimer() -> (ObservableReport, Vec<f64>, Config) {
    let cfg_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("configs")
        .join("dimer_validate.toml");
    let cfg = Config::from_file(&cfg_path).expect("config");
    let h = KohnShamHamiltonian::from_config(&cfg.hamiltonian).unwrap();
    let xc = ExchangeCorrelation::from_config(&cfg.xc_functional, &h).unwrap();
    let spec = SpectrumSource::from_config(&cfg.spectrum_source).unwrap();
    let dens = CanonicalDensityEvaluator::from_config(&cfg.density_evaluator).unwrap();
    let ks = CanonicalThermalDFTSolver::new(h, xc, spec, dens, cfg.scf.clone())
        .solve()
        .expect("dimer SCF");
    let observables = ObservableReport {
        free_energy: Some(ks.free_energy),
        partition_function: Some(ks.partition_function),
        ..ObservableReport::default()
    };
    (observables, ks.densities, cfg)
}

#[test]
fn dimer_matches_self_reference() {
    let (observables, density, cfg) = run_dimer();
    let validation_cfg = cfg.validation.as_ref().expect("[validation] section");
    let ref_path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&validation_cfg.reference_path);
    let reference = ReferenceDataset::from_file(&ref_path).expect("load reference");
    let report = compare(&observables, &density, &reference, validation_cfg).unwrap();
    assert!(
        report.all_passed,
        "dimer_smoke.json regression — report = {report:#?}"
    );
}

#[test]
fn tampered_reference_density_fails() {
    let (observables, _density, cfg) = run_dimer();
    let validation_cfg = cfg.validation.as_ref().unwrap();
    let reference = ReferenceDataset {
        schema_version: "0.2".into(),
        produced_by: "tampered".into(),
        observables: ObservableReport::default(),
        site_density: Some(vec![0.5, 1.5]),
    };
    let report = compare(&observables, &[1.0, 1.0], &reference, validation_cfg).unwrap();
    assert!(!report.all_passed);
}
