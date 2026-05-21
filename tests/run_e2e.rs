//! End-to-end integration test: build everything from a config, solve,
//! write outputs, read them back, and assert sanity.

use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::observables::ObservableReport;
use scrapbox::output::write_run_outputs_with_config;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::xc::ExchangeCorrelation;
use serde_json::Value;

const DIMER_TOML: &str = r#"
schema_version = "0.1"

[meta]
name = "dimer_e2e"

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 2
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 1
beta = 2.0
external_potential.kind = "uniform"
external_potential.amplitude = 0.0

[xc_functional]
kind = "hubbard_lda"

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 200
tolerance = 1e-10
mixing.kind = "linear"
mixing.alpha = 0.5
initial_density.kind = "uniform"

[observables]
mean_work = false
irreversible_entropy = false
free_energy = true
partition_function = true

[output]
directory = "ignored_in_test"
format = "json"
dump_density = true
dump_spectrum = true
dump_partition_function = true
overwrite = false
"#;

#[test]
fn e2e_dimer_writes_expected_outputs() {
    let cfg = Config::from_toml_str(DIMER_TOML).unwrap();
    let h = KohnShamHamiltonian::from_config(&cfg.hamiltonian).unwrap();
    let xc = ExchangeCorrelation::from_config(&cfg.xc_functional, &h).unwrap();
    let spec = SpectrumSource::from_config(&cfg.spectrum_source).unwrap();
    let dens = CanonicalDensityEvaluator::from_config(&cfg.density_evaluator).unwrap();
    let solver = CanonicalThermalDFTSolver::new(h, xc, spec, dens, cfg.scf.clone());
    let ks = solver.solve().expect("dimer SCF should converge");

    // Pratt sum rule: Σn = N (ACCEPTANCE.md §1.2)
    let total: f64 = ks.densities.iter().sum();
    assert!((total - 2.0).abs() < 1e-10);

    // Particle-hole symmetric → densities should be (1, 1).
    for &n in &ks.densities {
        assert!((n - 1.0).abs() < 1e-8);
    }

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("out");
    let observables = ObservableReport {
        free_energy: Some(ks.free_energy),
        partition_function: Some(ks.partition_function),
        ..ObservableReport::default()
    };
    write_run_outputs_with_config(&dir, &ks, &observables, &cfg.output, &cfg).unwrap();

    // run.toml must exist and re-deserialize back into Config.
    let reloaded = std::fs::read_to_string(dir.join("run.toml")).unwrap();
    let cfg_roundtrip: Config = toml::from_str(&reloaded).unwrap();
    assert_eq!(cfg_roundtrip.schema_version, "0.1");
    assert_eq!(cfg_roundtrip.meta.name, "dimer_e2e");

    // observables.json mirrors what we wrote.
    let obs_raw = std::fs::read_to_string(dir.join("observables.json")).unwrap();
    let obs_json: Value = serde_json::from_str(&obs_raw).unwrap();
    let f = obs_json["free_energy"].as_f64().expect("free_energy float");
    assert!((f - ks.free_energy).abs() < 1e-12);
}
