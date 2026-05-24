#![allow(clippy::doc_markdown)]
//! KS-DFT vs ED consistency at L = 6. Uses `reference::ed` as ground
//! truth, extending the L = 4 coverage in `tests/ks_ed_consistency.rs`
//! to a larger Hilbert space (400 states vs 36).

use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::reference::ed;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::xc::ExchangeCorrelation;

const L: usize = 6;

fn run_ks_l6(v_ext: &[f64]) -> Vec<f64> {
    let cfg_str = format!(
        r#"
schema_version = "0.2"

[meta]
name = "ks_ed_l6"
description = "L=6 KS-DFT auxiliary for ED consistency check"
created = "2026-05-24"
tags = ["reference", "cross-check", "l6"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 6
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 3
beta = 2.0
external_potential.kind = "explicit"
external_potential.values = [{}, {}, {}, {}, {}, {}]

[xc_functional]
kind = "hubbard_lda"

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 2000
tolerance = 1e-10
mixing.kind = "linear"
mixing.alpha = 0.05
initial_density.kind = "uniform"

[observables]
free_energy = true
partition_function = true

[output]
directory = "runs/ks_ed_l6"
format = "json"
overwrite = true
"#,
        v_ext[0], v_ext[1], v_ext[2], v_ext[3], v_ext[4], v_ext[5]
    );
    let cfg = Config::from_toml_str(&cfg_str).expect("config parse");
    let h = KohnShamHamiltonian::from_config(&cfg.hamiltonian).unwrap();
    let xc = ExchangeCorrelation::from_config(&cfg.xc_functional, &h).unwrap();
    let spec = SpectrumSource::from_config(&cfg.spectrum_source).unwrap();
    let dens = CanonicalDensityEvaluator::from_config(&cfg.density_evaluator).unwrap();
    CanonicalThermalDFTSolver::new(h, xc, spec, dens, cfg.scf)
        .solve()
        .expect("KS SCF")
        .densities
}

#[test]
fn l6_uniform_v_ks_density_matches_ed_to_machine_precision() {
    let v = [0.0_f64; 6];
    let ks = run_ks_l6(&v);
    let result = ed::canonical_thermal(L, 3, 3, 1.0, 4.0, &v);
    let n_ed = ed::thermal_density(&result, 2.0);
    for i in 0..L {
        assert!(
            (ks[i] - n_ed[i]).abs() < 1e-10,
            "site {i}: KS = {}, ED = {}",
            ks[i],
            n_ed[i]
        );
    }
}

#[test]
fn l6_comb_v_ks_density_matches_ed_within_lda_error() {
    let v = [0.02_f64, -0.02, 0.02, -0.02, 0.02, -0.02];
    let ks = run_ks_l6(&v);
    let result = ed::canonical_thermal(L, 3, 3, 1.0, 4.0, &v);
    let n_ed = ed::thermal_density(&result, 2.0);
    let total_ks: f64 = ks.iter().sum();
    let total_ed: f64 = n_ed.iter().sum();
    assert!((total_ks - 6.0).abs() < 1e-8, "KS mass = {total_ks}");
    assert!((total_ed - 6.0).abs() < 1e-8, "ED mass = {total_ed}");
    for i in 0..L {
        assert!(
            (ks[i] - n_ed[i]).abs() < 0.02,
            "site {i}: KS = {}, ED = {} - LDA error exceeded 2% budget",
            ks[i],
            n_ed[i]
        );
    }
}
