#![allow(clippy::doc_markdown)]
//! KS-DFT vs ED consistency check at L = 4 with a small comb potential.
//!
//! Uses `reference::ed::canonical_thermal` as the ground-truth source.
//! Runs the configurable KS-DFT solver on the same Hamiltonian and
//! checks that the converged density agrees with ED within the
//! HubbardLDA approximation error budget.
//!
//! The V = 0 cross-check is symmetry-trivial (both methods give
//! n_i = 1 by translation invariance); this test exercises the
//! comb V case where the LDA approximation actually contributes.

use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::reference::ed;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::xc::ExchangeCorrelation;

const L: usize = 4;

fn run_ks_l4(v_ext: &[f64]) -> Vec<f64> {
    let cfg_str = format!(
        r#"
schema_version = "0.2"

[meta]
name = "ks_ed_consistency_l4"
description = "L=4 KS-DFT auxiliary for ED consistency check"
created = "2026-05-24"
tags = ["reference", "cross-check"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 4
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 2
beta = 2.0
external_potential.kind = "explicit"
external_potential.values = [{}, {}, {}, {}]

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
directory = "runs/ks_ed_consistency_l4"
format = "json"
overwrite = true
"#,
        v_ext[0], v_ext[1], v_ext[2], v_ext[3]
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
fn l4_uniform_v_ks_density_matches_ed_to_machine_precision() {
    let v = [0.0_f64; 4];
    let ks = run_ks_l4(&v);
    let result = ed::canonical_thermal(L, 2, 2, 1.0, 4.0, &v);
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
fn l4_comb_v_ks_density_matches_ed_within_lda_error() {
    let v = [0.02_f64, -0.02, 0.02, -0.02];
    let ks = run_ks_l4(&v);
    let result = ed::canonical_thermal(L, 2, 2, 1.0, 4.0, &v);
    let n_ed = ed::thermal_density(&result, 2.0);
    for i in 0..L {
        assert!(
            (ks[i] - n_ed[i]).abs() < 0.02,
            "site {i}: KS = {}, ED = {} - LDA error exceeded 2% budget",
            ks[i],
            n_ed[i]
        );
        let same_dir = (ks[i] - 1.0).signum() == (n_ed[i] - 1.0).signum();
        assert!(
            same_dir || (n_ed[i] - 1.0).abs() < 1e-8,
            "site {i}: KS and ED disagree on symmetry-breaking direction"
        );
    }
}

#[test]
fn l4_moderate_comb_ed_breaks_symmetry_as_expected() {
    // ED-only check (no KS): a sizable comb V = (+0.1, -0.1, +0.1, -0.1)
    // should give a direct electrostatic response: high-V sites depleted,
    // low-V sites enhanced. Mass conservation: sum n = 4.
    let v0 = 0.1;
    let v = [v0, -v0, v0, -v0];
    let result = ed::canonical_thermal(L, 2, 2, 1.0, 4.0, &v);
    let n_ed = ed::thermal_density(&result, 2.0);
    let total: f64 = n_ed.iter().sum();
    assert!((total - 4.0).abs() < 1e-10, "mass sum = {total}");
    assert!(
        n_ed[0] < 1.0 && n_ed[2] < 1.0,
        "high-V sites: {} {}",
        n_ed[0],
        n_ed[2]
    );
    assert!(
        n_ed[1] > 1.0 && n_ed[3] > 1.0,
        "low-V sites: {} {}",
        n_ed[1],
        n_ed[3]
    );
}
