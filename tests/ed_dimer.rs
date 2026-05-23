//! Exact-diagonalization validation for the Hubbard dimer.
//!
//! Cross-checks scrapbox's KS-DFT density against the true many-body
//! density obtained via the `reference::dimer` analytic closed form and
//! the `reference::ed` generic many-body diagonaliser. The KS auxiliary
//! system carries its own `Z, F` that differ from the interacting `Z, F`;
//! only the density is expected to match exactly (by the Hohenberg-Kohn /
//! Mermin bijection).
//!
//! This file used to carry its own hand-built 4x4 Hubbard Hamiltonian
//! and Boltzmann sum; both have been consolidated into
//! `scrapbox::reference::{dimer, ed}` so that future tests at L>2 can
//! reuse the same machinery.

use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::reference::{dimer, ed};
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::xc::ExchangeCorrelation;

fn run_ks_dimer() -> Vec<f64> {
    let cfg_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("configs")
        .join("dimer_validate.toml");
    let cfg = Config::from_file(&cfg_path).expect("config");
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
fn ks_density_matches_ed_at_half_filling() {
    let j = 1.0_f64;
    let u = 4.0_f64;
    let beta = 2.0_f64;

    // ED via the generic builder.
    let result = ed::canonical_thermal(2, 1, 1, j, u, &[0.0, 0.0]);
    let ed_density = ed::thermal_density(&result, beta);

    // Half-filling, V=0 -> n_0 = n_1 = 1 by symmetry.
    assert!((ed_density[0] - 1.0).abs() < 1e-10);
    assert!((ed_density[1] - 1.0).abs() < 1e-10);

    // KS-DFT density should match ED density site-by-site.
    let ks_density = run_ks_dimer();
    for (site, &n_ed) in ed_density.iter().enumerate() {
        let n_ks = ks_density[site];
        assert!(
            (n_ks - n_ed).abs() < 1e-10,
            "site {site}: KS density {n_ks} differs from ED density {n_ed}"
        );
    }
}

#[test]
fn ed_spectrum_matches_dimer_analytic_closed_form() {
    let j = 1.0_f64;
    let u = 4.0_f64;
    let result = ed::canonical_thermal(2, 1, 1, j, u, &[0.0, 0.0]);
    let expected = dimer::spectrum(j, u);
    assert_eq!(result.eigenvalues.len(), 4);
    for (k, &e) in result.eigenvalues.iter().enumerate() {
        assert!(
            (e - expected[k]).abs() < 1e-10,
            "eigenvalue {k}: ED = {e}, analytic = {}",
            expected[k]
        );
    }
}

#[test]
fn dimer_analytic_and_generic_ed_agree_on_free_energy() {
    let j = 1.0_f64;
    let u = 4.0_f64;
    let beta = 2.0_f64;
    let result = ed::canonical_thermal(2, 1, 1, j, u, &[0.0, 0.0]);
    let f_ed = ed::free_energy(&result.eigenvalues, beta);
    let f_an = dimer::free_energy(beta, j, u);
    assert!(
        (f_ed - f_an).abs() < 1e-12,
        "F_ED = {f_ed} vs F_analytic = {f_an}"
    );
}
