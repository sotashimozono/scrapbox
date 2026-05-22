//! Exact-diagonalization validation for the Hubbard dimer.
//!
//! Compares scrapbox's KS-DFT density against the **true many-body**
//! density obtained by diagonalizing the 4x4 Hubbard Hamiltonian in the
//! `(N_↑=1, N_↓=1)` half-filling sector. The KS auxiliary system has its
//! own `Z, F` that differ from the interacting `Z, F`; only the
//! **density** is expected to match exactly (by the Hohenberg-Kohn /
//! Mermin bijection).
//!
//! Basis (4 states):
//!
//! ```text
//! |a> = c_{0u}^dag c_{1d}^dag |0>     // up at site 0, down at site 1
//! |b> = c_{1u}^dag c_{0d}^dag |0>     // up at site 1, down at site 0
//! |c> = c_{0u}^dag c_{0d}^dag |0>     // both at site 0
//! |d> = c_{1u}^dag c_{1d}^dag |0>     // both at site 1
//! ```
//!
//! Hamiltonian elements (`H = -J Sum_sigma (cdag_{0s} c_{1s} + h.c.) + U Sum_i n_{is_up} n_{is_dn}`):
//!
//! ```text
//! <a|H|c> = <a|H|d> = <b|H|c> = <b|H|d> = -J
//! <c|H|c> = <d|H|d> = U
//! (all other off-diagonals = 0)
//! ```

use faer::{Mat, Side};
use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::xc::ExchangeCorrelation;

fn dimer_hamiltonian(hopping_j: f64, on_site_u: f64) -> Mat<f64> {
    let mut h = Mat::<f64>::zeros(4, 4);
    h[(2, 2)] = on_site_u;
    h[(3, 3)] = on_site_u;
    let edges = [(0, 2), (0, 3), (1, 2), (1, 3)];
    for &(i, k) in &edges {
        h[(i, k)] = -hopping_j;
        h[(k, i)] = -hopping_j;
    }
    h
}

fn ed_thermal_density(eigenvalues: &[f64], eigenvectors: &Mat<f64>, beta: f64) -> [f64; 2] {
    let n_diag_0 = [1.0_f64, 1.0, 2.0, 0.0];
    let n_diag_1 = [1.0_f64, 1.0, 0.0, 2.0];

    let mut z = 0.0_f64;
    let mut n0 = 0.0_f64;
    let mut n1 = 0.0_f64;
    for (k, &eps_k) in eigenvalues.iter().enumerate() {
        let weight = (-beta * eps_k).exp();
        z += weight;
        let mut n0_k = 0.0;
        let mut n1_k = 0.0;
        for j in 0..4 {
            let psi_kj = eigenvectors[(j, k)];
            n0_k += psi_kj * psi_kj * n_diag_0[j];
            n1_k += psi_kj * psi_kj * n_diag_1[j];
        }
        n0 += weight * n0_k;
        n1 += weight * n1_k;
    }
    [n0 / z, n1 / z]
}

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

    let h = dimer_hamiltonian(j, u);
    let eigen = h
        .self_adjoint_eigen(Side::Lower)
        .expect("self-adjoint EVD failed");

    let s_col = eigen.S().column_vector();
    let u_mat = eigen.U();
    let mut eigvals = [0.0_f64; 4];
    let mut eigvecs = Mat::<f64>::zeros(4, 4);
    for k in 0..4 {
        eigvals[k] = s_col[k];
        for i in 0..4 {
            eigvecs[(i, k)] = u_mat[(i, k)];
        }
    }

    // Ground state in the singlet block: E = (U - sqrt(U^2+16J^2))/2.
    let e_singlet_lower = u.midpoint(-(u.mul_add(u, 16.0 * j * j)).sqrt());
    let min_eig = eigvals.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        (min_eig - e_singlet_lower).abs() < 1e-10,
        "ED ground-state energy off: got {min_eig}, expected {e_singlet_lower}"
    );

    let ed_density = ed_thermal_density(&eigvals, &eigvecs, beta);
    // Half-filling, V=0 -> by symmetry n_0 = n_1 = 1.
    assert!((ed_density[0] - 1.0).abs() < 1e-10);
    assert!((ed_density[1] - 1.0).abs() < 1e-10);

    let ks_density = run_ks_dimer();
    for (site, &ed) in ed_density.iter().enumerate() {
        let ks = ks_density[site];
        assert!(
            (ks - ed).abs() < 1e-10,
            "site {site}: KS density {ks} differs from ED density {ed}"
        );
    }

    let z_ed: f64 = eigvals.iter().map(|e| (-beta * e).exp()).sum();
    let f_ed = -z_ed.ln() / beta;
    eprintln!("ED dimer (U={u}, J={j}, beta={beta}): F_ED = {f_ed:.6}, density = {ed_density:?}");
}

#[test]
fn ed_spectrum_matches_analytic_formula() {
    let j = 1.0_f64;
    let u = 4.0_f64;
    let h = dimer_hamiltonian(j, u);
    let eigen = h
        .self_adjoint_eigen(Side::Lower)
        .expect("self-adjoint EVD failed");
    let s_col = eigen.S().column_vector();
    let mut eigvals: Vec<f64> = (0..4).map(|k| s_col[k]).collect();
    eigvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let e_singlet_lower = u.midpoint(-(u.mul_add(u, 16.0 * j * j)).sqrt());
    let e_singlet_upper = u.midpoint((u.mul_add(u, 16.0 * j * j)).sqrt());
    assert!((eigvals[0] - e_singlet_lower).abs() < 1e-10);
    assert!(eigvals[1].abs() < 1e-10, "triplet S_z=0 energy = 0"); // triplet S_z=0
    assert!((eigvals[2] - u).abs() < 1e-10); // anti-symmetric double-occupancy
    assert!((eigvals[3] - e_singlet_upper).abs() < 1e-10);
}
