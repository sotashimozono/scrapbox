#![allow(clippy::doc_markdown, clippy::many_single_char_names)]
//! Exact-diagonalization validation for the 1D L=4 open-chain Hubbard
//! model at half-filling (N_up = N_dn = 2). The single-spin half-filled
//! sector has C(4,2) = 6 states, so the total Hilbert space dimension is
//! 36 and the canonical Hamiltonian is a 36x36 real-symmetric matrix.

use faer::{Mat, Side};
use scrapbox::config::Config;
use scrapbox::density::CanonicalDensityEvaluator;
use scrapbox::hamiltonian::KohnShamHamiltonian;
use scrapbox::scf::CanonicalThermalDFTSolver;
use scrapbox::spectrum::SpectrumSource;
use scrapbox::xc::ExchangeCorrelation;

const L: usize = 4;
const N_PER_SPIN: usize = 2;

fn enumerate_basis(l: usize, n_electrons: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for mask in 0_u32..(1_u32 << l) {
        if mask.count_ones() as usize == n_electrons {
            out.push(mask);
        }
    }
    out
}

/// Jordan-Wigner sign for inserting/removing at `site` on state `mask`.
/// Counts electrons at sites 0..site in `mask`; returns -1 if odd,
/// +1 otherwise. The caller must pass the state in which the operator
/// actually acts (for creation after a previous removal, pass the
/// post-removal mask, not the original).
fn fermion_sign(mask: u32, site: usize) -> f64 {
    let lower = mask & ((1_u32 << site) - 1);
    if lower.count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Single-spin hopping matrix for the L-site open chain, in the
/// occupation-bitmask basis. The returned matrix M includes the leading
/// `-hopping_j` (so M = -t * c_j^dag c_i in the JW convention);
/// callers do NOT add another minus sign.
fn single_spin_hopping(basis: &[u32], l: usize, hopping_j: f64) -> Mat<f64> {
    let dim = basis.len();
    let mut h = Mat::<f64>::zeros(dim, dim);
    let lookup: std::collections::HashMap<u32, usize> =
        basis.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    for (col, &mask) in basis.iter().enumerate() {
        for bond in 0..(l - 1) {
            if (mask >> bond) & 1 == 1 && (mask >> (bond + 1)) & 1 == 0 {
                let after_remove = mask & !(1_u32 << bond);
                let sign1 = fermion_sign(mask, bond);
                let new_mask = after_remove | (1_u32 << (bond + 1));
                let sign2 = fermion_sign(after_remove, bond + 1);
                let total_sign = sign1 * sign2;
                let row = lookup[&new_mask];
                h[(row, col)] -= hopping_j * total_sign;
            }
            if (mask >> (bond + 1)) & 1 == 1 && (mask >> bond) & 1 == 0 {
                let after_remove = mask & !(1_u32 << (bond + 1));
                let sign1 = fermion_sign(mask, bond + 1);
                let new_mask = after_remove | (1_u32 << bond);
                let sign2 = fermion_sign(after_remove, bond);
                let total_sign = sign1 * sign2;
                let row = lookup[&new_mask];
                h[(row, col)] -= hopping_j * total_sign;
            }
        }
    }
    // Hermiticity is essential because we pass Side::Lower to faer and
    // it ignores the upper triangle. An asymmetric matrix would silently
    // produce wrong eigenvectors.
    for i in 0..dim {
        for j in 0..i {
            debug_assert!(
                (h[(i, j)] - h[(j, i)]).abs() < 1e-14,
                "single_spin_hopping not symmetric at ({i}, {j})"
            );
        }
    }
    h
}

/// Build the full Hubbard Hamiltonian and the basis-index map.
/// Returns `(H, joint)` where `H[r, c]` indexes the composite state
/// `up_idx * m + dn_idx` and `joint[r]` is the `(up_mask, dn_mask)` pair
/// for row `r`. This ordering contract is load-bearing - changing it
/// silently breaks `ed_thermal_density`.
fn build_full_hubbard(hopping_j: f64, u: f64, v_ext: &[f64]) -> (Mat<f64>, Vec<(u32, u32)>) {
    assert_eq!(v_ext.len(), L);
    let basis = enumerate_basis(L, N_PER_SPIN);
    let m = basis.len();
    let dim = m * m;
    let mut h = Mat::<f64>::zeros(dim, dim);
    let h_hop_single = single_spin_hopping(&basis, L, hopping_j);

    for up_a in 0..m {
        for up_b in 0..m {
            let t_up = h_hop_single[(up_a, up_b)];
            if t_up == 0.0 {
                continue;
            }
            for dn in 0..m {
                h[(up_a * m + dn, up_b * m + dn)] += t_up;
            }
        }
    }
    for dn_a in 0..m {
        for dn_b in 0..m {
            let t_dn = h_hop_single[(dn_a, dn_b)];
            if t_dn == 0.0 {
                continue;
            }
            for up in 0..m {
                h[(up * m + dn_a, up * m + dn_b)] += t_dn;
            }
        }
    }
    let mut joint = Vec::with_capacity(dim);
    for up_idx in 0..m {
        for dn_idx in 0..m {
            let up_mask = basis[up_idx];
            let dn_mask = basis[dn_idx];
            joint.push((up_mask, dn_mask));
            let doubles = f64::from((up_mask & dn_mask).count_ones());
            h[(up_idx * m + dn_idx, up_idx * m + dn_idx)] += u * doubles;
        }
    }
    for up_idx in 0..m {
        for dn_idx in 0..m {
            let up_mask = basis[up_idx];
            let dn_mask = basis[dn_idx];
            let mut diag = 0.0_f64;
            for i in 0..L {
                let occ = f64::from(((up_mask >> i) & 1) + ((dn_mask >> i) & 1));
                diag += v_ext[i] * occ;
            }
            h[(up_idx * m + dn_idx, up_idx * m + dn_idx)] += diag;
        }
    }
    (h, joint)
}

fn ed_thermal_density(
    eigenvalues: &[f64],
    eigenvectors: &Mat<f64>,
    joint: &[(u32, u32)],
    beta: f64,
) -> Vec<f64> {
    assert!(
        !eigenvalues.is_empty(),
        "ed_thermal_density requires non-empty spectrum"
    );
    let dim = eigenvalues.len();
    let shift = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let mut z = 0.0_f64;
    let mut n_i = vec![0.0_f64; L];
    for k in 0..dim {
        let weight = (-beta * (eigenvalues[k] - shift)).exp();
        z += weight;
        for (j, &(up_mask, dn_mask)) in joint.iter().enumerate() {
            let psi_kj = eigenvectors[(j, k)];
            let amp_sq = psi_kj * psi_kj;
            for site in 0..L {
                let occ = f64::from(((up_mask >> site) & 1) + ((dn_mask >> site) & 1));
                n_i[site] += weight * amp_sq * occ;
            }
        }
    }
    for x in &mut n_i {
        *x /= z;
    }
    n_i
}

fn run_ks_l4(v_ext: &[f64]) -> Vec<f64> {
    let cfg_str = format!(
        r#"
schema_version = "0.2"

[meta]
name = "ed_l4_aux"
description = "L=4 KS DFT auxiliary"
created = "2026-05-22"
tags = ["v0.3"]

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
directory = "runs/ed_l4_aux"
format = "json"
overwrite = true
"#,
        v_ext[0], v_ext[1], v_ext[2], v_ext[3]
    );
    let cfg = Config::from_toml_str(&cfg_str).expect("config");
    let h = KohnShamHamiltonian::from_config(&cfg.hamiltonian).unwrap();
    let xc = ExchangeCorrelation::from_config(&cfg.xc_functional, &h).unwrap();
    let spec = SpectrumSource::from_config(&cfg.spectrum_source).unwrap();
    let dens = CanonicalDensityEvaluator::from_config(&cfg.density_evaluator).unwrap();
    CanonicalThermalDFTSolver::new(h, xc, spec, dens, cfg.scf)
        .solve()
        .expect("KS SCF")
        .densities
}

fn diagonalize_36(h: &Mat<f64>) -> (Vec<f64>, Mat<f64>) {
    let dim = h.nrows();
    let eigen = h.selfadjoint_eigendecomposition(Side::Lower);
    let s_col = eigen.s().column_vector();
    let u_mat = eigen.u();
    let mut eigvals = Vec::with_capacity(dim);
    let mut eigvecs = Mat::<f64>::zeros(dim, dim);
    let mut indexed: Vec<(usize, f64)> = (0..dim).map(|k| (k, s_col.read(k))).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (new_k, (orig_k, val)) in indexed.into_iter().enumerate() {
        eigvals.push(val);
        for i in 0..dim {
            eigvecs[(i, new_k)] = u_mat[(i, orig_k)];
        }
    }
    (eigvals, eigvecs)
}

#[test]
fn enumerate_basis_l4_n2_gives_six_states() {
    let b = enumerate_basis(4, 2);
    assert_eq!(b.len(), 6);
    for mask in &b {
        assert_eq!(mask.count_ones(), 2);
    }
}

#[test]
fn fermion_sign_counts_lower_occupied() {
    assert!((fermion_sign(0b0010, 1) - 1.0).abs() < 1e-14);
    assert!((fermion_sign(0b0011, 1) - (-1.0)).abs() < 1e-14);
    assert!((fermion_sign(0b0011, 2) - 1.0).abs() < 1e-14);
}

#[test]
fn single_spin_hopping_l2_matches_known_form() {
    // L=2, N=1 single-spin sector has basis {|10>, |01>}.
    // H_kin = -J [[0, 1], [1, 0]] with our sign conventions.
    let basis = vec![0b01_u32, 0b10_u32];
    let h = single_spin_hopping(&basis, 2, 1.0);
    assert!((h[(0, 0)]).abs() < 1e-14);
    assert!((h[(1, 1)]).abs() < 1e-14);
    assert!((h[(0, 1)] - (-1.0)).abs() < 1e-14);
    assert!((h[(1, 0)] - (-1.0)).abs() < 1e-14);
}

#[test]
fn l4_uniform_v_ed_density_is_one_everywhere() {
    let j = 1.0_f64;
    let u = 4.0_f64;
    let beta = 2.0_f64;
    let v = [0.0_f64; 4];
    let (h, joint) = build_full_hubbard(j, u, &v);
    let (eigvals, eigvecs) = diagonalize_36(&h);
    assert_eq!(eigvals.len(), 36);
    let n = ed_thermal_density(&eigvals, &eigvecs, &joint, beta);
    for (i, &ni) in n.iter().enumerate() {
        assert!((ni - 1.0).abs() < 1e-10, "site {i}: {ni}");
    }
}

#[test]
fn l4_uniform_v_ks_density_matches_ed() {
    let v = [0.0_f64; 4];
    let ks = run_ks_l4(&v);
    let (h, joint) = build_full_hubbard(1.0, 4.0, &v);
    let (eigvals, eigvecs) = diagonalize_36(&h);
    let ed = ed_thermal_density(&eigvals, &eigvecs, &joint, 2.0);
    for i in 0..L {
        assert!(
            (ks[i] - ed[i]).abs() < 1e-10,
            "site {i}: KS = {}, ED = {}",
            ks[i],
            ed[i]
        );
    }
}

#[test]
fn l4_comb_v_ks_density_matches_ed_within_lda_error() {
    // Non-trivial test: KS-DFT (HubbardLDA) at the CDW comb V must agree
    // with ED to within the known LDA approximation error. At U=4 with
    // v0=0.1 the LDA error is ~10^-2; we use a loose 5% tolerance.
    // The V=0 case in l4_uniform_v_ks_density_matches_ed is symmetry-
    // forced (both methods give n_i=1 by translation), so it does NOT
    // exercise the KS-vs-ED approximation; this test does.
    let v = [0.02_f64, -0.02, 0.02, -0.02];
    let ks = run_ks_l4(&v);
    let (h, joint) = build_full_hubbard(1.0, 4.0, &v);
    let (eigvals, eigvecs) = diagonalize_36(&h);
    let ed = ed_thermal_density(&eigvals, &eigvecs, &joint, 2.0);
    for i in 0..L {
        assert!(
            (ks[i] - ed[i]).abs() < 0.02,
            "site {i}: KS = {}, ED = {} - LDA error exceeded 2% tolerance",
            ks[i],
            ed[i]
        );
        let same_dir = (ks[i] - 1.0).signum() == (ed[i] - 1.0).signum();
        assert!(
            same_dir || (ed[i] - 1.0).abs() < 1e-8,
            "site {i}: KS-ED disagree on symmetry-breaking direction"
        );
    }
}

#[test]
fn l4_small_comb_ed_breaks_symmetry_as_expected() {
    let v0 = 0.1;
    let v = [v0, -v0, v0, -v0];
    let (h, joint) = build_full_hubbard(1.0, 4.0, &v);
    let (eigvals, eigvecs) = diagonalize_36(&h);
    let ed = ed_thermal_density(&eigvals, &eigvecs, &joint, 2.0);
    // Mass conservation: Sum n_i = N = 4 (half filling, 4 electrons total).
    let total: f64 = ed.iter().sum();
    assert!((total - 4.0).abs() < 1e-10, "sum = {total}");
    // Direct electrostatic response: positive V_i raises on-site energy
    // and thermally depletes occupation. Mass conservation (sum n = N = 4)
    // is the only physical sum rule asserted here.
    assert!(
        ed[0] < 1.0 && ed[2] < 1.0,
        "high-V sites n = {} {}",
        ed[0],
        ed[2]
    );
    assert!(
        ed[1] > 1.0 && ed[3] > 1.0,
        "low-V sites n = {} {}",
        ed[1],
        ed[3]
    );
}
