#![allow(clippy::too_long_first_doc_paragraph)]
#![allow(clippy::too_many_lines)]
#![allow(
    unknown_lints,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::manual_is_multiple_of,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
//! Generic many-body exact diagonalisation of the 1D inhomogeneous
//! Hubbard model on an open chain.
//!
//! Hilbert space dimension is `C(L, n_up) * C(L, n_dn)`. Practical
//! workstation limit is roughly `L <= 6` for full spectrum.
//!
//! Bitmask basis with the convention `|n_0 n_1 ... n_{L-1}>
//! = (c+_0)^{n_0} ... (c+_{L-1})^{n_{L-1}} |0>`.

use faer::{Mat, Side};
use std::collections::HashMap;

/// Result of a full-spectrum diagonalisation.
#[derive(Debug, Clone)]
pub struct EdResult {
    /// Number of sites.
    pub num_sites: usize,
    /// Number of spin-up electrons.
    pub n_up: usize,
    /// Number of spin-down electrons.
    pub n_dn: usize,
    /// All eigenvalues, ascending.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector columns matching `eigenvalues` order.
    pub eigenvectors: Mat<f64>,
    /// `joint[r] = (up_mask, dn_mask)` for composite basis row `r`.
    /// Row index = `up_idx * m_dn + dn_idx`.
    pub joint: Vec<(u32, u32)>,
}

/// Enumerate the `C(L, n)` bitmasks with exactly `n` bits set.
#[must_use]
pub fn enumerate_basis(l: usize, n_electrons: usize) -> Vec<u32> {
    assert!(
        l <= u32::BITS as usize,
        "enumerate_basis: l = {l} exceeds 32-bit mask capacity"
    );
    (0_u32..(1_u32 << l))
        .filter(|m| m.count_ones() as usize == n_electrons)
        .collect()
}

/// Jordan-Wigner sign for inserting/removing at `site` on state `mask`.
/// Counts electrons at sites `0..site` in `mask`; returns -1 if odd,
/// +1 otherwise. Caller must pass the state in which the operator
/// actually acts (for creation after removal, pass the post-removal
/// mask, not the original).
#[must_use]
pub fn fermion_sign(mask: u32, site: usize) -> f64 {
    let lower = mask & ((1_u32 << site) - 1);
    if lower.count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Single-spin hopping matrix on an `l`-site open chain in the
/// occupation-bitmask basis. The returned matrix includes the leading
/// `-hopping_j` (i.e., M = -t * c+_j c_i with JW signs); callers do
/// NOT add another minus sign.
#[must_use]
pub fn single_spin_hopping(basis: &[u32], l: usize, hopping_j: f64) -> Mat<f64> {
    let dim = basis.len();
    let mut h = Mat::<f64>::zeros(dim, dim);
    let lookup: HashMap<u32, usize> = basis.iter().enumerate().map(|(i, &m)| (m, i)).collect();
    for (col, &mask) in basis.iter().enumerate() {
        for bond in 0..l.saturating_sub(1) {
            if (mask >> bond) & 1 == 1 && (mask >> (bond + 1)) & 1 == 0 {
                let after = mask & !(1_u32 << bond);
                let s1 = fermion_sign(mask, bond);
                let new_mask = after | (1_u32 << (bond + 1));
                let s2 = fermion_sign(after, bond + 1);
                let row = lookup[&new_mask];
                h[(row, col)] -= hopping_j * s1 * s2;
            }
            if (mask >> (bond + 1)) & 1 == 1 && (mask >> bond) & 1 == 0 {
                let after = mask & !(1_u32 << (bond + 1));
                let s1 = fermion_sign(mask, bond + 1);
                let new_mask = after | (1_u32 << bond);
                let s2 = fermion_sign(after, bond);
                let row = lookup[&new_mask];
                h[(row, col)] -= hopping_j * s1 * s2;
            }
        }
    }
    // Hermiticity is essential because callers diagonalise with
    // Side::Lower; an asymmetric matrix would silently produce wrong
    // eigenvectors.
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

/// Build the full Hubbard Hamiltonian, diagonalise, and package the
/// result. `v_ext.len()` must equal `num_sites`.
#[must_use]
pub fn canonical_thermal(
    num_sites: usize,
    n_up: usize,
    n_dn: usize,
    hopping_j: f64,
    on_site_u: f64,
    v_ext: &[f64],
) -> EdResult {
    assert_eq!(
        v_ext.len(),
        num_sites,
        "v_ext length {} must equal num_sites {}",
        v_ext.len(),
        num_sites
    );
    let basis_up = enumerate_basis(num_sites, n_up);
    let basis_dn = if n_up == n_dn {
        basis_up.clone()
    } else {
        enumerate_basis(num_sites, n_dn)
    };
    let m_up = basis_up.len();
    let m_dn = basis_dn.len();
    let dim = m_up * m_dn;
    let mut h = Mat::<f64>::zeros(dim, dim);

    let h_hop_up = single_spin_hopping(&basis_up, num_sites, hopping_j);
    let h_hop_dn = if n_up == n_dn {
        h_hop_up.clone()
    } else {
        single_spin_hopping(&basis_dn, num_sites, hopping_j)
    };

    for a in 0..m_up {
        for b in 0..m_up {
            let t = h_hop_up[(a, b)];
            if t == 0.0 {
                continue;
            }
            for dn in 0..m_dn {
                h[(a * m_dn + dn, b * m_dn + dn)] += t;
            }
        }
    }
    for a in 0..m_dn {
        for b in 0..m_dn {
            let t = h_hop_dn[(a, b)];
            if t == 0.0 {
                continue;
            }
            for up in 0..m_up {
                h[(up * m_dn + a, up * m_dn + b)] += t;
            }
        }
    }

    let mut joint = Vec::with_capacity(dim);
    for up_idx in 0..m_up {
        for dn_idx in 0..m_dn {
            let up_mask = basis_up[up_idx];
            let dn_mask = basis_dn[dn_idx];
            joint.push((up_mask, dn_mask));
            let doubles = f64::from((up_mask & dn_mask).count_ones());
            h[(up_idx * m_dn + dn_idx, up_idx * m_dn + dn_idx)] += on_site_u * doubles;
        }
    }
    for up_idx in 0..m_up {
        for dn_idx in 0..m_dn {
            let up_mask = basis_up[up_idx];
            let dn_mask = basis_dn[dn_idx];
            let mut diag = 0.0_f64;
            for i in 0..num_sites {
                let occ = f64::from(((up_mask >> i) & 1) + ((dn_mask >> i) & 1));
                diag += v_ext[i] * occ;
            }
            h[(up_idx * m_dn + dn_idx, up_idx * m_dn + dn_idx)] += diag;
        }
    }

    let eigen = h
        .self_adjoint_eigen(Side::Lower)
        .expect("self-adjoint EVD failed");
    let s_col = eigen.S().column_vector();
    let u_mat = eigen.U();
    let mut indexed: Vec<(usize, f64)> = (0..dim).map(|k| (k, s_col[k])).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("non-NaN eigenvalues"));
    let mut eigenvalues = Vec::with_capacity(dim);
    let mut eigenvectors = Mat::<f64>::zeros(dim, dim);
    for (new_k, (orig_k, val)) in indexed.into_iter().enumerate() {
        eigenvalues.push(val);
        for i in 0..dim {
            eigenvectors[(i, new_k)] = u_mat[(i, orig_k)];
        }
    }

    EdResult {
        num_sites,
        n_up,
        n_dn,
        eigenvalues,
        eigenvectors,
        joint,
    }
}

/// Canonical thermal density per site (sum over spin) from an ED result.
#[must_use]
pub fn thermal_density(ed: &EdResult, beta: f64) -> Vec<f64> {
    assert!(
        !ed.eigenvalues.is_empty(),
        "thermal_density requires non-empty spectrum"
    );
    let dim = ed.eigenvalues.len();
    let shift = ed.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let mut z = 0.0_f64;
    let mut n_i = vec![0.0_f64; ed.num_sites];
    for k in 0..dim {
        let weight = (-beta * (ed.eigenvalues[k] - shift)).exp();
        z += weight;
        for (j, &(up_mask, dn_mask)) in ed.joint.iter().enumerate() {
            let psi = ed.eigenvectors[(j, k)];
            let amp_sq = psi * psi;
            for site in 0..ed.num_sites {
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

/// Canonical thermal partition function from the spectrum.
#[must_use]
pub fn partition_function(eigenvalues: &[f64], beta: f64) -> f64 {
    let shift = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let z_shifted: f64 = eigenvalues
        .iter()
        .map(|e| (-beta * (e - shift)).exp())
        .sum();
    z_shifted * (-beta * shift).exp()
}

/// Canonical free energy `F = -ln Z / beta`.
#[must_use]
pub fn free_energy(eigenvalues: &[f64], beta: f64) -> f64 {
    let shift = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let z_shifted: f64 = eigenvalues
        .iter()
        .map(|e| (-beta * (e - shift)).exp())
        .sum();
    shift - z_shifted.ln() / beta
}

/// Exact Palamara 2024 III.3 quantum correction `Theta_2 = sigma_W^2 -
/// sigma_W^2_diag` for a sudden quench `H_init -> H_final`. Off-diagonal
/// contribution to the work variance in `H_init`'s eigenbasis:
///
/// ```text
/// Theta_2_exact = (1/Z_init) Σ_n e^{-β E_init,n} ( <n_init|W^2|n_init> - W_nn^2 )
/// W_nn          = <n_init| (H_final - H_init) |n_init>
/// ```
///
/// Vanishes when `H_init` and `H_final` share an eigenbasis (trivial
/// quench, or quench preserving symmetries that diagonalise both).
/// Strictly positive otherwise.
///
/// `ed_init.eigenvalues.len()` and `ed_final.eigenvalues.len()` must be
/// equal; both `EdResult`s must share the same joint occupation basis
/// (caller's responsibility — only dimension is checked).
#[must_use]
pub fn exact_theta_2(ed_init: &EdResult, ed_final: &EdResult, beta: f64) -> f64 {
    assert!(
        !ed_init.eigenvalues.is_empty(),
        "exact_theta_2: ed_init spectrum empty"
    );
    assert_eq!(
        ed_init.eigenvalues.len(),
        ed_final.eigenvalues.len(),
        "exact_theta_2: ed_init and ed_final must share Hilbert dimension (got {} vs {})",
        ed_init.eigenvalues.len(),
        ed_final.eigenvalues.len()
    );
    let dim = ed_init.eigenvalues.len();
    let shift = ed_init
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let mut z = 0.0_f64;
    let mut sum_w_nn = 0.0_f64;
    let mut sum_w_nn_sq = 0.0_f64;
    let mut sum_full_w_sq_n = 0.0_f64;

    for n in 0..dim {
        let weight = (-beta * (ed_init.eigenvalues[n] - shift)).exp();
        z += weight;

        let mut psi_n = vec![0.0_f64; dim];
        for j in 0..dim {
            psi_n[j] = ed_init.eigenvectors[(j, n)];
        }

        let h_final_psi = apply_hamiltonian(&psi_n, ed_final);
        let h_init_psi: Vec<f64> = psi_n.iter().map(|p| ed_init.eigenvalues[n] * p).collect();
        let delta_psi: Vec<f64> = h_final_psi
            .iter()
            .zip(h_init_psi.iter())
            .map(|(a, b)| a - b)
            .collect();

        let w_nn: f64 = psi_n.iter().zip(delta_psi.iter()).map(|(p, d)| p * d).sum();
        let full_w_sq_n: f64 = delta_psi.iter().map(|d| d * d).sum();

        sum_w_nn += weight * w_nn;
        sum_w_nn_sq += weight * w_nn * w_nn;
        sum_full_w_sq_n += weight * full_w_sq_n;
    }

    assert!(
        z.is_finite() && z > 0.0,
        "exact_theta_2: partition function collapsed (Z = {z}, beta = {beta}); check eigenvalues / beta"
    );
    let mean_w = sum_w_nn / z;
    let mean_w_sq = sum_full_w_sq_n / z;
    let mean_w_nn_sq = sum_w_nn_sq / z;

    let sigma_w_sq = mean_w_sq - mean_w * mean_w;
    let sigma_w_sq_diag = mean_w_nn_sq - mean_w * mean_w;
    sigma_w_sq - sigma_w_sq_diag
}

/// Apply a real-symmetric `H` (encoded by its eigen-decomposition in
/// `ed`) to `psi` in the original basis: `H psi = U diag(E) U^T psi`.
/// Shared between `exact_theta_2` and `reference::tpq` matrix-element
/// estimators; no dense `H` is materialised.
pub(crate) fn apply_hamiltonian(psi: &[f64], ed: &EdResult) -> Vec<f64> {
    let dim = ed.eigenvalues.len();
    debug_assert_eq!(psi.len(), dim);
    let mut d = vec![0.0_f64; dim];
    for alpha in 0..dim {
        let mut acc = 0.0_f64;
        for (j, &p) in psi.iter().enumerate() {
            acc += ed.eigenvectors[(j, alpha)] * p;
        }
        d[alpha] = acc;
    }
    let mut out = vec![0.0_f64; dim];
    for alpha in 0..dim {
        let w = ed.eigenvalues[alpha] * d[alpha];
        for (j, o) in out.iter_mut().enumerate() {
            *o += w * ed.eigenvectors[(j, alpha)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_basis_l4_n2_yields_six_states_each_with_two_bits() {
        let b = enumerate_basis(4, 2);
        assert_eq!(b.len(), 6);
        for mask in &b {
            assert_eq!(mask.count_ones(), 2);
        }
    }

    #[test]
    fn fermion_sign_counts_lower_occupied_bits() {
        assert!((fermion_sign(0b0010, 1) - 1.0).abs() < 1e-14);
        assert!((fermion_sign(0b0011, 1) - (-1.0)).abs() < 1e-14);
        assert!((fermion_sign(0b0011, 2) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn single_spin_hopping_l2_matches_known_minus_t_form() {
        let basis = vec![0b01_u32, 0b10_u32];
        let h = single_spin_hopping(&basis, 2, 1.0);
        assert!(h[(0, 0)].abs() < 1e-14);
        assert!(h[(1, 1)].abs() < 1e-14);
        assert!((h[(0, 1)] - (-1.0)).abs() < 1e-14);
        assert!((h[(1, 0)] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn dimer_ed_matches_analytic_closed_form() {
        let ed = canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        assert_eq!(ed.eigenvalues.len(), 4);
        let expected = super::super::dimer::spectrum(1.0, 4.0);
        for (k, &e) in ed.eigenvalues.iter().enumerate() {
            assert!(
                (e - expected[k]).abs() < 1e-10,
                "eigenvalue {k}: ED = {e}, analytic = {}",
                expected[k]
            );
        }
    }

    #[test]
    fn dimer_ed_thermal_density_is_one_at_half_filling() {
        let ed = canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let n = thermal_density(&ed, 2.0);
        for (i, &ni) in n.iter().enumerate() {
            assert!((ni - 1.0).abs() < 1e-12, "site {i}: {ni}");
        }
    }

    #[test]
    fn dimer_ed_free_energy_matches_analytic_closed_form() {
        let ed = canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let f_ed = free_energy(&ed.eigenvalues, 2.0);
        let f_an = super::super::dimer::free_energy(2.0, 1.0, 4.0);
        assert!(
            (f_ed - f_an).abs() < 1e-10,
            "F_ED = {f_ed}, F_analytic = {f_an}"
        );
    }

    #[test]
    fn l4_uniform_v_density_is_one_everywhere() {
        let ed = canonical_thermal(4, 2, 2, 1.0, 4.0, &[0.0; 4]);
        assert_eq!(ed.eigenvalues.len(), 36);
        let n = thermal_density(&ed, 2.0);
        for (i, &ni) in n.iter().enumerate() {
            assert!((ni - 1.0).abs() < 1e-10, "site {i}: {ni}");
        }
    }

    #[test]
    fn l6_half_filling_hilbert_dim_is_400() {
        // L = 6, N_up = N_dn = 3: C(6, 3)^2 = 20^2 = 400 states.
        let ed = canonical_thermal(6, 3, 3, 1.0, 4.0, &[0.0; 6]);
        assert_eq!(ed.eigenvalues.len(), 400);
        assert_eq!(ed.joint.len(), 400);
    }

    #[test]
    fn l6_uniform_v_density_is_one_everywhere() {
        // Half-filling V = 0: translation + particle-hole symmetry forces n_i = 1.
        let ed = canonical_thermal(6, 3, 3, 1.0, 4.0, &[0.0; 6]);
        let n = thermal_density(&ed, 2.0);
        for (i, &ni) in n.iter().enumerate() {
            assert!((ni - 1.0).abs() < 1e-10, "L=6 site {i}: {ni}");
        }
    }

    #[test]
    fn l6_u_zero_free_energy_matches_free_chain_pratt() {
        // At U = 0 the canonical free energy from the many-body ED must
        // agree with the single-particle Pratt recursion on the OBC chain
        // spectrum (cross-method consistency at the U = 0 limit).
        let l = 6_usize;
        let n_per_spin = 3_usize;
        let t = 1.0_f64;
        let beta = 2.0_f64;
        let v_ext = vec![0.0_f64; l];

        let ed_result = canonical_thermal(l, n_per_spin, n_per_spin, t, 0.0, &v_ext);
        let f_ed = free_energy(&ed_result.eigenvalues, beta);

        let spec = super::super::free_chain::single_particle_spectrum_obc(l, t);
        let f_pratt = super::super::free_chain::free_energy(&spec, n_per_spin, beta);

        assert!(
            (f_ed - f_pratt).abs() < 1e-9,
            "ED F = {f_ed}, Pratt F = {f_pratt}"
        );
    }

    #[test]
    fn exact_theta_2_trivial_quench_is_zero() {
        let v = [0.1_f64, -0.2, 0.3, -0.1];
        let ed_init = canonical_thermal(4, 2, 2, 1.0, 4.0, &v);
        let theta = exact_theta_2(&ed_init, &ed_init, 2.0);
        assert!(
            theta.abs() < 1e-10,
            "trivial quench should give 0, got {theta}"
        );
    }

    #[test]
    fn exact_theta_2_small_quench_is_positive_at_l4() {
        let v_init = [0.0_f64; 4];
        let v_final = [0.3_f64, -0.3, 0.3, -0.3];
        let ed_init = canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);
        let theta = exact_theta_2(&ed_init, &ed_final, 2.0);
        assert!(
            theta > 0.0,
            "non-commuting quench should give theta > 0, got {theta}"
        );
        assert!(
            theta < 1.0,
            "theta should be modest for delta_v = 0.3 quench, got {theta}"
        );
    }

    #[test]
    #[allow(clippy::suspicious_operation_groupings)]
    fn exact_theta_2_differs_from_diagonal_part_at_l4() {
        // sanity: full sigma_W^2 = exact_theta_2 + sigma_W^2_diag, where
        // sigma_W^2_diag is the variance of W_nn (diagonal-only). Here we
        // just check theta_2 < full variance and > 0.
        let v_init = [0.0_f64; 4];
        let v_final = [0.3_f64, -0.3, 0.3, -0.3];
        let ed_init = canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);
        let theta = exact_theta_2(&ed_init, &ed_final, 2.0);
        // Recompute sigma_W^2 directly for cross-check.
        let dim = ed_init.eigenvalues.len();
        let shift = ed_init
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let beta = 2.0;
        let mut z = 0.0_f64;
        let mut sum_w_nn = 0.0_f64;
        let mut sum_full = 0.0_f64;
        for n in 0..dim {
            let weight = (-beta * (ed_init.eigenvalues[n] - shift)).exp();
            z += weight;
            let mut psi_n = vec![0.0_f64; dim];
            for j in 0..dim {
                psi_n[j] = ed_init.eigenvectors[(j, n)];
            }
            let h_final_psi = apply_hamiltonian(&psi_n, &ed_final);
            let h_init_psi: Vec<f64> = psi_n.iter().map(|p| ed_init.eigenvalues[n] * p).collect();
            let delta: Vec<f64> = h_final_psi
                .iter()
                .zip(h_init_psi.iter())
                .map(|(a, b)| a - b)
                .collect();
            sum_w_nn += weight
                * psi_n
                    .iter()
                    .zip(delta.iter())
                    .map(|(p, d)| p * d)
                    .sum::<f64>();
            sum_full += weight * delta.iter().map(|d| d * d).sum::<f64>();
        }
        let mean_w = sum_w_nn / z;
        let sigma_w_sq = sum_full / z - mean_w * mean_w;
        assert!(
            theta > 0.0 && sigma_w_sq - theta >= -1e-12,
            "Theta_2 should be in (0, sigma_W^2]: theta = {theta}, sigma_W^2 = {sigma_w_sq}"
        );
    }

    #[test]
    fn exact_theta_2_zero_u_zero_quench_is_zero() {
        // At U=0 only kinetic + 1-body potential; W = delta_V is the
        // single-particle density operator. It still has off-diagonal
        // matrix elements in H_init's eigenbasis, so Theta_2 != 0 even
        // at U=0. But for the strict NO-quench case it must be 0.
        let v = [0.1_f64, 0.2, -0.1, 0.0];
        let ed_init = canonical_thermal(4, 2, 2, 1.0, 0.0, &v);
        let theta = exact_theta_2(&ed_init, &ed_init, 2.0);
        assert!(
            theta.abs() < 1e-10,
            "U=0 trivial quench should give 0, got {theta}"
        );
    }

    #[test]
    fn exact_theta_2_remains_finite_and_nonneg_across_beta_range() {
        // Theta_2 is a Hilbert-space invariant of W = H_final - H_init,
        // weighted thermally. It is non-negative for any beta but is not
        // monotone in beta (concentration on |0_init> at low T can leave
        // non-zero off-diagonal coupling via W). We just assert finite,
        // non-negative outputs across hot/warm/cold regimes.
        let v_init = [0.0_f64; 4];
        let v_final = [0.3_f64, -0.3, 0.3, -0.3];
        let ed_init = canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);
        for &beta in &[0.05_f64, 2.0, 20.0] {
            let theta = exact_theta_2(&ed_init, &ed_final, beta);
            assert!(
                theta.is_finite(),
                "beta={beta}: Theta_2 not finite, got {theta}"
            );
            assert!(theta >= 0.0, "beta={beta}: Theta_2 negative, got {theta}");
        }
    }

    #[test]
    fn exact_theta_2_l6_finite_and_positive_smoke() {
        // Smoke test at L=6 (dim = 400): exact_theta_2 must complete,
        // return a finite positive value for a non-commuting quench.
        let v_init = [0.0_f64; 6];
        let v_final = [0.2_f64, -0.2, 0.2, -0.2, 0.2, -0.2];
        let ed_init = canonical_thermal(6, 3, 3, 1.0, 4.0, &v_init);
        let ed_final = canonical_thermal(6, 3, 3, 1.0, 4.0, &v_final);
        let theta = exact_theta_2(&ed_init, &ed_final, 2.0);
        assert!(theta.is_finite(), "L=6 Theta_2 must be finite, got {theta}");
        assert!(
            theta > 0.0,
            "L=6 non-commuting quench Theta_2 should be > 0, got {theta}"
        );
    }
}
