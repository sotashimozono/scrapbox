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
}
