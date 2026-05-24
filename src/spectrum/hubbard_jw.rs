#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! Matrix-free 1D inhomogeneous Hubbard Hamiltonian in the Jordan-Wigner
//! occupation-bitmask basis.
//!
//! Implements [`LinearOperator`] so the same Lanczos kernel that
//! consumes [`faer::Mat`] dense matrices can also diagonalise the
//! many-body Hubbard Hamiltonian without ever materialising the
//! `dim x dim` matrix. Hilbert dimension is `C(L, n_up) * C(L, n_dn)`;
//! at L = 8 half-filling that is 4900 -- still tractable dense, but
//! L = 10 (63504) and beyond is where matrix-free starts mattering.
//!
//! Basis convention matches [`crate::reference::ed`]:
//! `joint[r] = (up_mask, dn_mask)` with `r = up_idx * m_dn + dn_idx`.
//! JW sign uses [`crate::reference::ed::fermion_sign`].

use super::linear_operator::LinearOperator;
use crate::reference::ed::{enumerate_basis, fermion_sign};
use std::collections::HashMap;

/// Matrix-free real-symmetric Hubbard operator for use with Lanczos.
#[derive(Debug, Clone)]
pub struct JwHubbard {
    num_sites: usize,
    hopping_j: f64,
    basis_up: Vec<u32>,
    basis_dn: Vec<u32>,
    /// joint-row stride: `joint[up_idx * m_dn + dn_idx]`.
    m_dn: usize,
    lookup_up: HashMap<u32, usize>,
    lookup_dn: HashMap<u32, usize>,
    /// Pre-computed diagonal contribution per joint row:
    /// `U * doubles + sum_i v_ext[i] * (n_up + n_dn)_i`.
    diag: Vec<f64>,
}

impl JwHubbard {
    /// Number of lattice sites L.
    #[must_use]
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    /// Per-row joint Jordan-Wigner masks (up_mask, dn_mask). Row index
    /// matches the LinearOperator basis ordering: r = up_idx * m_dn + dn_idx.
    #[must_use]
    pub fn joint_masks(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.dim());
        for &up in &self.basis_up {
            for &dn in &self.basis_dn {
                out.push((up, dn));
            }
        }
        out
    }

    /// Build the operator for a given particle-number sector. `v_ext.len()`
    /// must equal `num_sites`.
    #[must_use]
    pub fn new(
        num_sites: usize,
        n_up: usize,
        n_dn: usize,
        hopping_j: f64,
        on_site_u: f64,
        v_ext: &[f64],
    ) -> Self {
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
        let lookup_up: HashMap<u32, usize> =
            basis_up.iter().enumerate().map(|(i, &m)| (m, i)).collect();
        let lookup_dn: HashMap<u32, usize> =
            basis_dn.iter().enumerate().map(|(i, &m)| (m, i)).collect();

        let mut diag = Vec::with_capacity(m_up * m_dn);
        for &up_mask in &basis_up {
            for &dn_mask in &basis_dn {
                let doubles = f64::from((up_mask & dn_mask).count_ones());
                let mut d = on_site_u * doubles;
                for (i, &v) in v_ext.iter().enumerate() {
                    let occ = f64::from(((up_mask >> i) & 1) + ((dn_mask >> i) & 1));
                    d += v * occ;
                }
                diag.push(d);
            }
        }

        Self {
            num_sites,
            hopping_j,
            basis_up,
            basis_dn,
            m_dn,
            lookup_up,
            lookup_dn,
            diag,
        }
    }

    /// Per-spin nearest-neighbour hop contribution to `y`. Iterates each
    /// basis row, finds allowed hops on the active spin's mask, looks up
    /// the destination index, and writes `M[r', r] * x[r]` into `y[r']`.
    /// The Hermitian conjugate `M[r, r'] * x[r']` is captured by the
    /// symmetric condition on the opposite hop direction.
    fn add_hop_contribution(&self, x: &[f64], y: &mut [f64], spin_up: bool) {
        let m_dn = self.m_dn;
        let (active_basis, active_lookup) = if spin_up {
            (&self.basis_up, &self.lookup_up)
        } else {
            (&self.basis_dn, &self.lookup_dn)
        };
        let m_active = active_basis.len();
        let other_size = if spin_up { m_dn } else { self.basis_up.len() };
        for active_idx in 0..m_active {
            let mask = active_basis[active_idx];
            for bond in 0..self.num_sites.saturating_sub(1) {
                // forward hop: electron at `bond`, hole at `bond + 1`
                if (mask >> bond) & 1 == 1 && (mask >> (bond + 1)) & 1 == 0 {
                    let after = mask & !(1_u32 << bond);
                    let s1 = fermion_sign(mask, bond);
                    let new_mask = after | (1_u32 << (bond + 1));
                    let s2 = fermion_sign(after, bond + 1);
                    let new_active_idx = active_lookup[&new_mask];
                    let coupling = -self.hopping_j * s1 * s2;
                    for spectator in 0..other_size {
                        let (r, r_new) = if spin_up {
                            (
                                active_idx * m_dn + spectator,
                                new_active_idx * m_dn + spectator,
                            )
                        } else {
                            (
                                spectator * m_dn + active_idx,
                                spectator * m_dn + new_active_idx,
                            )
                        };
                        y[r_new] += coupling * x[r];
                    }
                }
                // backward hop: electron at `bond + 1`, hole at `bond`
                if (mask >> (bond + 1)) & 1 == 1 && (mask >> bond) & 1 == 0 {
                    let after = mask & !(1_u32 << (bond + 1));
                    let s1 = fermion_sign(mask, bond + 1);
                    let new_mask = after | (1_u32 << bond);
                    let s2 = fermion_sign(after, bond);
                    let new_active_idx = active_lookup[&new_mask];
                    let coupling = -self.hopping_j * s1 * s2;
                    for spectator in 0..other_size {
                        let (r, r_new) = if spin_up {
                            (
                                active_idx * m_dn + spectator,
                                new_active_idx * m_dn + spectator,
                            )
                        } else {
                            (
                                spectator * m_dn + active_idx,
                                spectator * m_dn + new_active_idx,
                            )
                        };
                        y[r_new] += coupling * x[r];
                    }
                }
            }
        }
    }
}

impl LinearOperator for JwHubbard {
    fn dim(&self) -> usize {
        self.basis_up.len() * self.basis_dn.len()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.dim();
        assert_eq!(x.len(), n, "x.len() = {} != dim {n}", x.len());
        assert_eq!(y.len(), n, "y.len() = {} != dim {n}", y.len());
        for (yi, (&d, &xi)) in y.iter_mut().zip(self.diag.iter().zip(x.iter())) {
            *yi = d * xi;
        }
        self.add_hop_contribution(x, y, true);
        self.add_hop_contribution(x, y, false);
    }
}

#[cfg(test)]
mod tests {
    use super::super::lanczos::{diagonalize, LanczosParams};
    use super::*;
    use crate::reference::ed;

    fn assert_matvec_matches_ed(num_sites: usize, n_up: usize, n_dn: usize, u: f64, v: &[f64]) {
        let jw = JwHubbard::new(num_sites, n_up, n_dn, 1.0, u, v);
        let ed = ed::canonical_thermal(num_sites, n_up, n_dn, 1.0, u, v);
        let dim = jw.dim();
        // We don't expose ED's raw H matrix, so reconstruct
        // `H * e_col = sum_alpha lambda_alpha * U[col, alpha] * U[:, alpha]`
        // from the eigen-decomposition and compare to jw.apply(e_col).
        for col in 0..dim.min(8) {
            let mut x = vec![0.0_f64; dim];
            x[col] = 1.0;
            let mut y_jw = vec![0.0_f64; dim];
            jw.apply(&x, &mut y_jw);
            let mut y_ed = vec![0.0_f64; dim];
            for row in 0..dim {
                let mut acc = 0.0_f64;
                for alpha in 0..dim {
                    acc += ed.eigenvalues[alpha]
                        * ed.eigenvectors[(col, alpha)]
                        * ed.eigenvectors[(row, alpha)];
                }
                y_ed[row] = acc;
            }
            for row in 0..dim {
                assert!(
                    (y_jw[row] - y_ed[row]).abs() < 1e-10,
                    "matvec mismatch row {row} col {col}: jw = {}, ed = {}",
                    y_jw[row],
                    y_ed[row]
                );
            }
        }
    }

    #[test]
    fn jw_matvec_dimer_uniform_matches_ed() {
        assert_matvec_matches_ed(2, 1, 1, 4.0, &[0.0, 0.0]);
    }

    #[test]
    fn jw_matvec_l4_inhomogeneous_matches_ed() {
        assert_matvec_matches_ed(4, 2, 2, 2.5, &[0.1, -0.2, 0.3, -0.1]);
    }

    #[test]
    fn jw_matvec_l4_spin_polarised_matches_ed() {
        assert_matvec_matches_ed(4, 2, 1, 1.0, &[0.0; 4]);
    }

    #[test]
    fn jw_lanczos_ground_state_matches_ed_at_l4() {
        let v = [0.05_f64, -0.05, 0.05, -0.05];
        let jw = JwHubbard::new(4, 2, 2, 1.0, 4.0, &v);
        let ed = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v);
        let params = LanczosParams {
            krylov_dim: Some(20),
            max_iter: 200,
            tol: 1e-14,
        };
        let eig = diagonalize(&jw, &params).unwrap();
        let mut got = eig.eigenvalues;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (got[0] - ed.eigenvalues[0]).abs() < 1e-9,
            "GS: jw lanczos = {}, ed = {}",
            got[0],
            ed.eigenvalues[0]
        );
    }

    #[test]
    fn jw_lanczos_low_spectrum_matches_ed_at_l6_half_filling() {
        let v = [0.0_f64; 6];
        let jw = JwHubbard::new(6, 3, 3, 1.0, 4.0, &v);
        let ed = ed::canonical_thermal(6, 3, 3, 1.0, 4.0, &v);
        assert_eq!(jw.dim(), 400);
        let params = LanczosParams {
            krylov_dim: Some(60),
            max_iter: 800,
            tol: 1e-14,
        };
        let eig = diagonalize(&jw, &params).unwrap();
        let mut got = eig.eigenvalues;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..3 {
            assert!(
                (got[k] - ed.eigenvalues[k]).abs() < 1e-7,
                "level {k}: jw = {}, ed = {}",
                got[k],
                ed.eigenvalues[k]
            );
        }
    }
}
