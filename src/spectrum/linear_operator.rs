#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! `LinearOperator` abstraction for Lanczos-style spectrum solvers.
//!
//! Lanczos only needs `y = H x` (matrix-vector product) plus the
//! dimension. Decoupling that contract from storage layout lets the
//! same iterative kernel consume:
//!
//! - **dense** `Mat<f64>` (the v0.2 path)
//! - **sparse** CSR-form Hubbard Hamiltonians
//! - any future custom representation (matrix-free, structured, ...)

use crate::reference::ed::{enumerate_basis, fermion_sign};
use faer::Mat;
use std::collections::HashMap;

/// Anything that can apply `y = H x` for real-symmetric `H`.
///
/// Implementations are consumed by Lanczos and must behave as a real
/// symmetric linear operator: `apply` is treated as left-multiplication
/// `y <- H x`, and the iterative kernel assumes `H = H^T`.
pub trait LinearOperator {
    /// Square matrix dimension.
    fn dim(&self) -> usize;

    /// Compute `y = self * x`. `x.len()` and `y.len()` must equal
    /// `self.dim()`.
    ///
    /// `y` is **fully overwritten**; prior contents are ignored, not
    /// accumulated into. Implementors must not assume `y` starts at
    /// zero, and callers must not rely on the operator adding into
    /// `y` (i.e. there is no `+=` semantics).
    fn apply(&self, x: &[f64], y: &mut [f64]);

    /// Compute `ys[i] = self * xs[i]` for every `i in 0..xs.len()`.
    ///
    /// Default impl delegates to [`Self::apply`] per right-hand side.
    /// Implementors with cache-friendly batch layouts (e.g. CSR
    /// matrices) may override to iterate over the operator's storage
    /// once and accumulate into all `ys` in the inner loop, which is
    /// often substantially faster than `N` independent `apply` calls.
    ///
    /// `xs.len()` must equal `ys.len()`, and every `xs[i].len()` and
    /// `ys[i].len()` must equal `self.dim()`.
    fn apply_batch(&self, xs: &[&[f64]], ys: &mut [Vec<f64>]) {
        assert_eq!(
            xs.len(),
            ys.len(),
            "apply_batch: xs.len() = {} != ys.len() = {}",
            xs.len(),
            ys.len()
        );
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            self.apply(x, y);
        }
    }
}

impl LinearOperator for Mat<f64> {
    fn dim(&self) -> usize {
        debug_assert_eq!(self.nrows(), self.ncols(), "Mat must be square");
        self.nrows()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let n = self.dim();
        assert_eq!(x.len(), n, "x.len() = {} != dim {n}", x.len());
        assert_eq!(y.len(), n, "y.len() = {} != dim {n}", y.len());
        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..n {
                acc = self[(i, j)].mul_add(x[j], acc);
            }
            y[i] = acc;
        }
    }
}

/// Compressed Sparse Row representation of a real-symmetric matrix.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    dim: usize,
    row_starts: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl SparseMatrix {
    /// Build a CSR matrix from `(row, col, value)` triples.
    ///
    /// Duplicate `(row, col)` entries are **summed**, not overwritten
    /// (so pass each coefficient only once if you want assignment
    /// semantics). Entries with `value == 0.0` are dropped to keep the
    /// CSR sparse.
    ///
    /// Symmetry is **not** checked. Callers consuming this through
    /// [`LinearOperator`] (notably Lanczos) must ensure the triples
    /// describe a symmetric matrix `A == A.T`; otherwise the resulting
    /// eigenpairs are wrong with no warning.
    #[must_use]
    pub fn from_triples(dim: usize, triples: &[(usize, usize, f64)]) -> Self {
        let mut row_buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); dim];
        for &(r, c, v) in triples {
            assert!(
                r < dim && c < dim,
                "triple ({r}, {c}) out of bounds for dim {dim}"
            );
            if v == 0.0 {
                continue;
            }
            row_buckets[r].push((c, v));
        }
        let mut row_starts = Vec::with_capacity(dim + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_starts.push(0);
        for bucket in &mut row_buckets {
            bucket.sort_by_key(|&(c, _)| c);
            let mut i = 0;
            while i < bucket.len() {
                let c = bucket[i].0;
                let mut acc = 0.0_f64;
                while i < bucket.len() && bucket[i].0 == c {
                    acc += bucket[i].1;
                    i += 1;
                }
                if acc != 0.0 {
                    col_indices.push(c);
                    values.push(acc);
                }
            }
            row_starts.push(col_indices.len());
        }
        Self {
            dim,
            row_starts,
            col_indices,
            values,
        }
    }

    /// Number of stored nonzero entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Build the CSR representation of the 1D inhomogeneous Hubbard
    /// Hamiltonian on the joint Jordan-Wigner occupation-bitmask basis,
    /// using exactly the same convention as
    /// [`crate::reference::ed::canonical_thermal`] and
    /// [`crate::spectrum::hubbard_jw::JwHubbard`].
    ///
    /// `v_ext.len()` must equal `num_sites`. The result is symmetric:
    /// per-direction hop loops over all basis states emit each
    /// off-diagonal entry exactly once on its own side; the conjugate
    /// is filled when iteration reaches the other basis state.
    #[must_use]
    pub fn from_hubbard(
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
        let dim = m_up * m_dn;
        let lookup_up: HashMap<u32, usize> =
            basis_up.iter().enumerate().map(|(i, &m)| (m, i)).collect();
        let lookup_dn: HashMap<u32, usize> =
            basis_dn.iter().enumerate().map(|(i, &m)| (m, i)).collect();

        let mut triples: Vec<(usize, usize, f64)> = Vec::new();

        for up_idx in 0..m_up {
            for dn_idx in 0..m_dn {
                let r = up_idx * m_dn + dn_idx;
                let up_mask = basis_up[up_idx];
                let dn_mask = basis_dn[dn_idx];

                let doubles = f64::from((up_mask & dn_mask).count_ones());
                let mut diag = on_site_u * doubles;
                for (i, &v) in v_ext.iter().enumerate() {
                    let occ = f64::from(((up_mask >> i) & 1) + ((dn_mask >> i) & 1));
                    diag += v * occ;
                }
                triples.push((r, r, diag));

                emit_spin_hops(
                    &mut triples,
                    up_mask,
                    &lookup_up,
                    up_idx,
                    dn_idx,
                    m_dn,
                    num_sites,
                    hopping_j,
                    true,
                );
                emit_spin_hops(
                    &mut triples,
                    dn_mask,
                    &lookup_dn,
                    up_idx,
                    dn_idx,
                    m_dn,
                    num_sites,
                    hopping_j,
                    false,
                );
            }
        }

        Self::from_triples(dim, &triples)
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_spin_hops(
    triples: &mut Vec<(usize, usize, f64)>,
    mask: u32,
    lookup: &HashMap<u32, usize>,
    up_idx: usize,
    dn_idx: usize,
    m_dn: usize,
    num_sites: usize,
    hopping_j: f64,
    spin_up: bool,
) {
    let r = up_idx * m_dn + dn_idx;
    for bond in 0..num_sites.saturating_sub(1) {
        if (mask >> bond) & 1 == 1 && (mask >> (bond + 1)) & 1 == 0 {
            let after = mask & !(1_u32 << bond);
            let s1 = fermion_sign(mask, bond);
            let new_mask = after | (1_u32 << (bond + 1));
            let s2 = fermion_sign(after, bond + 1);
            let new_idx = lookup[&new_mask];
            let r_new = if spin_up {
                new_idx * m_dn + dn_idx
            } else {
                up_idx * m_dn + new_idx
            };
            triples.push((r_new, r, -hopping_j * s1 * s2));
        }
        if (mask >> (bond + 1)) & 1 == 1 && (mask >> bond) & 1 == 0 {
            let after = mask & !(1_u32 << (bond + 1));
            let s1 = fermion_sign(mask, bond + 1);
            let new_mask = after | (1_u32 << bond);
            let s2 = fermion_sign(after, bond);
            let new_idx = lookup[&new_mask];
            let r_new = if spin_up {
                new_idx * m_dn + dn_idx
            } else {
                up_idx * m_dn + new_idx
            };
            triples.push((r_new, r, -hopping_j * s1 * s2));
        }
    }
}

impl LinearOperator for SparseMatrix {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(
            x.len(),
            self.dim,
            "x.len() = {} != dim {}",
            x.len(),
            self.dim
        );
        assert_eq!(
            y.len(),
            self.dim,
            "y.len() = {} != dim {}",
            y.len(),
            self.dim
        );
        for i in 0..self.dim {
            let mut acc = 0.0_f64;
            let start = self.row_starts[i];
            let end = self.row_starts[i + 1];
            for k in start..end {
                acc = self.values[k].mul_add(x[self.col_indices[k]], acc);
            }
            y[i] = acc;
        }
    }

    /// CSR batch matvec: walks every row once and accumulates into
    /// all output vectors in the inner col-index loop.
    ///
    /// The non-zero pattern (`row_starts`, `col_indices`, `values`)
    /// is loaded into cache once per row regardless of the batch
    /// size, so this is typically much faster than `N` per-RHS
    /// [`Self::apply`] calls for non-trivial sparsity.
    fn apply_batch(&self, xs: &[&[f64]], ys: &mut [Vec<f64>]) {
        assert_eq!(
            xs.len(),
            ys.len(),
            "apply_batch: xs.len() = {} != ys.len() = {}",
            xs.len(),
            ys.len()
        );
        let n_rhs = xs.len();
        for x in xs {
            assert_eq!(
                x.len(),
                self.dim,
                "apply_batch: x.len() = {} != dim {}",
                x.len(),
                self.dim
            );
        }
        for y in ys.iter() {
            assert_eq!(
                y.len(),
                self.dim,
                "apply_batch: y.len() = {} != dim {}",
                y.len(),
                self.dim
            );
        }
        for y in ys.iter_mut() {
            for v in y.iter_mut() {
                *v = 0.0;
            }
        }
        for i in 0..self.dim {
            let start = self.row_starts[i];
            let end = self.row_starts[i + 1];
            for r in 0..n_rhs {
                let mut acc = 0.0_f64;
                let x = xs[r];
                for k in start..end {
                    acc = self.values[k].mul_add(x[self.col_indices[k]], acc);
                }
                ys[r][i] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_matvec_l2_pauli_x_like() {
        let mut h = Mat::<f64>::zeros(2, 2);
        h[(0, 1)] = -1.0;
        h[(1, 0)] = -1.0;
        let mut y = vec![0.0; 2];
        h.apply(&[1.0, 0.0], &mut y);
        assert!(y[0].abs() < 1e-14);
        assert!((y[1] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn sparse_matches_dense_matvec() {
        let mut h = Mat::<f64>::zeros(3, 3);
        h[(0, 0)] = 1.0;
        h[(1, 1)] = -2.0;
        h[(2, 2)] = 3.0;
        h[(0, 1)] = 0.5;
        h[(1, 0)] = 0.5;
        h[(1, 2)] = -0.5;
        h[(2, 1)] = -0.5;
        let sparse = SparseMatrix::from_triples(
            3,
            &[
                (0, 0, 1.0),
                (1, 1, -2.0),
                (2, 2, 3.0),
                (0, 1, 0.5),
                (1, 0, 0.5),
                (1, 2, -0.5),
                (2, 1, -0.5),
            ],
        );
        assert_eq!(sparse.nnz(), 7);
        let x = vec![1.0, -1.0, 0.5];
        let mut y_dense = vec![0.0; 3];
        let mut y_sparse = vec![0.0; 3];
        h.apply(&x, &mut y_dense);
        sparse.apply(&x, &mut y_sparse);
        for i in 0..3 {
            assert!(
                (y_dense[i] - y_sparse[i]).abs() < 1e-14,
                "i={i}: dense={}, sparse={}",
                y_dense[i],
                y_sparse[i]
            );
        }
    }

    #[test]
    fn sparse_from_triples_dedups_duplicates() {
        let s = SparseMatrix::from_triples(2, &[(0, 0, 1.0), (0, 0, 2.0), (0, 1, 0.0)]);
        assert_eq!(s.nnz(), 1);
        let mut y = vec![0.0; 2];
        s.apply(&[1.0, 1.0], &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!(y[1].abs() < 1e-14);
    }

    #[test]
    fn sparse_dim_reports_correct_size() {
        let s = SparseMatrix::from_triples(10, &[]);
        assert_eq!(s.dim(), 10);
        assert_eq!(s.nnz(), 0);
    }

    #[test]
    fn apply_batch_default_impl_matches_per_rhs_apply_on_dense() {
        let mut h = Mat::<f64>::zeros(3, 3);
        h[(0, 0)] = 2.0;
        h[(1, 1)] = 3.0;
        h[(2, 2)] = 5.0;
        h[(0, 1)] = -1.0;
        h[(1, 0)] = -1.0;
        h[(1, 2)] = -1.0;
        h[(2, 1)] = -1.0;
        let xs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, -0.7, 0.3],
        ];
        let xs_refs: Vec<&[f64]> = xs.iter().map(Vec::as_slice).collect();
        let mut ys_batch: Vec<Vec<f64>> = vec![vec![0.0; 3]; 3];
        h.apply_batch(&xs_refs, &mut ys_batch);
        for (i, x) in xs.iter().enumerate() {
            let mut y_single = vec![0.0; 3];
            h.apply(x, &mut y_single);
            for (a, b) in ys_batch[i].iter().zip(y_single.iter()) {
                assert!(
                    (a - b).abs() < 1e-15,
                    "dense apply_batch[{i}] vs apply: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn apply_batch_csr_override_matches_per_rhs_apply_on_sparse() {
        let triples = [
            (0, 0, 2.0),
            (1, 1, 3.0),
            (2, 2, 5.0),
            (3, 3, 7.0),
            (4, 4, 11.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 2, -0.5),
            (2, 1, -0.5),
            (2, 3, -0.3),
            (3, 2, -0.3),
            (3, 4, -0.2),
            (4, 3, -0.2),
        ];
        let sparse = SparseMatrix::from_triples(5, &triples);
        let xs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.3, -0.7, 0.5, 0.2, 0.1],
            vec![0.0, 1.0, -1.0, 1.0, -1.0],
        ];
        let xs_refs: Vec<&[f64]> = xs.iter().map(Vec::as_slice).collect();
        let mut ys_batch: Vec<Vec<f64>> = vec![vec![0.0; 5]; 3];
        sparse.apply_batch(&xs_refs, &mut ys_batch);
        for (i, x) in xs.iter().enumerate() {
            let mut y_single = vec![0.0; 5];
            sparse.apply(x, &mut y_single);
            for (a, b) in ys_batch[i].iter().zip(y_single.iter()) {
                assert!(
                    (a - b).abs() < 1e-15,
                    "sparse apply_batch[{i}] vs apply: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn apply_batch_empty_batch_does_nothing() {
        let sparse = SparseMatrix::from_triples(3, &[(0, 0, 1.0)]);
        let xs: Vec<&[f64]> = Vec::new();
        let mut ys: Vec<Vec<f64>> = Vec::new();
        sparse.apply_batch(&xs, &mut ys);
        assert!(ys.is_empty());
    }
}
