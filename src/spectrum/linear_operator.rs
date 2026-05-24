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

use faer::Mat;

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
}
