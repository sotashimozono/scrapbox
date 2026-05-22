//! Dense eigendecomposition of a real-symmetric matrix via `faer`.
//!
//! Returns eigenvalues sorted ascending and the column-major eigenvector
//! matrix (`eigvecs.col(k)` = `|ψ_k⟩`, sorted to match `eigenvalues`).

use super::Eigendecomposition;
use crate::error::Result;
use faer::{Mat, Side};

/// Diagonalize a real-symmetric matrix.
pub fn diagonalize(matrix: &Mat<f64>) -> Result<Eigendecomposition> {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "diagonalize requires a square matrix (got {}x{})",
        matrix.nrows(),
        matrix.ncols(),
    );
    let n = matrix.nrows();
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .expect("self-adjoint EVD failed");

    // Read eigenvalues into a Vec.
    let s_col = eigen.S().column_vector();
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|k| (k, s_col[k])).collect();
    // faer returns ascending order in practice, but sort defensively.
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("eigenvalues must be finite"));

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Mat::<f64>::zeros(n, n);
    let u_ref = eigen.U();
    for (new_idx, (orig_idx, value)) in indexed.into_iter().enumerate() {
        eigenvalues.push(value);
        for i in 0..n {
            eigenvectors[(i, new_idx)] = u_ref[(i, orig_idx)];
        }
    }

    Ok(Eigendecomposition {
        eigenvalues,
        eigenvectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimer_tridiagonal_eigvals() {
        // H = [[0, -1], [-1, 0]] has eigenvalues ±1 with eigvecs (1, ±1)/√2.
        let mut h = Mat::<f64>::zeros(2, 2);
        h[(0, 1)] = -1.0;
        h[(1, 0)] = -1.0;
        let eig = diagonalize(&h).unwrap();
        assert_eq!(eig.eigenvalues.len(), 2);
        // Sorted ascending.
        assert!((eig.eigenvalues[0] - (-1.0)).abs() < 1e-12);
        assert!((eig.eigenvalues[1] - 1.0).abs() < 1e-12);
        // Eigenvectors normalized to unit length.
        for k in 0..2 {
            let mut norm_sq = 0.0;
            for i in 0..2 {
                norm_sq += eig.eigenvectors[(i, k)].powi(2);
            }
            assert!((norm_sq - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn diagonal_matrix_recovers_diagonal_entries() {
        let mut h = Mat::<f64>::zeros(3, 3);
        h[(0, 0)] = 3.0;
        h[(1, 1)] = -1.0;
        h[(2, 2)] = 5.0;
        let eig = diagonalize(&h).unwrap();
        assert!((eig.eigenvalues[0] - (-1.0)).abs() < 1e-12);
        assert!((eig.eigenvalues[1] - 3.0).abs() < 1e-12);
        assert!((eig.eigenvalues[2] - 5.0).abs() < 1e-12);
    }
}
