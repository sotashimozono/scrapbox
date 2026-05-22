//! Lanczos tridiagonalization of a real-symmetric matrix.
//!
//! For a dense `n × n` symmetric `H`, we build the Krylov subspace
//! `K_m(H, v_0)` and the tridiagonal matrix `T_m` whose Ritz pairs
//! approximate eigenpairs of `H`. With full re-orthogonalization and
//! `m = n`, Lanczos is mathematically equivalent to a similarity
//! transform to tridiagonal form; dense-diag of `T_m` then yields the
//! full spectrum (modulo round-off).
//!
//! For thermal canonical DFT we sum over all single-particle levels, so
//! the default behavior is **full** Lanczos (`m = n`). Smaller `m`
//! gives partial extremal pairs, useful for very large `n` where the
//! density evaluator (Pratt) tolerates a truncated spectrum at low T.

use super::Eigendecomposition;
use crate::error::{Result, ScrapboxError};
use faer::{Mat, Side};

/// Parameters for Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosParams {
    /// Krylov subspace dimension. `None` = full (`= n`).
    pub krylov_dim: Option<usize>,
    /// Hard cap on iterations (defends against infinite loops from
    /// pathological re-orthogonalization).
    pub max_iter: usize,
    /// `‖β_k v_{k+1}‖` below this triggers early termination
    /// (invariant subspace reached).
    pub tol: f64,
}

/// Run Lanczos on a real-symmetric `n × n` matrix and return the Ritz
/// pairs sorted ascending by eigenvalue.
pub fn diagonalize(matrix: &Mat<f64>, params: &LanczosParams) -> Result<Eigendecomposition> {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "lanczos requires a square matrix (got {}x{})",
        matrix.nrows(),
        matrix.ncols(),
    );
    let n = matrix.nrows();
    let m = params.krylov_dim.unwrap_or(n).min(n);
    if m == 0 {
        return Err(ScrapboxError::ConfigValidation {
            message: "Lanczos krylov_dim must be ≥ 1".into(),
        });
    }
    if params.max_iter < m {
        return Err(ScrapboxError::ConfigValidation {
            message: format!(
                "Lanczos max_iter ({}) is smaller than krylov_dim ({m})",
                params.max_iter,
            ),
        });
    }

    // Starting vector: uniform with deterministic per-site perturbation
    // so every eigenvector has non-zero overlap. Pure unit vectors fail
    // on diagonal matrices (Lanczos terminates before reaching skipped
    // sites); pure uniform vectors fail on translation-symmetric H.
    let mut v_prev = vec![0.0_f64; n];
    let mut v_curr = vec![0.0_f64; n];
    for (i, x) in v_curr.iter_mut().enumerate() {
        *x = ((i as f64) * 0.137).sin().mul_add(1.0e-3, 1.0);
    }
    let norm = vec_norm(&v_curr);
    for x in &mut v_curr {
        *x /= norm;
    }

    let mut q_basis: Vec<Vec<f64>> = Vec::with_capacity(m);
    q_basis.push(v_curr.clone());

    let mut alpha = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m);

    let mut beta_prev = 0.0_f64;
    let mut effective_m = m;

    for k in 0..m {
        let mut w = mat_vec(matrix, &v_curr);
        if k > 0 {
            axpy(&mut w, -beta_prev, &v_prev);
        }
        let a_k = dot(&v_curr, &w);
        alpha.push(a_k);
        axpy(&mut w, -a_k, &v_curr);

        // Full re-orthogonalization against all stored Lanczos vectors,
        // twice (single-pass Gram-Schmidt is not numerically stable).
        for _pass in 0..2 {
            for q in &q_basis {
                let c = dot(q, &w);
                axpy(&mut w, -c, q);
            }
        }

        let b_k = vec_norm(&w);
        beta.push(b_k);

        if k + 1 == m {
            break;
        }
        if b_k < params.tol {
            effective_m = k + 1;
            break;
        }

        let inv = 1.0 / b_k;
        for x in &mut w {
            *x *= inv;
        }
        v_prev.clone_from(&v_curr);
        v_curr = w;
        q_basis.push(v_curr.clone());
        beta_prev = b_k;
    }

    let mut t = Mat::<f64>::zeros(effective_m, effective_m);
    for k in 0..effective_m {
        t[(k, k)] = alpha[k];
        if k + 1 < effective_m {
            let off = beta[k];
            t[(k, k + 1)] = off;
            t[(k + 1, k)] = off;
        }
    }

    let eigen = t
        .self_adjoint_eigen(Side::Lower)
        .expect("self-adjoint EVD failed");
    let s_col = eigen.S().column_vector();
    let mut indexed: Vec<(usize, f64)> = (0..effective_m).map(|k| (k, s_col[k])).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("Ritz values must be finite"));

    let u_t = eigen.U();
    let mut eigenvalues = Vec::with_capacity(effective_m);
    let mut eigenvectors = Mat::<f64>::zeros(n, effective_m);
    for (new_idx, (orig_idx, value)) in indexed.into_iter().enumerate() {
        eigenvalues.push(value);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..effective_m {
                acc = u_t[(j, orig_idx)].mul_add(q_basis[j][i], acc);
            }
            eigenvectors[(i, new_idx)] = acc;
        }
    }

    Ok(Eigendecomposition {
        eigenvalues,
        eigenvectors,
    })
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi = a.mul_add(*xi, *yi);
    }
}

fn mat_vec(m: &Mat<f64>, v: &[f64]) -> Vec<f64> {
    let n = m.nrows();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc = m[(i, j)].mul_add(v[j], acc);
        }
        out[i] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params(n: usize) -> LanczosParams {
        LanczosParams {
            krylov_dim: None,
            max_iter: n * 4,
            tol: 1e-14,
        }
    }

    #[test]
    fn dimer_tridiagonal_matches_dense() {
        let mut h = Mat::<f64>::zeros(2, 2);
        h[(0, 1)] = -1.0;
        h[(1, 0)] = -1.0;
        let eig = diagonalize(&h, &default_params(2)).unwrap();
        assert_eq!(eig.eigenvalues.len(), 2);
        assert!((eig.eigenvalues[0] - (-1.0)).abs() < 1e-12);
        assert!((eig.eigenvalues[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn diagonal_matrix_recovers_diagonal_entries() {
        let mut h = Mat::<f64>::zeros(3, 3);
        h[(0, 0)] = 3.0;
        h[(1, 1)] = -1.0;
        h[(2, 2)] = 5.0;
        let eig = diagonalize(&h, &default_params(3)).unwrap();
        assert!((eig.eigenvalues[0] - (-1.0)).abs() < 1e-10);
        assert!((eig.eigenvalues[1] - 3.0).abs() < 1e-10);
        assert!((eig.eigenvalues[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn full_lanczos_matches_dense_on_random_symmetric_6x6() {
        let n = 6;
        let mut h = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = ((((i * 7 + j * 13 + 1) % 17) as f64) - 8.0) * 0.25;
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        let lanczos = diagonalize(&h, &default_params(n)).unwrap();
        let dense = super::super::dense_diag::diagonalize(&h).unwrap();
        for k in 0..n {
            assert!(
                (lanczos.eigenvalues[k] - dense.eigenvalues[k]).abs() < 1e-9,
                "k={k} lanczos={} dense={}",
                lanczos.eigenvalues[k],
                dense.eigenvalues[k],
            );
        }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let n = 5;
        let mut h = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = ((((i * 3 + j * 11 + 2) % 19) as f64) - 9.0) * 0.1;
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        let eig = diagonalize(&h, &default_params(n)).unwrap();
        for k in 0..n {
            let mut norm_sq = 0.0;
            for i in 0..n {
                norm_sq += eig.eigenvectors[(i, k)].powi(2);
            }
            assert!((norm_sq - 1.0).abs() < 1e-9, "‖v_{k}‖² = {norm_sq}");
        }
        for k in 0..n {
            for l in (k + 1)..n {
                let mut ip = 0.0;
                for i in 0..n {
                    ip += eig.eigenvectors[(i, k)] * eig.eigenvectors[(i, l)];
                }
                assert!(ip.abs() < 1e-9, "⟨v_{k}|v_{l}⟩ = {ip}");
            }
        }
    }

    #[test]
    fn rejects_max_iter_below_krylov_dim() {
        let h = Mat::<f64>::zeros(4, 4);
        let bad = LanczosParams {
            krylov_dim: None,
            max_iter: 2,
            tol: 1e-12,
        };
        assert!(diagonalize(&h, &bad).is_err());
    }
}
