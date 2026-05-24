#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
//! Krylov subspace exponentiation: compute `exp(scale * H) * psi`
//! without ever materialising `H` densely.
//!
//! For a real-symmetric linear operator `H` supplied via
//! [`LinearOperator`] and a starting vector `psi`, an `m`-step Lanczos
//! produces a tridiagonal `T_m` and an orthonormal Krylov basis
//! `Q = [q_1, ..., q_m]`. The Krylov approximation
//!
//! ```text
//! exp(scale * H) * psi  ~=  ||psi|| * Q * exp(scale * T_m) * e_1
//! ```
//!
//! converges fast for moderate `|scale| * ||H||`. The dense
//! `exp(scale * T_m)` is computed via the eigendecomposition of `T_m`
//! (faer self-adjoint EVD on the `m x m` tridiagonal block).
//!
//! Used by the matrix-free TPQ path in [`crate::reference::tpq`] to
//! produce `|psi_beta> = exp(-beta H / 2) |psi_0>` without ED.

use super::linear_operator::LinearOperator;
use faer::{Mat, Side};

/// Krylov approximation of `exp(scale * H) * psi` with an `m`-step
/// Lanczos. Returns a vector of `op.dim()` length. `m` is clamped to
/// `op.dim()`. Re-orthogonalises against all prior basis vectors at
/// every step (modified Gram-Schmidt) for numerical stability.
#[must_use]
pub fn expm_apply<O: LinearOperator>(op: &O, scale: f64, psi: &[f64], m: usize) -> Vec<f64> {
    let n = op.dim();
    assert_eq!(psi.len(), n, "psi.len() = {} != op.dim() = {n}", psi.len());
    let m = m.min(n).max(1);

    let norm = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return vec![0.0; n];
    }

    let mut q_prev = vec![0.0_f64; n];
    let mut q_curr: Vec<f64> = psi.iter().map(|x| x / norm).collect();
    let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(m);
    q_vecs.push(q_curr.clone());

    let mut alphas: Vec<f64> = Vec::with_capacity(m);
    let mut betas: Vec<f64> = Vec::with_capacity(m);
    let mut beta_prev = 0.0_f64;

    for k in 0..m {
        let mut w = vec![0.0_f64; n];
        op.apply(&q_curr, &mut w);
        for i in 0..n {
            w[i] -= beta_prev * q_prev[i];
        }
        let alpha_k: f64 = w.iter().zip(q_curr.iter()).map(|(a, b)| a * b).sum();
        for i in 0..n {
            w[i] -= alpha_k * q_curr[i];
        }
        for q in &q_vecs {
            let proj: f64 = w.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n {
                w[i] -= proj * q[i];
            }
        }
        let beta_k = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        alphas.push(alpha_k);
        betas.push(beta_k);
        if beta_k < 1e-14 || k == m - 1 {
            break;
        }
        q_prev = q_curr;
        q_curr = w.iter().map(|x| x / beta_k).collect();
        q_vecs.push(q_curr.clone());
        beta_prev = beta_k;
    }

    let actual_m = alphas.len();

    let mut t = Mat::<f64>::zeros(actual_m, actual_m);
    for k in 0..actual_m {
        t[(k, k)] = alphas[k];
        if k + 1 < actual_m {
            t[(k, k + 1)] = betas[k];
            t[(k + 1, k)] = betas[k];
        }
    }

    let eigen = t
        .self_adjoint_eigen(Side::Lower)
        .expect("Krylov tridiagonal EVD failed");
    let evals = eigen.S().column_vector();
    let u_t = eigen.U();

    let mut c = vec![0.0_f64; actual_m];
    for k in 0..actual_m {
        c[k] = (scale * evals[k]).exp() * u_t[(0, k)];
    }
    let mut y_m = vec![0.0_f64; actual_m];
    for i in 0..actual_m {
        let mut acc = 0.0_f64;
        for k in 0..actual_m {
            acc += u_t[(i, k)] * c[k];
        }
        y_m[i] = acc;
    }

    let mut y = vec![0.0_f64; n];
    for (i, q) in q_vecs.iter().enumerate() {
        for j in 0..n {
            y[j] += norm * y_m[i] * q[j];
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::linear_operator::SparseMatrix;

    #[test]
    fn expm_apply_matches_eigendecomp_on_5x5_spd() {
        let triples = [
            (0, 0, 2.0),
            (1, 1, 3.0),
            (2, 2, 4.0),
            (3, 3, 5.0),
            (4, 4, 6.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 4, -1.0),
            (4, 3, -1.0),
        ];
        let sparse = SparseMatrix::from_triples(5, &triples);

        let mut h_dense = Mat::<f64>::zeros(5, 5);
        for &(i, j, v) in &triples {
            h_dense[(i, j)] = v;
        }
        let eigen = h_dense.self_adjoint_eigen(Side::Lower).expect("dense EVD");
        let evals = eigen.S().column_vector();
        let u_mat = eigen.U();

        let scale = -1.5_f64;
        let psi = vec![0.3_f64, -0.7, 0.2, 0.5, -0.1];

        // Reference: exp(scale * H) psi = U diag(exp(scale * evals)) U^T psi
        let mut d = [0.0_f64; 5];
        for alpha in 0..5 {
            let mut a = 0.0_f64;
            for j in 0..5 {
                a += u_mat[(j, alpha)] * psi[j];
            }
            d[alpha] = a;
        }
        let mut y_ref = [0.0_f64; 5];
        for alpha in 0..5 {
            let w = (scale * evals[alpha]).exp() * d[alpha];
            for i in 0..5 {
                y_ref[i] += u_mat[(i, alpha)] * w;
            }
        }

        let y_kry = expm_apply(&sparse, scale, &psi, 5);
        for i in 0..5 {
            assert!(
                (y_ref[i] - y_kry[i]).abs() < 1e-10,
                "i = {i}: ref = {}, krylov = {}",
                y_ref[i],
                y_kry[i]
            );
        }
    }

    #[test]
    fn expm_apply_zero_input_yields_zero() {
        let sparse = SparseMatrix::from_triples(3, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        let y = expm_apply(&sparse, -1.0, &[0.0; 3], 3);
        for &v in &y {
            assert!(v.abs() < 1e-300);
        }
    }

    #[test]
    fn expm_apply_identity_scale_zero_recovers_input() {
        let sparse = SparseMatrix::from_triples(3, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        let psi = vec![0.5_f64, -0.3, 0.7];
        let y = expm_apply(&sparse, 0.0, &psi, 3);
        for i in 0..3 {
            assert!(
                (y[i] - psi[i]).abs() < 1e-12,
                "i={i}: y={}, psi={}",
                y[i],
                psi[i]
            );
        }
    }
}
