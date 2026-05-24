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

/// Cached Lanczos Krylov subspace generated from `(op, psi, m)`.
///
/// The underlying tridiagonal `T_m` and orthonormal basis `Q` are
/// independent of any spectral function applied afterwards, so the
/// expensive `m` matvecs of building the subspace can be amortised
/// across many evaluations of `exp(scale * H) * psi` at different
/// `scale` values (e.g. a beta sweep over TPQ exp-tilt strengths).
///
/// The empty subspace (`psi` was numerically zero) is represented by
/// an empty `alphas`; [`KrylovSubspace::apply_expm`] returns a zero
/// vector in that case.
#[must_use]
pub struct KrylovSubspace {
    /// `||psi||` of the original starting vector. Restored on output.
    pub norm: f64,
    /// Length-`n` Lanczos basis vectors, in build order.
    pub q_vecs: Vec<Vec<f64>>,
    /// Diagonal entries of the tridiagonal `T_m`.
    pub alphas: Vec<f64>,
    /// Off-diagonal entries; `betas[m - 1]` is the residual norm and is
    /// not needed by [`apply_expm`] but is kept for diagnostics.
    pub betas: Vec<f64>,
}

impl KrylovSubspace {
    /// Krylov subspace size that was actually built.
    ///
    /// Equals the requested `m` unless an invariant subspace forced
    /// early exit.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.alphas.len()
    }

    /// Evaluate `exp(scale * H) * psi` using the cached `(Q, T_m)`.
    ///
    /// Costs an `m x m` EVD plus an `O(n * m)` lift back to the full
    /// space; the `m` matvecs that built the subspace are not redone.
    /// Used by [`expm_apply_multi_scale`] for beta sweeps.
    #[must_use]
    pub fn apply_expm(&self, scale: f64) -> Vec<f64> {
        let n = self.q_vecs.first().map_or(0, Vec::len);
        if n == 0 || self.norm == 0.0 || self.alphas.is_empty() {
            return vec![0.0; n];
        }
        let actual_m = self.alphas.len();
        let mut t = Mat::<f64>::zeros(actual_m, actual_m);
        for k in 0..actual_m {
            t[(k, k)] = self.alphas[k];
            if k + 1 < actual_m {
                t[(k, k + 1)] = self.betas[k];
                t[(k + 1, k)] = self.betas[k];
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
        for (i, q) in self.q_vecs.iter().enumerate() {
            for j in 0..n {
                y[j] += self.norm * y_m[i] * q[j];
            }
        }
        y
    }
}

/// Build an `m`-step Lanczos Krylov subspace for the pair `(op, psi)`
/// with full re-orthogonalisation.
///
/// The result can be reused across many spectral function evaluations
/// via [`KrylovSubspace::apply_expm`]. Used by
/// [`expm_apply_multi_scale`] to amortise matvec cost across a beta
/// sweep at fixed `H` and fixed `psi`.
pub fn build_krylov_subspace<O: LinearOperator>(op: &O, psi: &[f64], m: usize) -> KrylovSubspace {
    let n = op.dim();
    assert_eq!(psi.len(), n, "psi.len() = {} != op.dim() = {n}", psi.len());
    let m = m.min(n).max(1);

    let norm = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return KrylovSubspace {
            norm: 0.0,
            q_vecs: vec![vec![0.0; n]],
            alphas: Vec::new(),
            betas: Vec::new(),
        };
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

    KrylovSubspace {
        norm,
        q_vecs,
        alphas,
        betas,
    }
}

/// Krylov approximation of `exp(scale * H) * psi` with an `m`-step
/// Lanczos. Returns a vector of `op.dim()` length. `m` is clamped to
/// `op.dim()`. Re-orthogonalises against all prior basis vectors at
/// every step (modified Gram-Schmidt) for numerical stability.
#[must_use]
pub fn expm_apply<O: LinearOperator>(op: &O, scale: f64, psi: &[f64], m: usize) -> Vec<f64> {
    build_krylov_subspace(op, psi, m).apply_expm(scale)
}

/// Evaluate `exp(scale_i * H) * psi` for every `scale_i` in `scales`,
/// reusing a single Krylov subspace built for `(op, psi, m)`.
///
/// Cost scales as `O(m * matvec + |scales| * m^3 + |scales| * n * m)`
/// versus `O(|scales| * m * matvec)` for naive per-scale
/// [`expm_apply`]. For TPQ beta sweeps at fixed `H` and `psi` this
/// turns matvec cost from `O(K * m)` to `O(m)`, often a 10-100x
/// speedup at large `n`.
///
/// Output ordering matches `scales`.
#[must_use]
pub fn expm_apply_multi_scale<O: LinearOperator>(
    op: &O,
    psi: &[f64],
    scales: &[f64],
    m: usize,
) -> Vec<Vec<f64>> {
    let subspace = build_krylov_subspace(op, psi, m);
    scales.iter().map(|&s| subspace.apply_expm(s)).collect()
}

/// Adaptive Krylov subspace exponentiation: same contract as
/// [`expm_apply`] but selects the subspace size `m` on the fly using a
/// posteriori error estimate (Saad 1992, eq 3.6):
///
/// ```text
/// |error_m| <= |beta_{m+1} * (exp(scale * T_m))[m, 1]|.
/// ```
///
/// Iterates from `m = 1` up to `max_m`, re-computing the dense
/// `exp(scale * T_m)` on the growing tridiagonal each step (cheap for
/// small `m`). Stops at the first `m` where the bound falls below
/// `tol`, or at `max_m` if the bound never tightens enough.
///
/// Returns `(y, m_used)` so callers can log or assert on the actual
/// subspace size consumed.
#[must_use]
pub fn expm_apply_adaptive<O: LinearOperator>(
    op: &O,
    scale: f64,
    psi: &[f64],
    tol: f64,
    max_m: usize,
) -> (Vec<f64>, usize) {
    let n = op.dim();
    assert_eq!(psi.len(), n, "psi.len() = {} != op.dim() = {n}", psi.len());
    assert!(tol > 0.0, "tol must be positive");
    let max_m = max_m.min(n).max(1);

    let norm = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return (vec![0.0; n], 0);
    }

    let mut q_prev = vec![0.0_f64; n];
    let mut q_curr: Vec<f64> = psi.iter().map(|x| x / norm).collect();
    let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(max_m);
    q_vecs.push(q_curr.clone());

    let mut alphas: Vec<f64> = Vec::with_capacity(max_m);
    let mut betas: Vec<f64> = Vec::with_capacity(max_m);
    let mut beta_prev = 0.0_f64;

    for k in 0..max_m {
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

        // Posteriori error estimate on current m = k + 1
        let m_cur = k + 1;
        if m_cur >= 2 && beta_k > 0.0 {
            // Build T_m_cur, diagonalise, compute |exp(scale * T)[m-1, 0]|
            let mut t = Mat::<f64>::zeros(m_cur, m_cur);
            for j in 0..m_cur {
                t[(j, j)] = alphas[j];
                if j + 1 < m_cur {
                    t[(j, j + 1)] = betas[j];
                    t[(j + 1, j)] = betas[j];
                }
            }
            let eigen = t
                .self_adjoint_eigen(Side::Lower)
                .expect("adaptive Krylov tridiagonal EVD failed");
            let evals = eigen.S().column_vector();
            let u_t = eigen.U();
            let mut elast_first = 0.0_f64;
            for j in 0..m_cur {
                elast_first += u_t[(m_cur - 1, j)] * (scale * evals[j]).exp() * u_t[(0, j)];
            }
            let est = (beta_k * elast_first).abs() * norm;
            if est < tol {
                return finalize(&q_vecs, &alphas, &betas, scale, norm, m_cur);
            }
        }

        if beta_k < 1e-14 || k == max_m - 1 {
            return finalize(&q_vecs, &alphas, &betas, scale, norm, m_cur);
        }
        q_prev = q_curr;
        q_curr = w.iter().map(|x| x / beta_k).collect();
        q_vecs.push(q_curr.clone());
        beta_prev = beta_k;
    }
    finalize(&q_vecs, &alphas, &betas, scale, norm, alphas.len())
}

fn finalize(
    q_vecs: &[Vec<f64>],
    alphas: &[f64],
    betas: &[f64],
    scale: f64,
    norm: f64,
    m: usize,
) -> (Vec<f64>, usize) {
    let n = q_vecs[0].len();
    let mut t = Mat::<f64>::zeros(m, m);
    for k in 0..m {
        t[(k, k)] = alphas[k];
        if k + 1 < m {
            t[(k, k + 1)] = betas[k];
            t[(k + 1, k)] = betas[k];
        }
    }
    let eigen = t
        .self_adjoint_eigen(Side::Lower)
        .expect("adaptive Krylov finalize EVD failed");
    let evals = eigen.S().column_vector();
    let u_t = eigen.U();

    let mut c = vec![0.0_f64; m];
    for k in 0..m {
        c[k] = (scale * evals[k]).exp() * u_t[(0, k)];
    }
    let mut y_m = vec![0.0_f64; m];
    for i in 0..m {
        let mut acc = 0.0_f64;
        for k in 0..m {
            acc += u_t[(i, k)] * c[k];
        }
        y_m[i] = acc;
    }

    let mut y = vec![0.0_f64; n];
    for (i, q) in q_vecs.iter().enumerate().take(m) {
        for j in 0..n {
            y[j] += norm * y_m[i] * q[j];
        }
    }
    (y, m)
}

/// Per-sample Krylov diagnostics collected over a TPQ run. For
/// `KrylovSpec::Fixed` the three fields collapse trivially to the
/// same `m`. For `KrylovSpec::Adaptive` they expose the spread of
/// subspace sizes the Saad bound settled on across samples; surfaced
/// in `tpq_report.json` so users can tune `krylov_tol`.
#[derive(Debug, Clone, Copy)]
pub struct KrylovStats {
    /// Smallest `m_used` across samples.
    pub min_m: usize,
    /// Largest `m_used` across samples.
    pub max_m: usize,
    /// Mean `m_used` across samples.
    pub mean_m: f64,
}

impl KrylovStats {
    /// Build from per-sample `m_used`. Panics if `m_used.is_empty()`.
    #[must_use]
    pub fn from_samples(m_used: &[usize]) -> Self {
        assert!(!m_used.is_empty(), "KrylovStats::from_samples: empty input");
        let min_m = *m_used.iter().min().expect("non-empty");
        let max_m = *m_used.iter().max().expect("non-empty");
        let sum: usize = m_used.iter().sum();
        let mean_m = sum as f64 / m_used.len() as f64;
        Self {
            min_m,
            max_m,
            mean_m,
        }
    }
}

/// User-facing Krylov subspace sizing strategy. `Fixed` uses a fixed
/// `m`-step Lanczos (cheaper if you know the right size). `Adaptive`
/// stops when the Saad 1992 posteriori residual bound falls below
/// `tol`, capped at `max_m`.
#[derive(Debug, Clone, Copy)]
pub enum KrylovSpec {
    /// Fixed `m`-step Lanczos.
    Fixed {
        /// Number of Lanczos steps.
        m: usize,
    },
    /// Adaptive Lanczos via Saad 1992 posteriori residual bound.
    Adaptive {
        /// Stop when residual estimate falls below `tol`.
        tol: f64,
        /// Maximum subspace dimension before forced stop.
        max_m: usize,
    },
}

/// Dispatch [`expm_apply`] or [`expm_apply_adaptive`] depending on the
/// `spec`. Returns just the result vector; the adaptive `m_used` is
/// dropped (use `expm_apply_adaptive` directly if you need it).
#[must_use]
pub fn expm_apply_with_spec<O: LinearOperator>(
    op: &O,
    scale: f64,
    psi: &[f64],
    spec: KrylovSpec,
) -> Vec<f64> {
    expm_apply_with_spec_m(op, scale, psi, spec).0
}

/// Variant of [`expm_apply_with_spec`] that also returns the Krylov
/// subspace size actually used. For [`KrylovSpec::Fixed { m }`] this is
/// trivially `m`; for [`KrylovSpec::Adaptive`] this is the value the
/// Saad 1992 stopping criterion settled on. Useful for diagnostics
/// (e.g. surfacing `min_m / max_m / mean_m` in TPQ output JSON to
/// help users tune `krylov_tol`).
#[must_use]
pub fn expm_apply_with_spec_m<O: LinearOperator>(
    op: &O,
    scale: f64,
    psi: &[f64],
    spec: KrylovSpec,
) -> (Vec<f64>, usize) {
    match spec {
        KrylovSpec::Fixed { m } => (expm_apply(op, scale, psi, m), m),
        KrylovSpec::Adaptive { tol, max_m } => expm_apply_adaptive(op, scale, psi, tol, max_m),
    }
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

    #[test]
    fn expm_apply_adaptive_matches_fixed_5x5() {
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
        let scale = -1.5_f64;
        let psi = vec![0.3_f64, -0.7, 0.2, 0.5, -0.1];
        let y_fixed = expm_apply(&sparse, scale, &psi, 5);
        let (y_adaptive, m_used) = expm_apply_adaptive(&sparse, scale, &psi, 1e-10, 5);
        for i in 0..5 {
            assert!(
                (y_fixed[i] - y_adaptive[i]).abs() < 1e-9,
                "i = {i}: fixed = {}, adaptive = {} (m_used = {m_used})",
                y_fixed[i],
                y_adaptive[i]
            );
        }
        assert!((1..=5).contains(&m_used), "m_used out of range: {m_used}");
    }

    #[test]
    fn expm_apply_adaptive_terminates_early_when_tol_loose() {
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
        let psi = vec![0.3_f64, -0.7, 0.2, 0.5, -0.1];
        let (_, m_loose) = expm_apply_adaptive(&sparse, -1.5, &psi, 1e-2, 5);
        let (_, m_tight) = expm_apply_adaptive(&sparse, -1.5, &psi, 1e-12, 5);
        assert!(
            m_loose <= m_tight,
            "loose tol should stop no later than tight: m_loose = {m_loose}, m_tight = {m_tight}"
        );
    }

    #[test]
    fn expm_apply_adaptive_zero_input_yields_zero() {
        let sparse = SparseMatrix::from_triples(3, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        let (y, m) = expm_apply_adaptive(&sparse, -1.0, &[0.0; 3], 1e-10, 3);
        for &v in &y {
            assert!(v.abs() < 1e-300);
        }
        assert_eq!(m, 0);
    }

    #[test]
    fn expm_apply_multi_scale_matches_per_scale_expm_apply() {
        // v0.12 beta: building one Krylov subspace and evaluating at
        // K scales must agree with calling expm_apply K times.
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
        let psi = vec![0.3, -0.7, 0.5, 0.2, 0.1];
        let scales = [-0.5, -1.0, -2.0, -3.0, -5.0];
        let m = 5;

        let multi = expm_apply_multi_scale(&sparse, &psi, &scales, m);
        assert_eq!(multi.len(), scales.len());
        for (i, &s) in scales.iter().enumerate() {
            let single = expm_apply(&sparse, s, &psi, m);
            for (a, b) in multi[i].iter().zip(single.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "scale {s}: multi = {a}, single = {b}, delta = {}",
                    (a - b).abs()
                );
            }
        }
    }

    #[test]
    fn build_krylov_subspace_dim_clamps_to_op_dim() {
        // Requesting m = 100 on a 3-dim operator returns at most 3.
        let sparse = SparseMatrix::from_triples(3, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        let subspace = build_krylov_subspace(&sparse, &[1.0, 0.0, 0.0], 100);
        assert!(subspace.dim() <= 3, "dim {} must be <= 3", subspace.dim());
        // Diagonal operator with e_0 start: alpha_0 = 1, beta_0 = 0
        // (invariant subspace at k = 0), so dim should be exactly 1.
        assert_eq!(subspace.dim(), 1);
    }

    #[test]
    fn build_krylov_subspace_zero_input_yields_empty_alphas() {
        let sparse = SparseMatrix::from_triples(3, &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        let subspace = build_krylov_subspace(&sparse, &[0.0; 3], 5);
        assert!(subspace.norm.abs() < f64::EPSILON);
        assert_eq!(subspace.dim(), 0);
        // apply_expm on empty subspace must return zero vector of correct length.
        let y = subspace.apply_expm(-1.0);
        assert_eq!(y.len(), 3);
        for &v in &y {
            assert!(v.abs() < f64::EPSILON);
        }
    }
}
