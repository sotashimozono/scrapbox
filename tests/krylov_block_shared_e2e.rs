#![allow(
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
//! Integration tests for `expm_apply_block_shared` (v0.15 epsilon)
//! exercising the real `JwHubbard` matrix-free Hamiltonian rather
//! than the 5x5 toy `SparseMatrix` used in the unit tests.
//!
//! Why these tests exist:
//!
//! - **JwHubbard vs dense EVD**: the unit tests in
//!   `src/spectrum/krylov.rs` validate the algebra on a tridiagonal
//!   `SparseMatrix`. The production caller is `JwHubbard` (Jordan-Wigner
//!   Hubbard, the v0.6 alpha matrix-free operator); we need to verify
//!   the shared block exponentiation gives the same answer as a dense
//!   `exp(scale * H) M` reference on a real Hubbard sector.
//! - **Convergence in m**: block Lanczos should improve monotonically
//!   as the subspace grows, and reach full accuracy once `mN >= n`.
//! - **Scale-zero identity**: `exp(0 * H) = I`, so the routine must
//!   return the input vectors essentially unchanged regardless of `m`.
//! - **Parity with the v0.14 beta per-RHS variant**: at sufficiently
//!   large `m` both schemes evaluate the same operator on the same
//!   inputs and should agree to within Krylov rounding.

use faer::{Mat, Side};
use scrapbox::spectrum::hubbard_jw::JwHubbard;
use scrapbox::spectrum::krylov::{expm_apply_block, expm_apply_block_shared};
use scrapbox::spectrum::linear_operator::LinearOperator;

/// Build an L=4 half-filled spinful Hubbard sector (dim = 6 * 6 = 36)
/// with a non-trivial inhomogeneous potential so the matrix is not
/// degenerate. Returns both the matrix-free operator and the dense
/// materialisation (built by applying the operator to each basis
/// vector — exercises the same code path the matrix-free callers use).
fn build_l4_hubbard() -> (JwHubbard, Mat<f64>) {
    let l = 4_usize;
    let n_up = 2_usize;
    let n_dn = 2_usize;
    let t = 1.0_f64;
    let u = 4.0_f64;
    let v: [f64; 4] = [0.1, -0.1, 0.05, -0.05];
    let jw = JwHubbard::new(l, n_up, n_dn, t, u, &v);
    let n = jw.dim();
    let mut dense = Mat::<f64>::zeros(n, n);
    let mut col = vec![0.0_f64; n];
    for j in 0..n {
        let mut e = vec![0.0_f64; n];
        e[j] = 1.0;
        jw.apply(&e, &mut col);
        for (i, &v_ij) in col.iter().enumerate() {
            dense[(i, j)] = v_ij;
        }
    }
    (jw, dense)
}

/// Reference `exp(scale * H) M` via dense self-adjoint EVD.
fn dense_expm_apply(dense: &Mat<f64>, scale: f64, m_block: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let eig = dense
        .self_adjoint_eigen(Side::Lower)
        .expect("dense reference EVD failed");
    let evals = eig.S().column_vector();
    let u = eig.U();
    let n = dense.nrows();
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(m_block.len());
    for psi in m_block {
        let mut proj = vec![0.0_f64; n];
        for k in 0..n {
            let mut acc = 0.0_f64;
            for (j, &p) in psi.iter().enumerate() {
                acc += u[(j, k)] * p;
            }
            proj[k] = acc * (scale * evals[k]).exp();
        }
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut acc = 0.0_f64;
            for (k, &c) in proj.iter().enumerate() {
                acc += u[(i, k)] * c;
            }
            y[i] = acc;
        }
        out.push(y);
    }
    out
}

/// Deterministic pseudo-random unit vector seeded by `r`.
fn make_unit_psi(n: usize, r: usize) -> Vec<f64> {
    let mut psi = vec![0.0_f64; n];
    for (i, p) in psi.iter_mut().enumerate() {
        *p = (((i + 1) as f64) * ((r as f64) + 7.0)).sin();
    }
    let norm: f64 = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut psi {
        *x /= norm;
    }
    psi
}

fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(ar, br)| ar.iter().zip(br.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f64, f64::max)
}

#[test]
fn expm_apply_block_shared_on_jw_hubbard_matches_dense_evd() {
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    assert_eq!(n, 36, "L=4 half-filled spinful sector has dim 36");
    let psis: Vec<Vec<f64>> = (0..3).map(|r| make_unit_psi(n, r)).collect();
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.5_f64;
    // m = 16 with N = 3 gives big = 48 > n = 36, so the shared block
    // subspace fully spans the Hilbert sector.
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 16);
    let want = dense_expm_apply(&dense, scale, &psis);
    let err = max_abs_diff(&shared, &want);
    assert!(
        err < 1e-8,
        "shared block on JwHubbard L=4 vs dense EVD: max delta {err}"
    );
}

#[test]
fn expm_apply_block_shared_converges_as_m_grows() {
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    let psis: Vec<Vec<f64>> = (0..2).map(|r| make_unit_psi(n, r + 11)).collect();
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.5_f64;
    let want = dense_expm_apply(&dense, scale, &psis);
    let err_low = max_abs_diff(&expm_apply_block_shared(&jw, &psi_refs, scale, 4), &want);
    let err_mid = max_abs_diff(&expm_apply_block_shared(&jw, &psi_refs, scale, 8), &want);
    let err_high = max_abs_diff(&expm_apply_block_shared(&jw, &psi_refs, scale, 20), &want);
    assert!(
        err_low > err_high,
        "Krylov should improve with m: err(m=4) {err_low} should exceed err(m=20) {err_high}"
    );
    // Allow a small slack on the mid-step monotonicity bound: subspace
    // exhaustion is not strictly monotone in the presence of rounding.
    assert!(
        err_mid <= err_low + 1e-10,
        "err(m=8) {err_mid} should not exceed err(m=4) {err_low}"
    );
    // m=20 with N=2 gives big=40 > n=36 → essentially exact.
    assert!(
        err_high < 1e-8,
        "shared block at m=20 N=2 (big=40 > n=36): expected err < 1e-8, got {err_high}"
    );
}

#[test]
fn expm_apply_block_shared_scale_zero_returns_input() {
    let (jw, _) = build_l4_hubbard();
    let n = jw.dim();
    let psis: Vec<Vec<f64>> = (0..3).map(|r| make_unit_psi(n, r + 31)).collect();
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let result = expm_apply_block_shared(&jw, &psi_refs, 0.0, 8);
    for (r, (got, want)) in result.iter().zip(psis.iter()).enumerate() {
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "scale=0 r={r} i={i}: got {a} want {b}"
            );
        }
    }
}

#[test]
fn expm_apply_block_shared_matches_per_rhs_block_at_high_m() {
    let (jw, _) = build_l4_hubbard();
    let n = jw.dim();
    let psis: Vec<Vec<f64>> = (0..3).map(|r| make_unit_psi(n, r + 51)).collect();
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.3_f64;
    // Per-RHS Lanczos at m = n is exact within Krylov arithmetic.
    let per_rhs = expm_apply_block(&jw, &psi_refs, scale, n);
    // Shared at m=12, N=3 → big = 36 = n, also exact.
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 12);
    let err = max_abs_diff(&per_rhs, &shared);
    assert!(
        err < 1e-7,
        "per-RHS vs shared at full subspace coverage: max delta {err}"
    );
}

#[test]
fn expm_apply_block_shared_handles_high_beta_tpq_scale() {
    // exp(-beta * H / 2) with beta = 4 (scale = -2): the regime
    // matrix-free TPQ actually runs in. Check that the shared block
    // path still matches dense EVD even when the spectrum gets stretched.
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    let psi: Vec<f64> = make_unit_psi(n, 5);
    let psis = vec![psi];
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -2.0_f64;
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 24);
    let want = dense_expm_apply(&dense, scale, &psis);
    let err = max_abs_diff(&shared, &want);
    assert!(
        err < 1e-8,
        "shared block at TPQ-realistic scale -2: max delta {err}"
    );
}
