#![allow(
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::redundant_clone,
    clippy::useless_vec
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

#[test]
fn expm_apply_block_shared_n_equals_5_well_covered_block() {
    // N=5 with m=10 on n=36 gives big=50 > 36, comfortably covered.
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    let psis: Vec<Vec<f64>> = (0..5).map(|r| make_unit_psi(n, r + 71)).collect();
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.4_f64;
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 10);
    let want = dense_expm_apply(&dense, scale, &psis);
    let err = max_abs_diff(&shared, &want);
    assert!(err < 1e-8, "N=5 shared block vs dense: max delta {err}");
}

#[test]
fn expm_apply_block_shared_handles_identical_rhs_rank_one_block() {
    // Three identical RHSs form a rank-1 initial block. block_qr must
    // deflate the duplicate columns cleanly; the three outputs must
    // remain identical and match the single-RHS reference.
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    let psi = make_unit_psi(n, 101);
    let psis = vec![psi.clone(), psi.clone(), psi.clone()];
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.5_f64;
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 24);
    let want = dense_expm_apply(&dense, scale, &psis);
    let err = max_abs_diff(&shared, &want);
    assert!(err < 1e-7, "rank-1 RHS block vs dense ref: max delta {err}");
    for r in 1..3 {
        for i in 0..n {
            let d = (shared[r][i] - shared[0][i]).abs();
            assert!(
                d < 1e-10,
                "outputs of identical RHSs diverged: r={r} i={i} delta {d}"
            );
        }
    }
}

#[test]
fn expm_apply_block_shared_zero_rhs_in_batch_yields_zero_output_for_that_slot() {
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    let psi_nonzero = make_unit_psi(n, 113);
    let psi_zero = vec![0.0_f64; n];
    let psis = vec![psi_nonzero.clone(), psi_zero];
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.5_f64;
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 20);
    for (i, &v) in shared[1].iter().enumerate() {
        assert!(v.abs() < 1e-11, "zero RHS slot r=1 i={i}: {v}");
    }
    let want = dense_expm_apply(&dense, scale, &vec![psi_nonzero]);
    for (i, (&a, &b)) in shared[0].iter().zip(want[0].iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8,
            "nonzero RHS slot r=0 i={i}: {a} vs dense {b}"
        );
    }
}

#[test]
fn expm_apply_block_shared_m_overflow_clamps_safely() {
    // Asking for m far beyond n/n_rhs must clamp without panic and
    // still return the correct answer (full subspace coverage).
    let (jw, dense) = build_l4_hubbard();
    let n = jw.dim();
    let psi = make_unit_psi(n, 127);
    let psis = vec![psi];
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.5_f64;
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 10_000);
    let want = dense_expm_apply(&dense, scale, &psis);
    let err = max_abs_diff(&shared, &want);
    assert!(err < 1e-9, "m overflow clamped: err {err}");
}

#[test]
fn expm_apply_block_shared_m_equals_one_is_finite_and_aligned_with_psi() {
    // m=1 with N=1 gives the diagonal approximation exp(scale * <psi|H|psi>) * psi.
    // It is not guaranteed to beat the zeroth-order "just return psi" guess for
    // arbitrary psi (the mean-energy exponential can sit on the wrong side of the
    // true value), but it must (a) not panic, (b) return a finite vector, and
    // (c) remain collinear with psi (single subspace direction).
    let (jw, _) = build_l4_hubbard();
    let n = jw.dim();
    let psi = make_unit_psi(n, 137);
    let psis = vec![psi.clone()];
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.2_f64;
    let m1 = expm_apply_block_shared(&jw, &psi_refs, scale, 1);
    assert_eq!(m1.len(), 1);
    assert_eq!(m1[0].len(), n);
    for &v in &m1[0] {
        assert!(v.is_finite(), "m=1 output must be finite: got {v}");
    }
    // Colinearity: m1[0] = alpha * psi for some scalar alpha (since the
    // 1-dim Krylov subspace is span{psi}). Compute alpha as the inner
    // product and verify the residual is essentially zero.
    let alpha: f64 = m1[0].iter().zip(psi.iter()).map(|(a, b)| a * b).sum();
    let residual: f64 = m1[0]
        .iter()
        .zip(psi.iter())
        .map(|(a, b)| (a - alpha * b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        residual < 1e-10,
        "m=1 output must be collinear with psi, residual {residual}"
    );
}

#[test]
fn expm_apply_block_shared_on_l6_hubbard_matches_per_rhs() {
    // L=6 half-filled spinful: dim = C(6,3)^2 = 400. Larger than L=4
    // and representative of the smallest non-trivial production sector.
    #[allow(clippy::cast_precision_loss)]
    let v: Vec<f64> = (0..6_usize).map(|i| 0.1_f64 * (i as f64 - 2.5)).collect();
    let jw = JwHubbard::new(6, 3, 3, 1.0, 4.0, &v);
    let n = jw.dim();
    assert_eq!(n, 400, "L=6 spinful half-filled sector has dim 400");
    let psis: Vec<Vec<f64>> = (0..2).map(|r| make_unit_psi(n, r + 173)).collect();
    let psi_refs: Vec<&[f64]> = psis.iter().map(Vec::as_slice).collect();
    let scale = -0.3_f64;
    let per_rhs = expm_apply_block(&jw, &psi_refs, scale, 80);
    let shared = expm_apply_block_shared(&jw, &psi_refs, scale, 40);
    let err = max_abs_diff(&per_rhs, &shared);
    assert!(
        err < 1e-6,
        "L=6 per-RHS (m=80) vs shared (m=40 N=2, big=80): max delta {err}"
    );
}

#[test]
fn expm_apply_block_shared_invariant_to_basis_rotation_of_rhs() {
    // exp(scale*H) is a linear map, so if M' = M Q for some N x N
    // orthogonal Q, then exp(scale*H) M' = (exp(scale*H) M) Q.
    // The shared block routine should respect this linearity.
    let (jw, _) = build_l4_hubbard();
    let n = jw.dim();
    let psi_a = make_unit_psi(n, 191);
    let psi_b = make_unit_psi(n, 197);
    let theta = 0.7_f64;
    let c = theta.cos();
    let s = theta.sin();
    let psi_a_rot: Vec<f64> = psi_a
        .iter()
        .zip(psi_b.iter())
        .map(|(a, b)| c * a - s * b)
        .collect();
    let psi_b_rot: Vec<f64> = psi_a
        .iter()
        .zip(psi_b.iter())
        .map(|(a, b)| s * a + c * b)
        .collect();
    let scale = -0.4_f64;
    let m = 18;
    let direct = expm_apply_block_shared(&jw, &[psi_a.as_slice(), psi_b.as_slice()], scale, m);
    let rotated =
        expm_apply_block_shared(&jw, &[psi_a_rot.as_slice(), psi_b_rot.as_slice()], scale, m);
    for i in 0..n {
        let lhs_a = c * direct[0][i] - s * direct[1][i];
        let lhs_b = s * direct[0][i] + c * direct[1][i];
        let d_a = (lhs_a - rotated[0][i]).abs();
        let d_b = (lhs_b - rotated[1][i]).abs();
        assert!(
            d_a < 1e-8 && d_b < 1e-8,
            "basis-rotation linearity i={i}: |Δa|={d_a}, |Δb|={d_b}"
        );
    }
}
