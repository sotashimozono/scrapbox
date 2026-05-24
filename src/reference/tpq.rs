#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
//! Typical Pure Quantum (TPQ) sampler for canonical thermal observables.
//!
//! For a Hamiltonian `H` with eigendecomposition `H |k> = E_k |k>`,
//! the canonical thermal expectation of an operator `O` is
//!
//! ```text
//! <O> = Tr(O e^{-beta H}) / Tr(e^{-beta H}).
//! ```
//!
//! TPQ replaces the trace by an average over random pure states. Given
//! a random vector `|psi_0> = sum_k c_k |k>` with real Gaussian
//! `c_k ~ N(0, 1)` (real amplitudes are sufficient because the lattice
//! Hubbard Hamiltonian is real-symmetric; complex Gaussian halves the
//! statistical variance but is not required for correctness), the
//! canonical TPQ state is
//!
//! ```text
//! |psi_beta> = e^{-beta H / 2} |psi_0> = sum_k e^{-beta E_k / 2} c_k |k>.
//! ```
//!
//! `<psi_beta| O |psi_beta> / <psi_beta|psi_beta>` converges to `<O>`
//! as `dim -> infinity` (concentration of measure). For finite `dim`,
//! averaging over multiple `|psi_0>` reduces variance as
//! `1 / sqrt(N_samples)`.

use super::ed::EdResult;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn box_muller(rng: &mut StdRng) -> f64 {
    // Generate one N(0,1) sample via Box-Muller on two U(0,1) draws.
    let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
    let u2: f64 = rng.r#gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate two independent N(0, 1) samples per call (no work wasted on
/// the sin partner). Used by the complex-amplitude TPQ paths where each
/// sample slot consumes one (re, im) pair.
fn box_muller_pair(rng: &mut StdRng) -> (f64, f64) {
    let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
    let u2: f64 = rng.r#gen();
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// TPQ canonical-thermal density estimate. Per-site occupation
/// (spin-summed) averaged over `n_samples` random pure states.
#[must_use]
pub fn tpq_density(ed: &EdResult, beta: f64, n_samples: usize, seed: u64) -> Vec<f64> {
    assert!(n_samples > 0, "n_samples must be >= 1");
    assert!(
        !ed.eigenvalues.is_empty(),
        "tpq_density requires non-empty spectrum"
    );
    let dim = ed.eigenvalues.len();
    let shift = ed.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut acc_density = vec![0.0_f64; ed.num_sites];
    let mut acc_norm_sq = 0.0_f64;

    for _ in 0..n_samples {
        // Random real Gaussian vector in the original basis; sufficient
        // for purely real Hermitian H (lattice Hubbard).
        let psi_0: Vec<f64> = (0..dim).map(|_| box_muller(&mut rng)).collect();

        // Energy-basis amplitudes c_k = <k|psi_0> = sum_j U[j, k] psi_0[j].
        let mut c_k = vec![0.0_f64; dim];
        for k in 0..dim {
            let mut sum = 0.0;
            for (j, &amp) in psi_0.iter().enumerate() {
                sum += ed.eigenvectors[(j, k)] * amp;
            }
            c_k[k] = sum;
        }

        // |psi_beta> in original basis:
        //   psi_beta[j] = sum_k U[j, k] e^{-beta (E_k - shift)/2} c_k.
        let mut psi_beta = vec![0.0_f64; dim];
        for k in 0..dim {
            let w = (-beta * (ed.eigenvalues[k] - shift) * 0.5).exp() * c_k[k];
            for (j, p) in psi_beta.iter_mut().enumerate() {
                *p += w * ed.eigenvectors[(j, k)];
            }
        }

        let mut sample_norm_sq = 0.0_f64;
        for &p in &psi_beta {
            sample_norm_sq += p * p;
        }
        for site in 0..ed.num_sites {
            let mut occ_acc = 0.0_f64;
            for (j, &(up_mask, dn_mask)) in ed.joint.iter().enumerate() {
                let occ = f64::from(((up_mask >> site) & 1) + ((dn_mask >> site) & 1));
                occ_acc += psi_beta[j] * psi_beta[j] * occ;
            }
            acc_density[site] += occ_acc;
        }
        acc_norm_sq += sample_norm_sq;
    }

    assert!(
        acc_norm_sq > 0.0,
        "tpq_density: accumulated norm is zero -- all samples vanished after          e^(-beta H / 2) projection; check that beta is finite and the          spectrum is non-degenerate"
    );
    for x in &mut acc_density {
        *x /= acc_norm_sq;
    }
    acc_density
}

/// Two-Hamiltonian canonical thermal work statistics for a sudden
/// quench `H_init -> H_final` from the canonical state at inverse
/// temperature `beta` of `H_init`.
///
/// `ed_init` and `ed_final` must share the same Hilbert dimension and
/// joint basis (caller's responsibility -- this only checks
/// dimension).
///
/// Per Sugiura-Shimizu the canonical thermal state of `H_init` is
/// `|psi_beta> = exp(-beta H_init / 2) |psi_0>` for a random Gaussian
/// `|psi_0>`. For a sudden quench the work operator is
/// `W = H_final - H_init` and
///
/// ```text
/// <W>      = <psi_beta| W      |psi_beta> / <psi_beta|psi_beta>,
/// <W^2>    = <psi_beta| W^2    |psi_beta> / <psi_beta|psi_beta>,
/// sigma_W^2 = <W^2> - <W>^2.
/// ```
///
/// Numerator and denominator are pooled across samples (Sugiura-Shimizu
/// unbiased estimator); `mean_w_stderr` is the sample-to-sample
/// standard error of the per-sample ratio `(psi_beta . W psi_beta) /
/// (psi_beta . psi_beta)`, suitable for FDR closure checks.
#[derive(Debug, Clone, Copy)]
pub struct TpqWorkStats {
    /// Pooled estimate of `<W>`.
    pub mean_w: f64,
    /// `<W^2> - <W>^2` (the second cumulant; i.e. work variance).
    pub work_variance: f64,
    /// Sample-to-sample standard error of `mean_w`.
    pub mean_w_stderr: f64,
    /// Number of TPQ samples used.
    pub n_samples: usize,
}

/// TPQ estimate of canonical thermal work statistics under a sudden
/// quench `ed_init -> ed_final` at inverse temperature `beta`.
#[must_use]
pub fn tpq_work_statistics(
    ed_init: &EdResult,
    ed_final: &EdResult,
    beta: f64,
    n_samples: usize,
    seed: u64,
) -> TpqWorkStats {
    assert!(n_samples > 0, "n_samples must be >= 1");
    assert!(
        !ed_init.eigenvalues.is_empty(),
        "tpq_work_statistics: ed_init spectrum is empty"
    );
    assert_eq!(
        ed_init.eigenvalues.len(),
        ed_final.eigenvalues.len(),
        "tpq_work_statistics: ed_init and ed_final must share Hilbert dimension          (got {} vs {})",
        ed_init.eigenvalues.len(),
        ed_final.eigenvalues.len()
    );

    let dim = ed_init.eigenvalues.len();
    let shift = ed_init
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut acc_num_w = 0.0_f64;
    let mut acc_num_w_sq = 0.0_f64;
    let mut acc_denom = 0.0_f64;
    let mut per_sample_mean_w: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let psi_0: Vec<f64> = (0..dim).map(|_| box_muller(&mut rng)).collect();

        // c_k = U_init^T psi_0  (initial-eigenbasis amplitudes)
        let mut c_k = vec![0.0_f64; dim];
        for k in 0..dim {
            let mut acc = 0.0_f64;
            for (j, &a) in psi_0.iter().enumerate() {
                acc += ed_init.eigenvectors[(j, k)] * a;
            }
            c_k[k] = acc;
        }

        // psi_beta[j] = sum_k U_init[j, k] * e^{-beta(E_init,k - shift)/2} * c_k
        let mut psi_beta = vec![0.0_f64; dim];
        for k in 0..dim {
            let w = (-beta * (ed_init.eigenvalues[k] - shift) * 0.5).exp() * c_k[k];
            for (j, p) in psi_beta.iter_mut().enumerate() {
                *p += w * ed_init.eigenvectors[(j, k)];
            }
        }

        // H_init psi_beta and H_final psi_beta via their eigen-decomps
        let h_init_psi = apply_hamiltonian(&psi_beta, ed_init);
        let h_final_psi = apply_hamiltonian(&psi_beta, ed_final);

        // delta_psi = (H_final - H_init) psi_beta
        let delta_psi: Vec<f64> = h_final_psi
            .iter()
            .zip(h_init_psi.iter())
            .map(|(a, b)| a - b)
            .collect();

        let num_w_i: f64 = psi_beta
            .iter()
            .zip(delta_psi.iter())
            .map(|(p, d)| p * d)
            .sum();
        let num_w_sq_i: f64 = delta_psi.iter().map(|d| d * d).sum();
        let denom_i: f64 = psi_beta.iter().map(|p| p * p).sum();

        acc_num_w += num_w_i;
        acc_num_w_sq += num_w_sq_i;
        acc_denom += denom_i;
        per_sample_mean_w.push(num_w_i / denom_i);
    }

    assert!(
        acc_denom > 0.0,
        "tpq_work_statistics: accumulated norm is zero -- all samples vanished after          e^(-beta H_init / 2)"
    );
    let mean_w = acc_num_w / acc_denom;
    let mean_w_sq = acc_num_w_sq / acc_denom;
    let work_variance = mean_w_sq - mean_w * mean_w;

    let n_f = n_samples as f64;
    let sample_mean: f64 = per_sample_mean_w.iter().sum::<f64>() / n_f;
    let sample_var: f64 = per_sample_mean_w
        .iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / n_f;
    let mean_w_stderr = (sample_var / n_f).sqrt();

    TpqWorkStats {
        mean_w,
        work_variance,
        mean_w_stderr,
        n_samples,
    }
}

/// Apply `H` (encoded by its eigen-decomposition in `ed`) to `psi`
/// in the original basis: `H psi = U diag(E) U^T psi`.
fn apply_hamiltonian(psi: &[f64], ed: &EdResult) -> Vec<f64> {
    let dim = ed.eigenvalues.len();
    debug_assert_eq!(psi.len(), dim);
    let mut d = vec![0.0_f64; dim];
    for alpha in 0..dim {
        let mut acc = 0.0_f64;
        for (j, &p) in psi.iter().enumerate() {
            acc += ed.eigenvectors[(j, alpha)] * p;
        }
        d[alpha] = acc;
    }
    let mut out = vec![0.0_f64; dim];
    for alpha in 0..dim {
        let w = ed.eigenvalues[alpha] * d[alpha];
        for (j, o) in out.iter_mut().enumerate() {
            *o += w * ed.eigenvectors[(j, alpha)];
        }
    }
    out
}

/// Complex-amplitude TPQ canonical-thermal density estimate.
///
/// Identical contract to [`tpq_density`] but draws `c_k = (re + i im)/sqrt(2)`
/// with `re, im ~ N(0, 1)`. The complex variance of `|c_k|^2` is half
/// the real variance, so for diagonal observables (per-site density,
/// per-state weights) this halves the per-sample variance.
///
/// Internally `psi_beta` is stored as two real vectors and only
/// `|psi_beta|^2 = re^2 + im^2` is accumulated; output is real.
#[must_use]
pub fn tpq_density_complex(ed: &EdResult, beta: f64, n_samples: usize, seed: u64) -> Vec<f64> {
    assert!(n_samples > 0, "n_samples must be >= 1");
    assert!(
        !ed.eigenvalues.is_empty(),
        "tpq_density_complex requires non-empty spectrum"
    );
    let dim = ed.eigenvalues.len();
    let shift = ed.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut acc_density = vec![0.0_f64; ed.num_sites];
    let mut acc_norm_sq = 0.0_f64;

    let scale = 1.0 / std::f64::consts::SQRT_2;
    for _ in 0..n_samples {
        let (psi_re, psi_im) = sample_complex_psi_beta(&mut rng, ed, beta, shift, scale, dim);
        let mut sample_norm_sq = 0.0_f64;
        for j in 0..dim {
            sample_norm_sq += psi_re[j] * psi_re[j] + psi_im[j] * psi_im[j];
        }
        for site in 0..ed.num_sites {
            let mut occ_acc = 0.0_f64;
            for (j, &(up_mask, dn_mask)) in ed.joint.iter().enumerate() {
                let occ = f64::from(((up_mask >> site) & 1) + ((dn_mask >> site) & 1));
                occ_acc += (psi_re[j] * psi_re[j] + psi_im[j] * psi_im[j]) * occ;
            }
            acc_density[site] += occ_acc;
        }
        acc_norm_sq += sample_norm_sq;
    }

    assert!(
        acc_norm_sq > 0.0,
        "tpq_density_complex: accumulated norm is zero"
    );
    for x in &mut acc_density {
        *x /= acc_norm_sq;
    }
    acc_density
}

/// Complex-amplitude TPQ canonical thermal work statistics.
///
/// Identical contract to [`tpq_work_statistics`] but uses
/// complex Gaussian `c_k = (re + i im)/sqrt(2)`. Halves the per-sample
/// variance of `<W>` and `<W^2>` relative to the real-amplitude variant,
/// at the cost of two real vector workspaces per sample.
#[must_use]
pub fn tpq_work_statistics_complex(
    ed_init: &EdResult,
    ed_final: &EdResult,
    beta: f64,
    n_samples: usize,
    seed: u64,
) -> TpqWorkStats {
    assert!(n_samples > 0, "n_samples must be >= 1");
    assert!(
        !ed_init.eigenvalues.is_empty(),
        "tpq_work_statistics_complex: ed_init spectrum is empty"
    );
    assert_eq!(
        ed_init.eigenvalues.len(),
        ed_final.eigenvalues.len(),
        "tpq_work_statistics_complex: ed_init and ed_final must share Hilbert dimension          (got {} vs {})",
        ed_init.eigenvalues.len(),
        ed_final.eigenvalues.len()
    );

    let dim = ed_init.eigenvalues.len();
    let shift = ed_init
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut acc_num_w = 0.0_f64;
    let mut acc_num_w_sq = 0.0_f64;
    let mut acc_denom = 0.0_f64;
    let mut per_sample_mean_w: Vec<f64> = Vec::with_capacity(n_samples);

    let scale = 1.0 / std::f64::consts::SQRT_2;
    for _ in 0..n_samples {
        let (psi_re, psi_im) = sample_complex_psi_beta(&mut rng, ed_init, beta, shift, scale, dim);

        let h_init_psi_re = apply_hamiltonian(&psi_re, ed_init);
        let h_init_psi_im = apply_hamiltonian(&psi_im, ed_init);
        let h_final_psi_re = apply_hamiltonian(&psi_re, ed_final);
        let h_final_psi_im = apply_hamiltonian(&psi_im, ed_final);

        let delta_re: Vec<f64> = h_final_psi_re
            .iter()
            .zip(h_init_psi_re.iter())
            .map(|(a, b)| a - b)
            .collect();
        let delta_im: Vec<f64> = h_final_psi_im
            .iter()
            .zip(h_init_psi_im.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mut num_w_i = 0.0_f64;
        let mut num_w_sq_i = 0.0_f64;
        let mut denom_i = 0.0_f64;
        for j in 0..dim {
            // <psi|W|psi> picks up the real part of psi^* W psi; since
            // H_final and H_init are real-symmetric the imag cross-terms
            // cancel and we get (re * delta_re + im * delta_im).
            num_w_i += psi_re[j] * delta_re[j] + psi_im[j] * delta_im[j];
            num_w_sq_i += delta_re[j] * delta_re[j] + delta_im[j] * delta_im[j];
            denom_i += psi_re[j] * psi_re[j] + psi_im[j] * psi_im[j];
        }

        acc_num_w += num_w_i;
        acc_num_w_sq += num_w_sq_i;
        acc_denom += denom_i;
        per_sample_mean_w.push(num_w_i / denom_i);
    }

    assert!(
        acc_denom > 0.0,
        "tpq_work_statistics_complex: accumulated norm is zero"
    );
    let mean_w = acc_num_w / acc_denom;
    let mean_w_sq = acc_num_w_sq / acc_denom;
    let work_variance = mean_w_sq - mean_w * mean_w;

    let n_f = n_samples as f64;
    let sample_mean: f64 = per_sample_mean_w.iter().sum::<f64>() / n_f;
    let sample_var: f64 = per_sample_mean_w
        .iter()
        .map(|x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / n_f;
    let mean_w_stderr = (sample_var / n_f).sqrt();

    TpqWorkStats {
        mean_w,
        work_variance,
        mean_w_stderr,
        n_samples,
    }
}

/// Build one (psi_beta_re, psi_beta_im) sample in the original basis
/// from a complex Gaussian `c_k = scale * (re + i im)`.
fn sample_complex_psi_beta(
    rng: &mut StdRng,
    ed: &EdResult,
    beta: f64,
    shift: f64,
    scale: f64,
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut psi_0_re = vec![0.0_f64; dim];
    let mut psi_0_im = vec![0.0_f64; dim];
    for j in 0..dim {
        let (a, b) = box_muller_pair(rng);
        psi_0_re[j] = scale * a;
        psi_0_im[j] = scale * b;
    }
    let mut c_k_re = vec![0.0_f64; dim];
    let mut c_k_im = vec![0.0_f64; dim];
    for k in 0..dim {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for j in 0..dim {
            sr += ed.eigenvectors[(j, k)] * psi_0_re[j];
            si += ed.eigenvectors[(j, k)] * psi_0_im[j];
        }
        c_k_re[k] = sr;
        c_k_im[k] = si;
    }
    let mut psi_re = vec![0.0_f64; dim];
    let mut psi_im = vec![0.0_f64; dim];
    for k in 0..dim {
        let w = (-beta * (ed.eigenvalues[k] - shift) * 0.5).exp();
        let wr = w * c_k_re[k];
        let wi = w * c_k_im[k];
        for j in 0..dim {
            psi_re[j] += wr * ed.eigenvectors[(j, k)];
            psi_im[j] += wi * ed.eigenvectors[(j, k)];
        }
    }
    (psi_re, psi_im)
}

#[cfg(test)]
mod tests {
    use super::super::ed;
    use super::*;

    #[test]
    fn tpq_dimer_density_at_half_filling_is_one() {
        let result = ed::canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let tpq = tpq_density(&result, 2.0, 50, 42);
        for (i, &n) in tpq.iter().enumerate() {
            assert!(
                (n - 1.0).abs() < 0.1,
                "site {i}: TPQ = {n}, ED = 1 (50 samples, dim=4)"
            );
        }
    }

    #[test]
    fn tpq_l4_converges_to_ed_density_with_enough_samples() {
        let v = [0.1_f64, -0.1, 0.1, -0.1];
        let result = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v);
        let n_ed = ed::thermal_density(&result, 2.0);
        let n_tpq = tpq_density(&result, 2.0, 500, 7);
        for i in 0..4 {
            assert!(
                (n_tpq[i] - n_ed[i]).abs() < 0.05,
                "site {i}: TPQ = {}, ED = {} (500 samples, dim=36)",
                n_tpq[i],
                n_ed[i]
            );
        }
    }

    #[test]
    fn tpq_l6_self_averaging_with_modest_samples() {
        let result = ed::canonical_thermal(6, 3, 3, 1.0, 4.0, &[0.0; 6]);
        let tpq = tpq_density(&result, 2.0, 100, 99);
        for (i, &n) in tpq.iter().enumerate() {
            assert!(
                (n - 1.0).abs() < 0.02,
                "site {i}: TPQ = {n} (100 samples, dim=400)"
            );
        }
    }

    #[test]
    fn tpq_deterministic_with_same_seed() {
        let result = ed::canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let a = tpq_density(&result, 2.0, 20, 123);
        let b = tpq_density(&result, 2.0, 20, 123);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-14, "site {i}: not reproducible");
        }
    }

    #[test]
    fn tpq_different_seeds_produce_different_estimates() {
        let result = ed::canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let a = tpq_density(&result, 2.0, 20, 1);
        let b = tpq_density(&result, 2.0, 20, 2);
        let diff: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-10, "different seeds should not collide");
    }

    #[test]
    fn tpq_work_zero_quench_gives_zero_mean_and_variance() {
        let v = [0.0_f64, 0.0, 0.0, 0.0];
        let ed_init = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v);
        let stats = tpq_work_statistics(&ed_init, &ed_init, 2.0, 30, 11);
        assert!(stats.mean_w.abs() < 1e-10, "got mean_w = {}", stats.mean_w);
        assert!(
            stats.work_variance.abs() < 1e-10,
            "got work_variance = {}",
            stats.work_variance
        );
    }

    #[test]
    fn tpq_work_matches_ed_for_l4_sudden_quench() {
        let v_init = [0.0_f64; 4];
        let v_final = [0.3_f64, -0.3, 0.3, -0.3];
        let ed_init = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);

        let dim = ed_init.eigenvalues.len();
        let beta = 2.0;
        let shift = ed_init
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let mut z = 0.0_f64;
        let mut mean_w_ed = 0.0_f64;
        let mut mean_w_sq_ed = 0.0_f64;
        for n in 0..dim {
            let weight = (-beta * (ed_init.eigenvalues[n] - shift)).exp();
            z += weight;
            let mut psi_n = vec![0.0_f64; dim];
            for j in 0..dim {
                psi_n[j] = ed_init.eigenvectors[(j, n)];
            }
            let h_init_psi = apply_hamiltonian(&psi_n, &ed_init);
            let h_final_psi = apply_hamiltonian(&psi_n, &ed_final);
            let delta_psi: Vec<f64> = h_final_psi
                .iter()
                .zip(h_init_psi.iter())
                .map(|(a, b)| a - b)
                .collect();
            let w_n: f64 = psi_n.iter().zip(delta_psi.iter()).map(|(p, d)| p * d).sum();
            let w_sq_n: f64 = delta_psi.iter().map(|d| d * d).sum();
            mean_w_ed += weight * w_n;
            mean_w_sq_ed += weight * w_sq_n;
        }
        mean_w_ed /= z;
        mean_w_sq_ed /= z;
        let sigma_w_sq_ed = mean_w_sq_ed - mean_w_ed * mean_w_ed;

        let stats = tpq_work_statistics(&ed_init, &ed_final, beta, 600, 7);
        assert!(
            (stats.mean_w - mean_w_ed).abs() < 0.05,
            "TPQ <W> = {} vs ED {} (stderr {})",
            stats.mean_w,
            mean_w_ed,
            stats.mean_w_stderr
        );
        assert!(
            (stats.work_variance - sigma_w_sq_ed).abs() < 0.1,
            "TPQ sigma_W^2 = {} vs ED {}",
            stats.work_variance,
            sigma_w_sq_ed
        );
        assert!(stats.mean_w_stderr > 0.0);
    }

    #[test]
    fn tpq_work_deterministic_with_same_seed() {
        let v_init = [0.0_f64; 4];
        let v_final = [0.2_f64, -0.2, 0.2, -0.2];
        let ed_init = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);
        let a = tpq_work_statistics(&ed_init, &ed_final, 2.0, 25, 42);
        let b = tpq_work_statistics(&ed_init, &ed_final, 2.0, 25, 42);
        assert!((a.mean_w - b.mean_w).abs() < 1e-14);
        assert!((a.work_variance - b.work_variance).abs() < 1e-14);
    }

    #[test]
    fn tpq_density_complex_converges_at_l4() {
        let v = [0.1_f64, -0.1, 0.1, -0.1];
        let result = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v);
        let n_ed = ed::thermal_density(&result, 2.0);
        let n_tpq = tpq_density_complex(&result, 2.0, 500, 7);
        for i in 0..4 {
            assert!(
                (n_tpq[i] - n_ed[i]).abs() < 0.05,
                "site {i}: complex TPQ = {}, ED = {}",
                n_tpq[i],
                n_ed[i]
            );
        }
    }

    #[test]
    fn tpq_work_complex_matches_ed_for_l4_sudden_quench() {
        let v_init = [0.0_f64; 4];
        let v_final = [0.3_f64, -0.3, 0.3, -0.3];
        let ed_init = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);

        let dim = ed_init.eigenvalues.len();
        let beta = 2.0;
        let shift = ed_init
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let mut z = 0.0_f64;
        let mut mean_w_ed = 0.0_f64;
        let mut mean_w_sq_ed = 0.0_f64;
        for n in 0..dim {
            let weight = (-beta * (ed_init.eigenvalues[n] - shift)).exp();
            z += weight;
            let mut psi_n = vec![0.0_f64; dim];
            for j in 0..dim {
                psi_n[j] = ed_init.eigenvectors[(j, n)];
            }
            let h_init_psi = apply_hamiltonian(&psi_n, &ed_init);
            let h_final_psi = apply_hamiltonian(&psi_n, &ed_final);
            let delta: Vec<f64> = h_final_psi
                .iter()
                .zip(h_init_psi.iter())
                .map(|(a, b)| a - b)
                .collect();
            mean_w_ed += weight
                * psi_n
                    .iter()
                    .zip(delta.iter())
                    .map(|(p, d)| p * d)
                    .sum::<f64>();
            mean_w_sq_ed += weight * delta.iter().map(|d| d * d).sum::<f64>();
        }
        mean_w_ed /= z;
        mean_w_sq_ed /= z;
        let sigma_w_sq_ed = mean_w_sq_ed - mean_w_ed * mean_w_ed;

        let stats = tpq_work_statistics_complex(&ed_init, &ed_final, beta, 600, 7);
        assert!(
            (stats.mean_w - mean_w_ed).abs() < 0.05,
            "complex TPQ <W> = {} vs ED {} (stderr {})",
            stats.mean_w,
            mean_w_ed,
            stats.mean_w_stderr
        );
        assert!(
            (stats.work_variance - sigma_w_sq_ed).abs() < 0.1,
            "complex TPQ sigma_W^2 = {} vs ED {}",
            stats.work_variance,
            sigma_w_sq_ed
        );
    }

    #[test]
    fn tpq_work_complex_reduces_variance_versus_real() {
        // Complex Gaussian amplitude should approximately halve the
        // per-sample variance of <W>, i.e. mean_w_stderr_complex /
        // mean_w_stderr_real ~ 1 / sqrt(2) ~ 0.71. We require ratio < 0.9
        // (clear reduction; some seed-dependent slack).
        let v_init = [0.0_f64; 4];
        let v_final = [0.3_f64, -0.3, 0.3, -0.3];
        let ed_init = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_init);
        let ed_final = ed::canonical_thermal(4, 2, 2, 1.0, 4.0, &v_final);
        let n = 800;
        let real = tpq_work_statistics(&ed_init, &ed_final, 2.0, n, 13);
        let cplx = tpq_work_statistics_complex(&ed_init, &ed_final, 2.0, n, 13);
        let ratio = cplx.mean_w_stderr / real.mean_w_stderr;
        assert!(
            ratio < 0.9,
            "complex stderr {} not noticeably below real stderr {} (ratio {})",
            cplx.mean_w_stderr,
            real.mean_w_stderr,
            ratio
        );
    }

    #[test]
    fn tpq_complex_deterministic_with_same_seed() {
        let result = ed::canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let a = tpq_density_complex(&result, 2.0, 20, 99);
        let b = tpq_density_complex(&result, 2.0, 20, 99);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-14, "site {i}: not reproducible");
        }
    }
}
