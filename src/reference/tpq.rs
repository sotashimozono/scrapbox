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
//! a random vector `|psi_0> = sum_k c_k |k>` with `c_k ~ N(0, 1) +
//! i N(0, 1)`, the canonical TPQ state is
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

    for x in &mut acc_density {
        *x /= acc_norm_sq;
    }
    acc_density
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
}
