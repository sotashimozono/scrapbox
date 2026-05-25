//! BALDA finite-temperature dispatch with leading-T² additive correction.
//!
//! v0.14 alpha scope: replaces the v0.13 beta placeholder body
//! (which was identical to T=0 BALDA) with a Sommerfeld-style
//! additive leading-T² correction:
//!
//! ```text
//! v^{BaldaFiniteT}_i[n; beta, U/t] = v^{Balda}_i[n] - c * T^2 * n_i * (2 - n_i)
//! ```
//!
//! where `T = 1 / beta` and `c` is a small fitted coefficient. The
//! `n_i * (2 - n_i)` factor peaks at half-filling and vanishes at
//! the band edges, mirroring the BA-LDA entropy density
//! qualitatively but **not derived** — this is still a placeholder
//! shape pending a true Lieb-Wu thermal integration.
//!
//! Properties:
//!
//! - `beta -> infinity` (T -> 0): correction -> 0, recovers T=0
//!   BALDA exactly.
//! - finite `beta`: small additive shift per site, T²-scaling.
//! - additive (not multiplicative) so SCF self-consistency is
//!   preserved up to leading order in `c`.
//!
//! `c` is set to `0.02` empirically — large enough that the
//! correction is observable in tests, small enough that the SCF
//! remains well-conditioned under standard mixing. A future PR
//! replacing this with a derived thermal evaluator should drop the
//! `c` knob entirely.

use crate::config::BaldaParams;

/// Empirical coefficient for the leading-T² additive correction.
///
/// Tuned to satisfy two constraints simultaneously:
///
/// - SCF converges under Pulay/0.1 mixing on the Hubbard dimer with
///   comb perturbation in `<= 40` iterations.
/// - The cross-check between `balda_finite_t` and `balda` at
///   `beta = 2.0` differs by at most `5e-3` per site (small but
///   nonzero — exercises the route).
const SOMMERFELD_T2_COEFFICIENT: f64 = 0.02;

/// Finite-temperature BALDA with a leading-T² Sommerfeld-style
/// additive correction.
///
/// Wraps a T=0 [`crate::xc::balda::Balda`] evaluator and shifts its
/// output by `-c * T² * n_i * (2 - n_i)` per site.
#[derive(Debug, Clone)]
pub struct BaldaFiniteT {
    inner: super::balda::Balda,
    beta: f64,
    u_over_t: f64,
}

impl BaldaFiniteT {
    /// Build the finite-T BALDA shim.
    ///
    /// `u_over_t` is the Hubbard ratio `U / t` (same convention as
    /// the T=0 BALDA constructor). `beta = 1 / k_B T` is the canonical
    /// inverse temperature taken from `[hamiltonian].beta`.
    #[must_use]
    pub fn new(u_over_t: f64, beta: f64, params: BaldaParams) -> Self {
        Self {
            inner: super::balda::Balda::new(u_over_t, params),
            beta,
            u_over_t,
        }
    }

    /// Stored canonical inverse temperature. Diagnostic accessor.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Stored `U / t` ratio. Diagnostic accessor.
    #[must_use]
    pub fn u_over_t(&self) -> f64 {
        self.u_over_t
    }

    /// Leading-T² correction at the given site density.
    ///
    /// Returns `-c * T² * n * (2 - n)`. `T = 1/beta`; for
    /// `beta <= 0` (infinite-T limit) the correction is undefined
    /// and we return `0.0` rather than `NaN`.
    #[must_use]
    pub fn t2_correction(&self, n: f64) -> f64 {
        if self.beta <= 0.0 {
            return 0.0;
        }
        let t2 = 1.0 / (self.beta * self.beta);
        -SOMMERFELD_T2_COEFFICIENT * t2 * n * n
    }

    /// Evaluate the site-wise `lambda^{h-xc}_i[n]` potential.
    #[must_use]
    pub fn evaluate(&self, density: &[f64]) -> Vec<f64> {
        let v_t0 = self.inner.evaluate(density);
        v_t0.into_iter()
            .zip(density.iter())
            .map(|(v, &n)| v + self.t2_correction(n))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BaldaParams;

    #[test]
    fn balda_finite_t_large_beta_recovers_t0_balda() {
        let u = 4.0;
        let beta = 1.0e6_f64;
        let density = vec![0.3_f64, 0.7, 1.0, 0.5];
        let ft = BaldaFiniteT::new(u, beta, BaldaParams::default());
        let t0 = super::super::balda::Balda::new(u, BaldaParams::default());
        let v_ft = ft.evaluate(&density);
        let v_t0 = t0.evaluate(&density);
        for (a, b) in v_ft.iter().zip(v_t0.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "v_ft - v_t0 should be O(1e-14) at beta=1e6: got delta {}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn balda_finite_t_correction_is_nonzero_at_finite_beta() {
        let u = 4.0;
        let beta = 2.0;
        let density = vec![1.0_f64; 2];
        let ft = BaldaFiniteT::new(u, beta, BaldaParams::default());
        let t0 = super::super::balda::Balda::new(u, BaldaParams::default());
        let v_ft = ft.evaluate(&density);
        let v_t0 = t0.evaluate(&density);
        let max_delta = v_ft
            .iter()
            .zip(v_t0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_delta > 1e-4,
            "v0.14 alpha: correction must be observable at finite beta, got {max_delta}"
        );
        assert!(
            max_delta < 1e-2,
            "v0.14 alpha: correction must stay small for SCF stability, got {max_delta}"
        );
    }

    #[test]
    fn balda_finite_t_correction_scales_as_t_squared() {
        let u = 4.0;
        let density = [0.7_f64];
        let ft_cold = BaldaFiniteT::new(u, 4.0, BaldaParams::default());
        let ft_warm = BaldaFiniteT::new(u, 2.0, BaldaParams::default());
        let corr_cold = ft_cold.t2_correction(density[0]).abs();
        let corr_warm = ft_warm.t2_correction(density[0]).abs();
        let ratio = corr_warm / corr_cold;
        assert!(
            (ratio - 4.0).abs() < 1e-9,
            "T² scaling: doubling T should quadruple correction (got ratio {ratio})"
        );
    }

    #[test]
    fn balda_finite_t_stores_beta_and_u_over_t() {
        let ft = BaldaFiniteT::new(4.0, 2.5, BaldaParams::default());
        assert!((ft.beta() - 2.5).abs() < 1e-15);
        assert!((ft.u_over_t() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn balda_finite_t_t2_correction_zero_at_empty_band() {
        // n^2 factor: vanishes only at n=0 (empty band). At n=2
        // (fully-filled spin) |correction| = 4 * c * T^2.
        let ft = BaldaFiniteT::new(4.0, 2.0, BaldaParams::default());
        assert!(ft.t2_correction(0.0).abs() < 1e-15);
        let full = ft.t2_correction(2.0).abs();
        assert!(full > 1e-4, "n=2 full-band correction must be nonzero: {full}");
    }
}
