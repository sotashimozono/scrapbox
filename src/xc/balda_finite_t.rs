//! BALDA finite-temperature dispatch with a KST-inspired
//! leading-T² additive correction.
//!
//! v0.15 delta replaces the v0.14 alpha `c * T^2 * n^2` shape with a
//! Karasiev-Sjostrom-Trickey-inspired form that incorporates the
//! Hubbard interaction strength `U / t`:
//!
//! ```text
//! v^{BaldaFiniteT}_i[n; beta, U/t]
//!     = v^{Balda}_i[n] - c * T^2 * (U / t) * g(n_i)
//!     g(n) = n^2 / (1 + n)
//! ```
//!
//! where `T = 1 / beta`. The shape decisions:
//!
//! - `g(n) = n^2 / (1 + n)` is monotonic in `n`, asymmetric around
//!   `n = 1` (so SCF on the canonical dimer sees a real density
//!   shift instead of an absorbed chemical-potential zero-point),
//!   and saturates at `n -> 2` instead of growing as `n^2` — closer
//!   to the bounded reduction factor shape KST 2014 proposes for the
//!   homogeneous electron gas at strong coupling.
//! - The explicit `U / t` factor lets the correction magnitude
//!   reflect interaction strength: stronger `U` widens the Mott gap
//!   that finite-T washes out, so the leading thermal correction
//!   should scale with the interaction scale.
//! - Additive (not multiplicative) so SCF self-consistency is
//!   preserved up to leading order in `c` — same reasoning as v0.14
//!   alpha. The original v0.13 beta multiplicative draft broke
//!   variational structure and failed to converge.
//!
//! Honest status: still a placeholder, not a derived BA-LDA
//! thermal evaluator. KST 2014 (Karasiev, Sjostrom, Trickey,
//! PRB 88, 161108) is for the 3D homogeneous electron gas; we
//! borrow its `U`-aware shape but not its tabulated `F(rs, T)`
//! coefficients. A true thermal BA-LDA needs Lieb-Wu thermal
//! integration over the Bethe-ansatz solution; deferred to v0.16+.
//!
//! `c` retuned to `0.005` so that with `U/t = 4` the magnitude at
//! `n = 1`, `beta = 2` matches v0.14 alpha's `~5e-3` band and the
//! same SCF convergence cross-checks still pass.

use crate::config::BaldaParams;

/// Empirical coefficient for the KST-inspired leading-T² additive
/// correction.
///
/// Tuned so that with `U/t = 4`, `beta = 2.0`, `n = 1`, the per-site
/// correction lands in `[1e-4, 5e-3]` (the band v0.14 alpha
/// established as observable yet SCF-stable). Reduced from v0.14
/// alpha's `0.02` to compensate for the additional `U/t` prefactor.
const KST_T2_COEFFICIENT: f64 = 0.005;

/// Finite-temperature BALDA with a KST-inspired leading-T² additive
/// correction.
///
/// Wraps a T=0 [`crate::xc::balda::Balda`] evaluator and shifts its
/// output by `-c * T^2 * (U/t) * n^2 / (1 + n)` per site.
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
    /// Returns `-c * T^2 * (U/t) * n^2 / (1 + n)`. `T = 1/beta`; for
    /// `beta <= 0` (infinite-T limit) the correction is undefined
    /// and we return `0.0` rather than `NaN`.
    #[must_use]
    pub fn t2_correction(&self, n: f64) -> f64 {
        if self.beta <= 0.0 {
            return 0.0;
        }
        let t2 = 1.0 / (self.beta * self.beta);
        let g = n * n / (1.0 + n);
        -KST_T2_COEFFICIENT * t2 * self.u_over_t * g
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
            "v0.15 delta: correction must be observable at finite beta, got {max_delta}"
        );
        assert!(
            max_delta < 1e-2,
            "v0.15 delta: correction must stay small for SCF stability, got {max_delta}"
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
    fn balda_finite_t_correction_scales_with_u_over_t() {
        let beta = 2.0;
        let n = 0.7_f64;
        let ft_weak = BaldaFiniteT::new(2.0, beta, BaldaParams::default());
        let ft_strong = BaldaFiniteT::new(4.0, beta, BaldaParams::default());
        let corr_weak = ft_weak.t2_correction(n).abs();
        let corr_strong = ft_strong.t2_correction(n).abs();
        let ratio = corr_strong / corr_weak;
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "U/t scaling: doubling U/t should double correction (got ratio {ratio})"
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
        let ft = BaldaFiniteT::new(4.0, 2.0, BaldaParams::default());
        assert!(ft.t2_correction(0.0).abs() < 1e-15);
        let full = ft.t2_correction(2.0).abs();
        assert!(
            full > 1e-4,
            "n=2 full-band correction must be nonzero: {full}"
        );
    }
}
