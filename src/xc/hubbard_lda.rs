//! Single-site analytical Hubbard LDA functional.
//!
//! From `notes/discipline/canonical_thermal_dft.md` Sec V:
//!
//! ```text
//! V_i^HXC[n_i]  =  U/2  +  (1/β) · ln Γ_U^β[n_i]
//! Γ_U^β[n_i]    =  ( (n_i - 1) + sqrt((n_i - 1)^2 + exp(-βU) · (2n_i - n_i^2)) ) / n_i
//! ```
//!
//! ## Numerical stability
//!
//! The direct formula suffers catastrophic cancellation in the numerator
//! `(n−1) + sqrt(A)` when `n < 1` (the two terms have opposite sign and
//! nearly equal magnitudes). We branch on the sign of `n−1`:
//!
//! - `n ≤ 1`: rationalize numerator-by-conjugate, giving
//!   `Γ = exp(−βU)·(2−n) / (sqrt(A) + (1−n))`.
//! - `n > 1`: the direct form is numerically stable because both `(n−1)`
//!   and `sqrt(A)` are positive.
//!
//! Density is clamped to `[η, 2−η]` to keep `Γ` strictly positive.

use crate::config::HubbardLdaParams;

/// Single-site analytical Hubbard LDA.
#[derive(Debug, Clone)]
pub struct HubbardLda {
    /// On-site interaction strength `U`.
    pub on_site_interaction: f64,
    /// Inverse temperature `β`.
    pub beta: f64,
    /// Numerical parameters.
    pub params: HubbardLdaParams,
}

impl HubbardLda {
    /// Create a new Hubbard LDA evaluator.
    #[must_use]
    pub fn new(on_site_interaction: f64, beta: f64, params: HubbardLdaParams) -> Self {
        Self {
            on_site_interaction,
            beta,
            params,
        }
    }

    /// Evaluate `V_i^HXC` for each site density `n_i` in `density`.
    #[must_use]
    pub fn evaluate(&self, density: &[f64]) -> Vec<f64> {
        density.iter().map(|&n| self.evaluate_site(n)).collect()
    }

    fn evaluate_site(&self, n_raw: f64) -> f64 {
        let eta = self.params.clamp_eta;
        let n = n_raw.clamp(eta, 2.0 - eta);
        let u = self.on_site_interaction;
        let beta = self.beta;
        let exp_neg_bu = (-beta * u).exp();
        let one_minus_n = 1.0 - n;
        let delta = exp_neg_bu * n * (2.0 - n);
        let radicand = one_minus_n.mul_add(one_minus_n, delta);
        let sqrt_a = radicand.sqrt();
        let gamma = if n <= 1.0 {
            // Rationalized form (no cancellation): Γ = δ / (n·(√A + 1−n))
            // simplified using δ = exp(−βU)·n·(2−n).
            exp_neg_bu * (2.0 - n) / (sqrt_a + one_minus_n)
        } else {
            // Direct form is stable when (n−1) and sqrt(A) are both positive.
            ((n - 1.0) + sqrt_a) / n
        };
        u.mul_add(0.5, gamma.ln() / beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lda(u: f64, beta: f64) -> HubbardLda {
        HubbardLda::new(u, beta, HubbardLdaParams::default())
    }

    #[test]
    fn vanishes_at_u_zero() {
        // At U=0, V^HXC should be 0 for any density.
        let v = lda(0.0, 2.0).evaluate(&[0.3, 0.7, 1.0, 1.3, 1.7]);
        for x in v {
            assert!(x.abs() < 1e-10, "expected 0, got {x}");
        }
    }

    #[test]
    fn antisymmetric_around_half_filling() {
        // Particle-hole symmetry of the single-site map:
        //   V^HXC[n] + V^HXC[2-n] = 0
        // (derivation: Γ[n] · Γ[2−n] = exp(−βU) ⇒ ln Γ[n] + ln Γ[2−n] = −βU,
        //  which cancels the leading U/2 terms.)
        let l = lda(4.0, 2.0);
        for n in [0.3_f64, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7] {
            let sum = l.evaluate_site(n) + l.evaluate_site(2.0 - n);
            assert!(
                sum.abs() < 1e-10,
                "particle-hole identity broke at n={n}: sum={sum}"
            );
        }
    }

    #[test]
    fn half_filling_vanishes() {
        let l = lda(4.0, 2.0);
        assert!(l.evaluate_site(1.0).abs() < 1e-12);
    }

    #[test]
    fn clamps_edge_densities_finite() {
        // At n=0 or n=2 (clamped to η or 2−η), the rationalized + direct
        // branches must keep V^HXC finite.
        let v = lda(4.0, 2.0).evaluate(&[0.0, 2.0]);
        for x in v {
            assert!(x.is_finite(), "edge density produced non-finite {x}");
        }
    }

    #[test]
    fn edge_low_density_limit() {
        // For n→0, Γ → exp(−βU), so V^HXC → U/2 + (−βU)/β = −U/2.
        let u = 4.0;
        let beta = 2.0;
        let v = lda(u, beta).evaluate_site(1e-10);
        assert!(
            (v - (-u / 2.0)).abs() < 1e-6,
            "expected V^HXC → -U/2 at n→0, got {v}"
        );
    }

    #[test]
    fn edge_high_density_limit() {
        // By particle-hole symmetry, V^HXC → +U/2 at n→2.
        let u = 4.0;
        let beta = 2.0;
        let v = lda(u, beta).evaluate_site(2.0 - 1e-10);
        assert!(
            (v - (u / 2.0)).abs() < 1e-6,
            "expected V^HXC → +U/2 at n→2, got {v}"
        );
    }
}
