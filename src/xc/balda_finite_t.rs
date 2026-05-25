//! BALDA finite-temperature dispatch shim.
//!
//! v0.13 beta scope: this wires a `balda_finite_t` route through the
//! XC dispatcher and stores the canonical inverse temperature `beta`
//! taken from `[hamiltonian].beta`, but the evaluator body currently
//! delegates verbatim to the T=0 BALDA functional. A physically
//! derived finite-T BALDA (Lieb-Wu thermal integration over the
//! Bethe-ansatz solution, or a Sommerfeld-style temperature expansion
//! of `e^{BA}`) is deferred — see MILESTONE-v13 deferred items.
//!
//! Why a placeholder rather than a naive `w(beta * U) * v_xc^{T=0}`
//! interpolator: a temperature-scaled BALDA potential is no longer
//! the variational derivative of any free-energy density, so it
//! breaks the SCF self-consistency the BALDA route relies on. The
//! v0.13 beta dispatch lands the route without compromising SCF
//! stability; the next sprint can swap in a real thermal evaluator
//! without touching the dispatch surface.

use crate::config::BaldaParams;

/// Finite-temperature BALDA dispatch shim (v0.13 beta placeholder).
///
/// Wraps a T=0 [`crate::xc::balda::Balda`] evaluator and stores
/// `beta` for use by a future thermal evaluator. Currently
/// [`BaldaFiniteT::evaluate`] returns the T=0 BALDA potential
/// verbatim.
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
    /// inverse temperature taken from `[hamiltonian].beta`, stored
    /// for a future thermal evaluator.
    #[must_use]
    pub fn new(u_over_t: f64, beta: f64, params: BaldaParams) -> Self {
        Self {
            inner: super::balda::Balda::new(u_over_t, params),
            beta,
            u_over_t,
        }
    }

    /// Stored canonical inverse temperature. Diagnostic accessor;
    /// not used by [`Self::evaluate`] in v0.13 beta.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Stored `U / t` ratio. Diagnostic accessor.
    #[must_use]
    pub fn u_over_t(&self) -> f64 {
        self.u_over_t
    }

    /// Evaluate the site-wise `lambda^{h-xc}_i[n]` potential.
    ///
    /// v0.13 beta placeholder: delegates verbatim to T=0 BALDA so SCF
    /// remains stable. A future PR will replace this with a true
    /// thermal evaluator.
    #[must_use]
    pub fn evaluate(&self, density: &[f64]) -> Vec<f64> {
        self.inner.evaluate(density)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BaldaParams;

    #[test]
    fn balda_finite_t_evaluate_matches_zero_t_balda_in_v013() {
        let u = 4.0;
        let beta = 2.0;
        let density = vec![0.3_f64, 0.7, 1.0, 0.5];
        let ft = BaldaFiniteT::new(u, beta, BaldaParams::default());
        let t0 = super::super::balda::Balda::new(u, BaldaParams::default());
        let v_ft = ft.evaluate(&density);
        let v_t0 = t0.evaluate(&density);
        for (a, b) in v_ft.iter().zip(v_t0.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "v0.13 beta: BaldaFiniteT must equal T=0 BALDA exactly, got {a} vs {b}"
            );
        }
    }

    #[test]
    fn balda_finite_t_stores_beta_and_u_over_t() {
        let ft = BaldaFiniteT::new(4.0, 2.5, BaldaParams::default());
        assert!((ft.beta() - 2.5).abs() < 1e-15);
        assert!((ft.u_over_t() - 4.0).abs() < 1e-15);
    }
}
