#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! Heisenberg (large-U Hubbard) effective-model references.
//!
//! At half filling and `U >> t`, the 1D Hubbard model maps onto the
//! antiferromagnetic Heisenberg chain
//!
//! ```text
//! H_H = J_H * sum_i S_i . S_{i+1}     with J_H = 4 t^2 / U.
//! ```
//!
//! The Bethe-ansatz exact ground-state energy per site is
//! (Hulthen 1938 / Bethe):
//!
//! ```text
//! e_AFM / J_H = -ln(2) + 1/4 = -0.443147...
//! ```
//!
//! This gives an asymptotic per-site cross-check for the half-filled
//! Hubbard at large U:
//!
//! ```text
//! e_Hubbard(u -> infinity) ~ -(4/u) (ln(2) - 1/4) t   per site.
//! ```

/// Per-bond AFM Heisenberg superexchange coupling at half-filled
/// large-U Hubbard: `J_H = 4 t^2 / U`.
#[must_use]
pub fn superexchange_coupling(t: f64, u: f64) -> f64 {
    assert!(u > 0.0, "superexchange_coupling requires U > 0");
    4.0 * t * t / u
}

/// Bethe-ansatz exact ground-state energy per site for the
/// nearest-neighbor AFM Heisenberg chain with coupling `j_h`:
/// `e = J_H * (-ln(2) + 1/4)`.
#[must_use]
pub fn heisenberg_ground_state_energy_per_site(j_h: f64) -> f64 {
    j_h * (-(2.0_f64).ln() + 0.25)
}

/// Asymptotic Hubbard half-filling per-site energy at large `u = U / t`:
/// `e ~ -(4 ln 2) / u` (in units of t). This is the strict 1/u
/// leading order of the Lieb-Wu integral; the per-bond Heisenberg
/// `+1/4 J_H` constant in Hulthen is a basis-dependent offset that
/// does NOT enter the standard Hubbard energy convention.
#[must_use]
pub fn hubbard_large_u_per_site_energy(u: f64) -> f64 {
    assert!(u > 0.0, "hubbard_large_u_per_site_energy requires u > 0");
    -4.0 * (2.0_f64).ln() / u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superexchange_at_u_four_is_one() {
        assert!((superexchange_coupling(1.0, 4.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn heisenberg_per_site_matches_bethe_constant() {
        let e = heisenberg_ground_state_energy_per_site(1.0);
        let expected = -((2.0_f64).ln()) + 0.25;
        assert!((e - expected).abs() < 1e-14);
        assert!((e - (-0.443_147_180_559_945_3)).abs() < 1e-12, "got {e}");
    }

    #[test]
    fn large_u_hubbard_per_site_matches_minus_four_ln_two_over_u() {
        // Strict 1/u leading order: e = -(4 ln 2) / u.
        let u = 16.0;
        let e = hubbard_large_u_per_site_energy(u);
        let expected = -4.0 * (2.0_f64).ln() / u;
        assert!((e - expected).abs() < 1e-14, "got {e}");
    }

    #[test]
    fn large_u_asymptotic_matches_lieb_wu_within_one_percent_at_u_twenty() {
        // The -(4 ln 2)/u asymptotic should agree with Lieb-Wu at u = 20
        // to within ~1% (next-order correction is O(1/u^3)).
        let lw = super::super::bethe::lieb_wu_half_filling_energy(20.0);
        let approx = hubbard_large_u_per_site_energy(20.0);
        let rel_err = (lw - approx).abs() / lw.abs();
        assert!(
            rel_err < 0.02,
            "Lieb-Wu = {lw}, asymptotic = {approx}, rel_err = {rel_err}"
        );
    }
}
