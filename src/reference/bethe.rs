#![allow(
    unknown_lints,
    clippy::many_single_char_names,
    clippy::manual_is_multiple_of
)]
#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! Bethe-ansatz analytic / integral references for the 1D Hubbard model.
//!
//! Currently implements the Lieb-Wu half-filling ground-state energy
//! (Lieb, Wu, PRL 20, 1445, 1968):
//!
//! ```text
//! e_h(u) = -4 * int_0^inf [ J_0(x) J_1(x) / (x (1 + exp(u x / 2))) ] dx
//! ```
//!
//! at U / t = u, expressed per site for the half-filled chain. The
//! formula is exact at the thermodynamic limit; finite-L corrections
//! are 1/L^2-suppressed.

use std::f64::consts::PI;

/// Numerical parameters for the Lieb-Wu Bessel integral.
#[derive(Debug, Clone, Copy)]
pub struct LiebWuParams {
    /// Composite Simpson interval count (rounded up to even internally).
    pub simpson_intervals: usize,
}

impl Default for LiebWuParams {
    fn default() -> Self {
        Self {
            simpson_intervals: 4096,
        }
    }
}

/// Lieb-Wu half-filling ground-state energy per site for the 1D
/// Hubbard model at `u = U / t`. Uses composite Simpson on
/// `[0, x_max]` with `x_max = max(60, 28 / max(u, 0.05))` to cover the
/// `exp(-u x / 2)` decay scale.
///
/// At `u = 0` the closed form `e(0) = -4 / pi` is returned directly.
#[must_use]
pub fn lieb_wu_half_filling_energy(u: f64) -> f64 {
    lieb_wu_half_filling_energy_with_params(u, &LiebWuParams::default())
}

/// Same as [`lieb_wu_half_filling_energy`] with explicit numerical
/// parameters, for convergence studies.
#[must_use]
pub fn lieb_wu_half_filling_energy_with_params(u: f64, params: &LiebWuParams) -> f64 {
    if u.abs() < 1.0e-14 {
        return -4.0 / PI;
    }
    let x_max = 60.0_f64.max(28.0 / u.max(0.05));
    let n = params.simpson_intervals;
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = x_max / (n as f64);

    let mut acc = 0.0_f64;
    for i in 0..=n {
        let x = (i as f64).mul_add(h, 1.0e-10);
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        let j0 = libm::j0(x);
        let j1 = libm::j1(x);
        let denom = x * (1.0 + (u * x / 2.0).exp());
        acc += weight * j0 * j1 / denom;
    }
    -4.0 * acc * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u_zero_matches_closed_form_minus_four_over_pi() {
        let e = lieb_wu_half_filling_energy(0.0);
        assert!((e - (-4.0 / PI)).abs() < 1e-12, "got {e}");
    }

    #[test]
    fn u_four_matches_lieb_wu_1968_value() {
        let e = lieb_wu_half_filling_energy(4.0);
        assert!(
            (e - (-0.5727)).abs() < 0.005,
            "e(u=4) = {e}, expected ~ -0.5727"
        );
    }

    #[test]
    fn monotone_decreasing_in_absolute_value_with_u() {
        let e0 = lieb_wu_half_filling_energy(0.0).abs();
        let e2 = lieb_wu_half_filling_energy(2.0).abs();
        let e4 = lieb_wu_half_filling_energy(4.0).abs();
        let e8 = lieb_wu_half_filling_energy(8.0).abs();
        assert!(e0 > e2, "|e(0)| = {e0} should exceed |e(2)| = {e2}");
        assert!(e2 > e4);
        assert!(e4 > e8);
    }

    #[test]
    fn u_infinity_limit_approaches_zero_from_below() {
        // e(u=50) should satisfy the 1/u asymptotic: e ~ -(4 ln 2)/50 ~ -0.055.
        let e = lieb_wu_half_filling_energy(50.0);
        let asymptotic = -4.0 * (2.0_f64).ln() / 50.0;
        assert!(
            (e - asymptotic).abs() < 0.005,
            "e(u=50) = {e}, asymptotic = {asymptotic}"
        );
        assert!(e < 0.0, "e_h(u) is negative for all finite u");
    }

    #[test]
    fn simpson_resolution_converges() {
        let e_default = lieb_wu_half_filling_energy(4.0);
        let e_fine = lieb_wu_half_filling_energy_with_params(
            4.0,
            &LiebWuParams {
                simpson_intervals: 8192,
            },
        );
        assert!(
            (e_default - e_fine).abs() < 1e-5,
            "default = {e_default}, fine = {e_fine}"
        );
    }
}
