#![allow(unknown_lints, clippy::manual_is_multiple_of)]
//! BALDA — Bethe-Ansatz Local Density Approximation (Lima, Silva,
//! Capelle 2003 PRL 90 146402).
//!
//! ## Theory
//!
//! For the 1D homogeneous Hubbard model with hopping `t` and on-site
//! interaction `U`, define `u = U/t`. The Lima 2003 parametrization
//! expresses the per-site ground-state energy as
//!
//! ```text
//! e_BA(n, u) = -(2 β(u) / π) · sin(π n / β(u))            for 0 ≤ n ≤ 1
//! ```
//!
//! with particle-hole symmetric extension for `1 ≤ n ≤ 2`. The parameter
//! `β(u) ∈ [1, 2]` interpolates between non-interacting (`β(0) = 2`)
//! and the Mott insulator (`β(∞) = 1`), and is determined by matching
//! the half-filling exact Lieb-Wu energy
//!
//! ```text
//! e_BA(1, u) = -2 β / π · sin(π / β) = ε_h(u)
//! ε_h(u)     = -4 · ∫_0^∞ J_0(x) J_1(x) / [x (1 + exp(u x / 2))] dx
//! ```
//!
//! The BALDA hxc potential is the derivative of `ε_xc(n, u) = e_BA(n, u)
//! − e_BA(n, 0) − U n² / 4`:
//!
//! ```text
//! V_HXC^BALDA(n, u) = 2 [cos(π n / 2) - cos(π n / β(u))]
//! ```
//!
//! ## Finite-T caveat
//!
//! The functional is `T = 0` BALDA evaluated at the finite-temperature
//! density (the standard "T = 0 xc" approximation for finite-T DFT).
//! A true finite-T BALDA awaits dedicated work.

use crate::config::BaldaParams;

/// BALDA evaluator. Caches `β(u)` since `u` is fixed for a run.
#[derive(Debug, Clone)]
pub struct Balda {
    /// On-site interaction `U` (in units of `t`).
    pub on_site_interaction: f64,
    /// `β(u)` — cached at construction.
    pub beta_u: f64,
    /// Numerical parameters.
    pub params: BaldaParams,
}

impl Balda {
    /// Construct, computing `β(u)` numerically.
    #[must_use]
    pub fn new(on_site_interaction: f64, params: BaldaParams) -> Self {
        assert!(
            params.mott_gap_smoothing_width + params.clamp_eta <= 1.0,
            "BaldaParams: mott_gap_smoothing_width ({}) + clamp_eta ({}) must be <= 1.0; otherwise the smoothing window reaches outside the clamped density domain",
            params.mott_gap_smoothing_width,
            params.clamp_eta,
        );
        let beta_u = if on_site_interaction.abs() < 1.0e-14 {
            2.0
        } else {
            solve_beta(on_site_interaction, &params)
        };
        debug_assert!(
            (1.0..=2.0).contains(&beta_u),
            "BALDA solve_beta returned beta_u outside expected range [1, 2]"
        );
        Self {
            on_site_interaction,
            beta_u,
            params,
        }
    }

    /// Evaluate `V_HXC^BALDA(n_i, u)` for each site density.
    #[must_use]
    pub fn evaluate(&self, density: &[f64]) -> Vec<f64> {
        density.iter().map(|&n| self.evaluate_site(n)).collect()
    }

    fn evaluate_site(&self, n_raw: f64) -> f64 {
        if !n_raw.is_finite() {
            tracing::error!(
                n_raw,
                "non-finite site density in BALDA evaluate_site - returning NaN"
            );
            return f64::NAN;
        }
        let eta = self.params.clamp_eta;
        let n = n_raw.clamp(eta, 2.0 - eta);
        let delta = self.params.mott_gap_smoothing_width;
        if delta > 0.0 && (n - 1.0).abs() < delta {
            // Linear blend across the Mott-gap discontinuity for SCF
            // continuity. Width `delta` is small (default 0.02).
            let v_lo = lower_branch(1.0 - delta, self.beta_u);
            let v_hi = upper_branch(1.0 + delta, self.beta_u) + self.on_site_interaction;
            let t = (n - (1.0 - delta)) / (2.0 * delta);
            v_lo + t * (v_hi - v_lo)
        } else if n <= 1.0 {
            lower_branch(n, self.beta_u)
        } else {
            upper_branch(n, self.beta_u) + self.on_site_interaction
        }
    }
}

fn lower_branch(n: f64, beta_u: f64) -> f64 {
    use std::f64::consts::PI;
    2.0 * ((PI * n / 2.0).cos() - (PI * n / beta_u).cos())
}

fn upper_branch(n: f64, beta_u: f64) -> f64 {
    use std::f64::consts::PI;
    let m = 2.0 - n;
    -2.0 * ((PI * m / 2.0).cos() - (PI * m / beta_u).cos())
}

/// Solve `-2 β / π · sin(π / β) = ε_h(u)` for `β ∈ [1, 2]` by bisection.
fn solve_beta(u: f64, params: &BaldaParams) -> f64 {
    debug_assert!(
        u >= 0.0,
        "BALDA bisection bracket [1, 2] maps to f in [-4/pi, 0]; u < 0 lies outside and would silently saturate at beta = 2"
    );
    let target = lieb_wu_integral(u, params);
    // f(β) = -2 β / π · sin(π / β) is monotone on [1, 2]:
    //   f(1) = -2/π · sin π = 0
    //   f(2) = -4/π · sin(π/2) = -4/π ≈ -1.273
    let mut lo = 1.0_f64;
    let mut hi = 2.0_f64;
    let mut converged = false;
    for _ in 0..params.beta_max_bisect_iter {
        let mid = 0.5 * (lo + hi);
        let f_mid = -2.0 * mid / std::f64::consts::PI * (std::f64::consts::PI / mid).sin();
        if f_mid > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < params.beta_tol {
            converged = true;
            break;
        }
    }
    if !converged {
        tracing::warn!(
            residual = hi - lo,
            iters = params.beta_max_bisect_iter,
            target,
            "solve_beta did not converge within beta_max_bisect_iter; using mid as best estimate",
        );
    }
    0.5 * (lo + hi)
}

/// `ε_h(u) = -4 ∫_0^∞ J_0(x) J_1(x) / [x (1 + exp(u x / 2))] dx`
/// via composite Simpson on `[ε, x_max]`.
fn lieb_wu_integral(u: f64, params: &BaldaParams) -> f64 {
    if u.abs() < 1.0e-14 {
        return -4.0 / std::f64::consts::PI;
    }
    let x_max = (60.0_f64).max(28.0 / u.max(0.05));
    let n = params.lieb_simpson_intervals;
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
    use crate::config::BaldaParams;

    fn balda(u: f64) -> Balda {
        Balda::new(u, BaldaParams::default())
    }

    #[test]
    fn beta_u_zero_is_two() {
        assert!((balda(0.0).beta_u - 2.0).abs() < 1e-12);
    }

    #[test]
    fn beta_decreases_with_u() {
        let b_low = balda(2.0).beta_u;
        let b_mid = balda(4.0).beta_u;
        let b_high = balda(16.0).beta_u;
        assert!(b_low < 2.0, "β(2) = {b_low}");
        assert!(b_mid < b_low, "β(4) = {b_mid} not < β(2) = {b_low}");
        assert!(b_high < b_mid, "β(16) = {b_high} not < β(4) = {b_mid}");
        assert!(b_high > 1.0, "β(16) = {b_high} should still be > 1");
    }

    #[test]
    fn beta_u_large_approaches_one() {
        let b = balda(100.0).beta_u;
        assert!((b - 1.0).abs() < 0.05, "β(100) = {b} should be ≈ 1");
    }

    #[test]
    fn vanishes_at_u_zero() {
        let v = balda(0.0).evaluate(&[0.3, 0.7, 1.0, 1.3, 1.7]);
        for x in v {
            assert!(x.abs() < 1e-12, "expected 0, got {x}");
        }
    }

    #[test]
    fn particle_hole_relation_has_constant_shift_u() {
        // V_HXC(n) + V_HXC(2-n) = U exactly when n stays outside the
        // smoothing window; we measure with smoothing disabled.
        let u = 4.0;
        let params = BaldaParams {
            mott_gap_smoothing_width: 0.0,
            ..BaldaParams::default()
        };
        let b = Balda::new(u, params);
        for n in [0.3_f64, 0.5, 0.8, 1.2, 1.5, 1.7] {
            let sum = b.evaluate_site(n) + b.evaluate_site(2.0 - n);
            assert!((sum - u).abs() < 1e-10, "n={n}: sum-U = {}", sum - u);
        }
    }

    #[test]
    fn mott_gap_at_half_filling_for_finite_u() {
        // The raw discontinuity V_HXC(1+) - V_HXC(1-) is the BALDA Mott
        // gap U + 4 cos(π/β) (positive since cos(π/β) < 0 on β in (1, 2)).
        // Disable smoothing to measure it directly.
        use std::f64::consts::PI;
        let u = 4.0;
        let params = BaldaParams {
            mott_gap_smoothing_width: 0.0,
            ..BaldaParams::default()
        };
        let b = Balda::new(u, params);
        let lower = b.evaluate_site(1.0);
        let upper = b.evaluate_site(1.0 + 1e-8);
        let gap = upper - lower;
        let predicted = (4.0_f64).mul_add((PI / b.beta_u).cos(), u);
        assert!(
            (gap - predicted).abs() < 1e-4,
            "gap measured = {gap}, predicted = {predicted}"
        );
    }

    #[test]
    fn half_filling_lower_branch_is_minus_two_cos_pi_over_beta() {
        // V_HXC(1-) = -2 cos(π/β) — measure with smoothing disabled.
        use std::f64::consts::PI;
        let params = BaldaParams {
            mott_gap_smoothing_width: 0.0,
            ..BaldaParams::default()
        };
        let v0 = Balda::new(0.0, params.clone()).evaluate_site(1.0);
        let b = Balda::new(4.0, params);
        let v4 = b.evaluate_site(1.0);
        assert!(v0.abs() < 1e-12);
        let expected = -2.0 * (PI / b.beta_u).cos();
        assert!(
            (v4 - expected).abs() < 1e-10,
            "v4 = {v4}, expected {expected}"
        );
    }

    #[test]
    fn matches_known_half_filling_energy_at_u_four() {
        let b = balda(4.0);
        let e = -2.0 * b.beta_u / std::f64::consts::PI * (std::f64::consts::PI / b.beta_u).sin();
        // Lieb-Wu (1968): e(1, 4) ≈ -0.5727 in units of t.
        assert!((e - (-0.5727)).abs() < 0.01, "e(1, 4) = {e}");
    }

    #[test]
    fn clamps_edge_densities_finite() {
        let v = balda(4.0).evaluate(&[0.0, 2.0]);
        for x in v {
            assert!(x.is_finite(), "edge density produced non-finite {x}");
        }
    }
}
