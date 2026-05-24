#![allow(clippy::too_long_first_doc_paragraph)]
#![allow(
    unknown_lints,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::manual_is_multiple_of,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
//! Analytic Hubbard dimer (L = 2) at half-filling, V_ext = 0.
//!
//! Basis (4 states in the `(N_up = 1, N_dn = 1)` sector):
//!
//! ```text
//! |a> = c+_{0u} c+_{1d} |0>     (up at 0, dn at 1)
//! |b> = c+_{1u} c+_{0d} |0>     (up at 1, dn at 0)
//! |c> = c+_{0u} c+_{0d} |0>     (both at 0)
//! |d> = c+_{1u} c+_{1d} |0>     (both at 1)
//! ```
//!
//! Closed-form spectrum (4 eigenvalues, ascending):
//!
//! ```text
//! E_singlet_low  = (U - sqrt(U^2 + 16 J^2)) / 2
//! E_triplet      = 0                                    (S_z = 0 from |a> - |b>)
//! E_excited      = U                                    (antisymmetric double-occ)
//! E_singlet_high = (U + sqrt(U^2 + 16 J^2)) / 2
//! ```

use std::f64::consts::PI;

/// Ground-state energy of the Hubbard dimer at half-filling V_ext = 0.
/// Lower singlet branch: `E_GS = (U - sqrt(U^2 + 16 J^2)) / 2`.
#[must_use]
pub fn ground_state_energy(hopping_j: f64, on_site_u: f64) -> f64 {
    0.5 * (on_site_u
        - on_site_u
            .mul_add(on_site_u, 16.0 * hopping_j * hopping_j)
            .sqrt())
}

/// Full 4-eigenvalue spectrum of the half-filled Hubbard dimer at V_ext = 0,
/// returned in ascending order.
#[must_use]
pub fn spectrum(hopping_j: f64, on_site_u: f64) -> [f64; 4] {
    let disc = on_site_u
        .mul_add(on_site_u, 16.0 * hopping_j * hopping_j)
        .sqrt();
    let e_low = 0.5 * (on_site_u - disc);
    let e_high = 0.5 * (on_site_u + disc);
    [e_low, 0.0, on_site_u, e_high]
}

/// Canonical thermal partition function at half-filling V_ext = 0:
/// `Z = sum_k exp(-beta * E_k)`.
#[must_use]
pub fn partition_function(beta: f64, hopping_j: f64, on_site_u: f64) -> f64 {
    spectrum(hopping_j, on_site_u)
        .iter()
        .map(|e| (-beta * e).exp())
        .sum()
}

/// Free energy `F = -ln Z / beta` at half-filling V_ext = 0.
#[must_use]
pub fn free_energy(beta: f64, hopping_j: f64, on_site_u: f64) -> f64 {
    -partition_function(beta, hopping_j, on_site_u).ln() / beta
}

/// Thermal density at half-filling V_ext = 0. Forced to `[1, 1]` by
/// particle-hole + site-exchange symmetry of the Hamiltonian.
#[must_use]
pub fn thermal_density_half_filling(_beta: f64, _hopping_j: f64, _on_site_u: f64) -> [f64; 2] {
    [1.0, 1.0]
}

/// Thermodynamic-limit per-site GS energy of the non-interacting (U = 0)
/// 1D chain at filling `n in [0, 2]`: `e(n) = -(4t/pi) sin(pi*n/2)`.
/// Convenience for dimer cross-checks at U = 0 against the L = 2 formula.
#[must_use]
pub fn non_interacting_thermodynamic_per_site_energy(n: f64, t: f64) -> f64 {
    -(4.0 * t / PI) * (PI * n * 0.5).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_state_energy_u_zero_matches_minus_two_j() {
        // U = 0, J = 1: E_GS = (0 - sqrt(16)) / 2 = -2.
        assert!((ground_state_energy(1.0, 0.0) - (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn ground_state_energy_u_four_j_one_matches_closed_form() {
        // U = 4, J = 1: E_GS = (4 - sqrt(16 + 16)) / 2 = 2 - 2 sqrt(2) ~= -0.828.
        let e = ground_state_energy(1.0, 4.0);
        let expected = 2.0 - 2.0 * (2.0_f64).sqrt();
        assert!((e - expected).abs() < 1e-14, "got {e}, expected {expected}");
    }

    #[test]
    fn spectrum_trace_equals_2u_by_invariance() {
        // Trace(H) = U + U = 2U (only the doubly-occupied diagonal contributes).
        let s = spectrum(1.0, 4.0);
        assert!((s.iter().sum::<f64>() - 2.0 * 4.0).abs() < 1e-12);
    }

    #[test]
    fn partition_function_high_t_limit_is_state_count() {
        // beta -> 0: Z -> 4 (number of states).
        let z = partition_function(1.0e-10, 1.0, 4.0);
        assert!((z - 4.0).abs() < 1e-8);
    }

    #[test]
    fn partition_function_low_t_picks_ground_state() {
        // beta -> infinity: Z -> exp(-beta * E_GS).
        let beta = 50.0;
        let z = partition_function(beta, 1.0, 4.0);
        let e_gs = ground_state_energy(1.0, 4.0);
        let z_naive = (-beta * e_gs).exp();
        let ratio = z / z_naive;
        assert!(
            (ratio - 1.0).abs() < 1e-6,
            "Z = {z} not dominated by E_GS = {e_gs}: ratio {ratio}"
        );
    }

    #[test]
    fn free_energy_approaches_ground_state_at_low_t() {
        // beta -> infinity: F -> E_GS.
        // beta -> 0 (high T): F = -T ln Z ~ -T ln N_states, which is much more
        // negative than E_GS. So the inequality direction is F_highT < F_lowT.
        let f_low_t = free_energy(50.0, 1.0, 4.0);
        let f_high_t = free_energy(0.5, 1.0, 4.0);
        let e_gs = ground_state_energy(1.0, 4.0);
        assert!(
            f_high_t < f_low_t,
            "F_highT = {f_high_t}, F_lowT = {f_low_t} - high T should give more negative F"
        );
        assert!(
            (f_low_t - e_gs).abs() < 0.1,
            "F(beta=50) = {f_low_t} should approach E_GS = {e_gs}"
        );
    }

    #[test]
    fn non_interacting_per_site_energy_half_filling() {
        // n = 1: e(1) = -(4/pi) * sin(pi/2) = -4/pi.
        let e = non_interacting_thermodynamic_per_site_energy(1.0, 1.0);
        assert!((e - (-4.0 / PI)).abs() < 1e-14);
    }
}
