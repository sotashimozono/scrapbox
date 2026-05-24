#![allow(
    unknown_lints,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::manual_is_multiple_of,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
//! Atomic limit (J = 0) closed-form references.
//!
//! With zero hopping the lattice decouples into independent sites.
//! Each site is a 4-state local Hilbert space:
//!
//! ```text
//! |0>            energy = 0
//! |up>           energy = V_i
//! |dn>           energy = V_i
//! |up dn>        energy = 2 V_i + U
//! ```
//!
//! In the **grand canonical** ensemble at chemical potential `mu`, the
//! per-site partition function and density are closed forms:
//!
//! ```text
//! z(beta, V, U, mu) = 1 + 2 exp(-beta (V - mu)) + exp(-beta (2V + U - 2 mu))
//! n(beta, V, U, mu) = [ 2 exp(-beta (V - mu))
//!                     + 2 exp(-beta (2V + U - 2 mu)) ] / z
//! ```

/// Per-site grand-canonical partition function for the atomic Hubbard
/// model at site potential `v`, on-site interaction `u`, chemical
/// potential `mu`, inverse temperature `beta`.
#[must_use]
pub fn site_partition_function(beta: f64, v: f64, u: f64, mu: f64) -> f64 {
    let e_single = beta * (v - mu);
    let e_double = beta * (2.0_f64.mul_add(v, u) - 2.0 * mu);
    1.0 + 2.0 * (-e_single).exp() + (-e_double).exp()
}

/// Per-site grand-canonical density (sum over spin) for the atomic
/// Hubbard model.
#[must_use]
pub fn site_density(beta: f64, v: f64, u: f64, mu: f64) -> f64 {
    let e_single = beta * (v - mu);
    let e_double = beta * (2.0_f64.mul_add(v, u) - 2.0 * mu);
    let w_single = 2.0 * (-e_single).exp();
    let w_double = (-e_double).exp();
    let z = 1.0 + w_single + w_double;
    (w_single + 2.0 * w_double) / z
}

/// Per-site grand-canonical double-occupancy `<n_up n_dn>`.
#[must_use]
pub fn site_double_occupancy(beta: f64, v: f64, u: f64, mu: f64) -> f64 {
    let e_single = beta * (v - mu);
    let e_double = beta * (2.0_f64.mul_add(v, u) - 2.0 * mu);
    let w_single = 2.0 * (-e_single).exp();
    let w_double = (-e_double).exp();
    let z = 1.0 + w_single + w_double;
    w_double / z
}

/// Per-site grand-canonical free energy contribution `-T ln z`.
/// For the atomic limit total `F = sum_i (-T ln z_i)`.
#[must_use]
pub fn site_free_energy(beta: f64, v: f64, u: f64, mu: f64) -> f64 {
    -site_partition_function(beta, v, u, mu).ln() / beta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_filling_chemical_potential_is_u_over_two() {
        // Particle-hole symmetric point: mu = U/2, V = 0 -> <n> = 1.
        let n = site_density(2.0, 0.0, 4.0, 4.0 * 0.5);
        assert!((n - 1.0).abs() < 1e-12, "<n>(mu=U/2) = {n}, expected 1");
    }

    #[test]
    fn high_t_limit_distributes_uniformly_at_mu_zero() {
        // beta -> 0: all 4 states equally weighted -> z = 4, <n> = (2*1+2*1)/4 = 1.
        let z = site_partition_function(1.0e-10, 0.5, 4.0, 0.0);
        assert!((z - 4.0).abs() < 1e-8, "got z = {z}");
        let n = site_density(1.0e-10, 0.5, 4.0, 0.0);
        assert!((n - 1.0).abs() < 1e-8, "got <n> = {n}");
    }

    #[test]
    fn empty_band_limit_n_to_zero() {
        // mu -> -infinity: vacuum dominates, <n> -> 0.
        let n = site_density(2.0, 0.0, 4.0, -50.0);
        assert!(n.abs() < 1e-12, "<n>(mu = -50) = {n}");
    }

    #[test]
    fn full_band_limit_n_to_two() {
        // mu -> +infinity: double-occ dominates, <n> -> 2.
        let n = site_density(2.0, 0.0, 4.0, 50.0);
        assert!((n - 2.0).abs() < 1e-12, "<n>(mu = 50) = {n}");
    }

    #[test]
    fn double_occupancy_zero_at_u_infinity_half_filling() {
        // beta = 2, U -> infinity, mu = U/2: double-occ suppressed.
        let u = 100.0;
        let d = site_double_occupancy(2.0, 0.0, u, u * 0.5);
        assert!(d < 1e-30, "double occ should be negligible, got {d}");
    }

    #[test]
    fn density_and_partition_function_satisfy_thermodynamic_identity() {
        // d/d(mu) ln z = beta <n>.
        let beta = 1.5;
        let v = 0.1;
        let u = 3.0;
        let mu = 1.0;
        let eps = 1e-5;
        let dlnz = (site_partition_function(beta, v, u, mu + eps).ln()
            - site_partition_function(beta, v, u, mu - eps).ln())
            / (2.0 * eps);
        let n = site_density(beta, v, u, mu);
        assert!(
            (dlnz - beta * n).abs() < 1e-6,
            "d/dmu ln z = {dlnz}, beta * <n> = {}",
            beta * n
        );
    }
}
