#![allow(unknown_lints, clippy::many_single_char_names)]
#![allow(
    clippy::needless_borrows_for_generic_args,
    clippy::needless_pass_by_value
)]
#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! Numerical-identity verifiers.
//!
//! These helpers take callback closures (the user's free-energy and
//! density evaluators) and check that they satisfy known
//! thermodynamic identities. Useful for cross-checking either the KS
//! solver, the ED reference, or any in-development functional.
//!
//! ## Identities
//!
//! - **Hellmann-Feynman**: `dF / dV_i = <n_i>`.
//! - **`d(beta F) / dbeta = E`** equivalently `dF/dbeta = (E - F) / beta`.
//! - **Mermin minimum**: `F[n_perturbed] > F[n_SCF]` for any small
//!   perturbation `delta_n` from the self-consistent density.
//!
//! Each helper returns the numerical residual; callers assert it is
//! below their tolerance.

/// Hellmann-Feynman residual at site `site`: `|dF / dV_i - <n_i>|`.
///
/// `free_energy` is a closure mapping `V_ext -> F`, `density` is
/// `V_ext -> [n_0, n_1, ...]` indexed at `site`. Uses central finite
/// differences with step `epsilon`.
#[must_use]
pub fn hellmann_feynman_residual<F1, F2>(
    free_energy: F1,
    density: F2,
    v_ext: &[f64],
    site: usize,
    epsilon: f64,
) -> f64
where
    F1: Fn(&[f64]) -> f64,
    F2: Fn(&[f64]) -> Vec<f64>,
{
    assert!(
        site < v_ext.len(),
        "site index {site} >= v_ext length {}",
        v_ext.len()
    );
    let mut v_plus = v_ext.to_vec();
    v_plus[site] += epsilon;
    let mut v_minus = v_ext.to_vec();
    v_minus[site] -= epsilon;
    let df_dv = (free_energy(&v_plus) - free_energy(&v_minus)) / (2.0 * epsilon);
    let n_i = density(v_ext)[site];
    (df_dv - n_i).abs()
}

/// Vector form of [`hellmann_feynman_residual`] - returns per-site residuals.
#[must_use]
pub fn hellmann_feynman_residual_all_sites<F1, F2>(
    free_energy: &F1,
    density: &F2,
    v_ext: &[f64],
    epsilon: f64,
) -> Vec<f64>
where
    F1: Fn(&[f64]) -> f64,
    F2: Fn(&[f64]) -> Vec<f64>,
{
    (0..v_ext.len())
        .map(|i| hellmann_feynman_residual(free_energy, density, v_ext, i, epsilon))
        .collect()
}

/// Residual of `d(beta F) / d beta - E`, i.e.,
/// `|(beta_+ F(beta_+) - beta_- F(beta_-)) / (2 eps) - E(beta)|`.
#[must_use]
pub fn d_beta_f_minus_energy_residual<FF, FE>(
    free_energy_of_beta: FF,
    energy: FE,
    beta: f64,
    epsilon: f64,
) -> f64
where
    FF: Fn(f64) -> f64,
    FE: Fn(f64) -> f64,
{
    let beta_p = beta + epsilon;
    let beta_m = beta - epsilon;
    let d_betaf = (beta_p * free_energy_of_beta(beta_p) - beta_m * free_energy_of_beta(beta_m))
        / (2.0 * epsilon);
    let e = energy(beta);
    (d_betaf - e).abs()
}

/// Mermin minimum-principle residual: returns `F[n_perturbed] - F[n_SCF]`.
/// Positive value means the identity holds.
#[must_use]
pub fn mermin_minimum_residual<FN>(f_from_density: FN, n_scf: &[f64], delta_n: &[f64]) -> f64
where
    FN: Fn(&[f64]) -> f64,
{
    assert_eq!(
        n_scf.len(),
        delta_n.len(),
        "n_scf and delta_n length mismatch"
    );
    let n_pert: Vec<f64> = n_scf
        .iter()
        .zip(delta_n.iter())
        .map(|(&n, &d)| n + d)
        .collect();
    f_from_density(&n_pert) - f_from_density(n_scf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ed;

    fn ed_free_energy_at_v(v_ext: &[f64], beta: f64, j: f64, u: f64) -> f64 {
        let result = ed::canonical_thermal(v_ext.len(), 1, 1, j, u, v_ext);
        ed::free_energy(&result.eigenvalues, beta)
    }

    fn ed_density_at_v(v_ext: &[f64], beta: f64, j: f64, u: f64) -> Vec<f64> {
        let result = ed::canonical_thermal(v_ext.len(), 1, 1, j, u, v_ext);
        ed::thermal_density(&result, beta)
    }

    #[test]
    fn hellmann_feynman_holds_for_ed_dimer_at_uniform_v() {
        let beta = 2.0;
        let j = 1.0;
        let u = 4.0;
        let v_ext = [0.0, 0.0];
        let f = |v: &[f64]| ed_free_energy_at_v(v, beta, j, u);
        let n = |v: &[f64]| ed_density_at_v(v, beta, j, u);
        for site in 0..2 {
            let r = hellmann_feynman_residual(&f, &n, &v_ext, site, 1e-4);
            assert!(r < 1e-6, "site {site}: HF residual = {r}");
        }
    }

    #[test]
    fn hellmann_feynman_holds_for_ed_dimer_with_comb_v() {
        let beta = 2.0;
        let j = 1.0;
        let u = 4.0;
        let v_ext = [0.3_f64, -0.3];
        let f = |v: &[f64]| ed_free_energy_at_v(v, beta, j, u);
        let n = |v: &[f64]| ed_density_at_v(v, beta, j, u);
        for site in 0..2 {
            let r = hellmann_feynman_residual(&f, &n, &v_ext, site, 1e-4);
            assert!(r < 1e-6, "site {site}: HF residual = {r}");
        }
    }

    #[test]
    fn d_beta_f_equals_energy_for_ed_dimer() {
        let j = 1.0;
        let u = 4.0;
        let v_ext = [0.0, 0.0];
        let f_of_b = |b: f64| ed_free_energy_at_v(&v_ext, b, j, u);
        let energy_of_b = |b: f64| {
            let res = ed::canonical_thermal(2, 1, 1, j, u, &v_ext);
            let shift = res.eigenvalues[0];
            let z_shifted: f64 = res
                .eigenvalues
                .iter()
                .map(|e| (-b * (e - shift)).exp())
                .sum();
            let e_weighted: f64 = res
                .eigenvalues
                .iter()
                .map(|e| e * (-b * (e - shift)).exp())
                .sum();
            e_weighted / z_shifted
        };
        for beta in [0.5_f64, 1.0, 2.0] {
            let r = d_beta_f_minus_energy_residual(&f_of_b, &energy_of_b, beta, 1e-4);
            assert!(r < 1e-6, "beta = {beta}: |d(beta F)/dbeta - E| = {r}");
        }
    }
}
