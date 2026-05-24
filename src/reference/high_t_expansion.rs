#![allow(
    unknown_lints,
    clippy::manual_is_multiple_of,
    clippy::many_single_char_names
)]
#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! High-temperature (small-beta) expansion references for canonical
//! Hubbard.
//!
//! The canonical partition function in the `(N_up, N_dn)` spin sector
//! at L sites has Hilbert dimension `D = C(L, N_up) * C(L, N_dn)`.
//! Expanding `Z(beta) = Tr exp(-beta H)` to second order:
//!
//! ```text
//! Z(beta) = D - beta * Tr H + (beta^2 / 2) * Tr H^2 + O(beta^3)
//! F(beta) = -ln Z / beta = -ln D / beta + <H>_micro
//!           - (beta / 2) * (<H^2>_micro - <H>_micro^2) + O(beta^2)
//! ```
//!
//! The leading `-T ln D` term is universal (depends only on the
//! Hilbert dimension); the next correction `<H>_micro` is what
//! survives at finite T.

/// Binomial coefficient `C(n, k)` as `u128`.
#[must_use]
pub fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc * (n - i) as u128 / (i + 1) as u128;
    }
    acc
}

/// Hilbert-space dimension for the canonical sector `(N_up, N_dn)`
/// at L sites.
#[must_use]
pub fn canonical_hilbert_dim(l: usize, n_up: usize, n_dn: usize) -> u128 {
    binomial(l, n_up) * binomial(l, n_dn)
}

/// Leading high-temperature free-energy term: `F_inf_T = -T ln D`.
/// Universal: depends only on the Hilbert dimension, not on the
/// Hamiltonian.
#[must_use]
pub fn free_energy_leading(beta: f64, l: usize, n_up: usize, n_dn: usize) -> f64 {
    let dim = canonical_hilbert_dim(l, n_up, n_dn);
    assert!(dim > 0, "empty Hilbert sector");
    -((dim as f64).ln()) / beta
}

/// Beta -> 0 limit of the free energy at half filling
/// (N_up = N_dn = L / 2) for an L-site spinful Hubbard chain.
#[must_use]
pub fn free_energy_leading_half_filling(beta: f64, l: usize) -> f64 {
    assert!(l % 2 == 0, "half_filling requires even L, got {l}");
    free_energy_leading(beta, l, l / 2, l / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_l_choose_zero_and_l_is_one() {
        for l in 0..10 {
            assert_eq!(binomial(l, 0), 1);
            assert_eq!(binomial(l, l), 1);
        }
    }

    #[test]
    fn binomial_l4_k2_is_six() {
        assert_eq!(binomial(4, 2), 6);
    }

    #[test]
    fn canonical_hilbert_dim_l4_half_filling_is_36() {
        assert_eq!(canonical_hilbert_dim(4, 2, 2), 36);
    }

    #[test]
    fn canonical_hilbert_dim_l2_half_filling_is_4() {
        assert_eq!(canonical_hilbert_dim(2, 1, 1), 4);
    }

    #[test]
    fn dim_matches_dimer_ed_at_l2() {
        let ed = super::super::ed::canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        assert_eq!(
            canonical_hilbert_dim(2, 1, 1) as usize,
            ed.eigenvalues.len()
        );
    }

    #[test]
    fn leading_free_energy_at_beta_small_matches_minus_t_ln_d() {
        let beta = 1.0e-3;
        let f = free_energy_leading_half_filling(beta, 2);
        let expected = -(4.0_f64).ln() / beta;
        assert!((f - expected).abs() < 1e-12);
    }

    #[test]
    fn leading_free_energy_dominates_full_z_at_small_beta() {
        // For very small beta the true F (from full ED) should agree
        // with the leading -T ln D term to relative error O(beta).
        let beta = 1.0e-3;
        let ed = super::super::ed::canonical_thermal(2, 1, 1, 1.0, 4.0, &[0.0, 0.0]);
        let f_ed = super::super::ed::free_energy(&ed.eigenvalues, beta);
        let f_lead = free_energy_leading_half_filling(beta, 2);
        let rel_err = (f_ed - f_lead).abs() / f_lead.abs();
        assert!(
            rel_err < 1e-2,
            "ED F = {f_ed}, leading = {f_lead}, rel_err = {rel_err}"
        );
    }
}
