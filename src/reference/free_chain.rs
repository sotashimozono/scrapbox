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
//! Non-interacting (U = 0) 1D tight-binding chain references.
//!
//! Single-particle Hamiltonian on `L` sites with hopping `t`:
//!
//! - **OBC** (open boundary): `eps_k = -2t cos(k*pi/(L+1))`, k = 1..=L.
//! - **PBC** (periodic): `eps_k = -2t cos(2*pi*k/L)`, k = 0..L.
//!
//! Canonical thermal observables are computed by the Pratt-Borrmann-
//! Franke recursion on the supplied single-particle spectrum.

use std::f64::consts::PI;

/// Single-particle spectrum for an L-site open-boundary chain with hopping `t`.
/// Returned ascending. Length `l`.
#[must_use]
pub fn single_particle_spectrum_obc(l: usize, t: f64) -> Vec<f64> {
    let denom = (l + 1) as f64;
    let mut spec: Vec<f64> = (1..=l)
        .map(|k| -2.0 * t * (PI * (k as f64) / denom).cos())
        .collect();
    spec.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN spectrum"));
    spec
}

/// Single-particle spectrum for an L-site periodic-boundary chain.
/// Returned ascending. Length `l`.
#[must_use]
pub fn single_particle_spectrum_pbc(l: usize, t: f64) -> Vec<f64> {
    let mut spec: Vec<f64> = (0..l)
        .map(|k| -2.0 * t * (2.0 * PI * (k as f64) / (l as f64)).cos())
        .collect();
    spec.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN spectrum"));
    spec
}

/// Ground-state energy for L-site OBC chain at `n_per_spin` electrons per
/// spin: `E_GS = 2 * sum_{k=1..=n_per_spin} eps_k` (factor 2 = spin).
#[must_use]
pub fn ground_state_energy_obc(l: usize, n_per_spin: usize, t: f64) -> f64 {
    let spec = single_particle_spectrum_obc(l, t);
    2.0 * spec.iter().take(n_per_spin).sum::<f64>()
}

/// Thermodynamic-limit ground-state energy per site for the
/// non-interacting chain at filling `n` (total, spin-summed):
/// `e(n) = -(4t/pi) sin(pi*n/2)`.
#[must_use]
pub fn ground_state_energy_per_site_thermodynamic(n: f64, t: f64) -> f64 {
    -(4.0 * t / PI) * (PI * n * 0.5).sin()
}

///  via Pratt recursion on the supplied single-particle
/// spectrum. Computed in log-space to survive large `beta` without
/// overflow: `Z_per_spin = exp(ln Z_shifted - n_sigma * beta * shift)`.
#[must_use]
pub fn log_canonical_partition_function_per_spin(
    spectrum: &[f64],
    n_per_spin: usize,
    beta: f64,
) -> f64 {
    let n_sigma = n_per_spin;
    let shift = spectrum.first().copied().unwrap_or(0.0);

    let mut z1 = vec![0.0_f64; n_sigma + 1];
    for m in 1..=n_sigma {
        let m_f = m as f64;
        z1[m] = spectrum
            .iter()
            .map(|&eps| (-m_f * beta * (eps - shift)).exp())
            .sum();
    }
    let mut z_canon = vec![0.0_f64; n_sigma + 1];
    z_canon[0] = 1.0;
    for k in 1..=n_sigma {
        let mut acc = 0.0_f64;
        for m in 1..=k {
            let sign = if (m - 1) % 2 == 0 { 1.0 } else { -1.0 };
            acc += sign * z1[m] * z_canon[k - m];
        }
        z_canon[k] = acc / (k as f64);
    }
    z_canon[n_sigma].ln() - (n_sigma as f64) * beta * shift
}

/// Canonical partition function per spin sector. Direct value (may
/// overflow at large `beta`; prefer the log-space getter when only
/// the free energy is needed).
#[must_use]
pub fn canonical_partition_function_per_spin(
    spectrum: &[f64],
    n_per_spin: usize,
    beta: f64,
) -> f64 {
    log_canonical_partition_function_per_spin(spectrum, n_per_spin, beta).exp()
}

/// Total canonical Z for the spinful non-interacting chain. May
/// overflow; use [`free_energy`] when at large `beta`.
#[must_use]
pub fn canonical_partition_function(spectrum: &[f64], n_per_spin: usize, beta: f64) -> f64 {
    let log_z = 2.0 * log_canonical_partition_function_per_spin(spectrum, n_per_spin, beta);
    log_z.exp()
}

/// Canonical free energy `F = -ln Z / beta`. Safe for large `beta`.
#[must_use]
pub fn free_energy(spectrum: &[f64], n_per_spin: usize, beta: f64) -> f64 {
    let log_z = 2.0 * log_canonical_partition_function_per_spin(spectrum, n_per_spin, beta);
    -log_z / beta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obc_l2_spectrum_is_plus_minus_one_at_t_one() {
        // L = 2, t = 1: eps = -2 cos(k*pi/3), k = 1, 2.
        // k=1: -2 cos(pi/3) = -1; k=2: -2 cos(2 pi/3) = +1. Ascending: [-1, +1].
        let s = single_particle_spectrum_obc(2, 1.0);
        assert_eq!(s.len(), 2);
        assert!((s[0] - (-1.0)).abs() < 1e-12, "got {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-12, "got {}", s[1]);
    }

    #[test]
    fn pbc_l4_spectrum_has_degenerate_k_pi_half_levels() {
        // L = 4: k = 0, 1, 2, 3 -> -2 cos = -2, 0 (twice), +2 ascending.
        let s = single_particle_spectrum_pbc(4, 1.0);
        assert!((s[0] - (-2.0)).abs() < 1e-12);
        assert!(s[1].abs() < 1e-12);
        assert!(s[2].abs() < 1e-12);
        assert!((s[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn gs_energy_obc_l2_n1_per_spin_matches_dimer_at_u_zero() {
        let e = ground_state_energy_obc(2, 1, 1.0);
        assert!((e - (-2.0)).abs() < 1e-12);
        let e_dimer = super::super::dimer::ground_state_energy(1.0, 0.0);
        assert!((e - e_dimer).abs() < 1e-12);
    }

    #[test]
    fn thermodynamic_per_site_matches_half_filling() {
        // n = 1: e(1) = -(4/pi) sin(pi/2) = -4/pi.
        let e = ground_state_energy_per_site_thermodynamic(1.0, 1.0);
        assert!((e - (-4.0 / PI)).abs() < 1e-14);
    }

    #[test]
    fn canonical_z_high_t_limit_counts_combinations() {
        // beta -> 0, n_per_spin = 1 of L = 4 OBC: Z_per_spin -> 4, total Z -> 16.
        let spec = single_particle_spectrum_obc(4, 1.0);
        let z = canonical_partition_function(&spec, 1, 1.0e-10);
        assert!((z - 16.0).abs() < 1e-6, "got {z}, expected 16");
    }

    #[test]
    fn canonical_f_low_t_approaches_gs_energy() {
        // Pratt recursion suffers catastrophic cancellation at very
        // large beta on small N (Z_2 = (Z_1(b)^2 - Z_1(2b))/2 with both
        // terms approaching 1). beta = 10 is small enough to keep the
        // recursion well-conditioned and large enough that F ~= E_GS.
        let spec = single_particle_spectrum_obc(4, 1.0);
        let f = free_energy(&spec, 2, 10.0);
        let e_gs = ground_state_energy_obc(4, 2, 1.0);
        assert!(
            (f - e_gs).abs() < 1e-3,
            "F = {f}, E_GS = {e_gs} should agree at beta = 10"
        );
    }
}
