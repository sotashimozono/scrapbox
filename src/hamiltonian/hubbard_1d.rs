//! 1D inhomogeneous Hubbard Hamiltonian.

use crate::config::Hamiltonian as HamiltonianConfig;
use crate::error::Result;
use faer::Mat;

/// The Kohn-Sham Hamiltonian operating on a single-spin sector.
#[derive(Debug, Clone)]
pub struct KohnShamHamiltonian {
    /// Lattice length `L`.
    pub num_sites: usize,
    /// Hopping parameter `J`.
    pub hopping_j: f64,
    /// On-site interaction `U`.
    pub on_site_interaction: f64,
    /// External potential `V_i`, length `num_sites`.
    pub external_potential: Vec<f64>,
    /// Inverse temperature `β`.
    pub beta: f64,
    /// Electrons per spin sector, `N_↑ = N_↓`.
    pub num_electrons_per_spin: usize,
}

impl KohnShamHamiltonian {
    /// Build from a parsed [`HamiltonianConfig`].
    pub fn from_config(cfg: &HamiltonianConfig) -> Result<Self> {
        let external_potential = cfg.external_potential.to_site_values(cfg.num_sites);
        Ok(Self {
            num_sites: cfg.num_sites,
            hopping_j: cfg.hopping_j,
            on_site_interaction: cfg.on_site_interaction,
            external_potential,
            beta: cfg.beta,
            num_electrons_per_spin: cfg.num_electrons_per_spin,
        })
    }

    /// Assemble the per-spin single-particle KS matrix given the current
    /// site-wise HXC potential `λ^{h-xc}_i`.
    #[must_use]
    pub fn build_ks_matrix(&self, hxc_potential: &[f64]) -> Mat<f64> {
        assert_eq!(
            hxc_potential.len(),
            self.num_sites,
            "hxc_potential length must equal num_sites"
        );
        let n = self.num_sites;
        let mut h = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            h[(i, i)] = self.external_potential[i] + hxc_potential[i];
            if i + 1 < n {
                h[(i, i + 1)] = -self.hopping_j;
                h[(i + 1, i)] = -self.hopping_j;
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExternalPotential;

    fn dimer_cfg() -> HamiltonianConfig {
        HamiltonianConfig {
            model: "hubbard_1d_inhomogeneous".into(),
            num_sites: 2,
            hopping_j: 1.0,
            on_site_interaction: 4.0,
            spinful: true,
            num_electrons_per_spin: 1,
            beta: 2.0,
            units: "j_units".into(),
            external_potential: ExternalPotential::Uniform { amplitude: 0.0 },
        }
    }

    #[test]
    fn builds_dimer_ks_matrix_with_zero_hxc() {
        let h = KohnShamHamiltonian::from_config(&dimer_cfg()).unwrap();
        let m = h.build_ks_matrix(&[0.0, 0.0]);
        assert!((m[(0, 0)] - 0.0).abs() < 1e-12);
        assert!((m[(0, 1)] - (-1.0)).abs() < 1e-12);
        assert!((m[(1, 0)] - (-1.0)).abs() < 1e-12);
        assert!((m[(1, 1)] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn ks_matrix_picks_up_external_and_hxc() {
        let mut cfg = dimer_cfg();
        cfg.external_potential = ExternalPotential::Comb { amplitude: 0.5 };
        let h = KohnShamHamiltonian::from_config(&cfg).unwrap();
        let m = h.build_ks_matrix(&[0.1, -0.2]);
        assert!((m[(0, 0)] - 0.6).abs() < 1e-12);
        assert!((m[(1, 1)] - (-0.7)).abs() < 1e-12);
    }
}
