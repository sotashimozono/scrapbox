//! Layer 5 — self-consistent field loop.
//!
//! See `notes/discipline/canonical_thermal_dft.md` Sec VII and
//! `notes/Zettelkasten/PermanentNote/canonical-ks-construction.md`.

pub mod mixing;
pub mod state;

pub use mixing::DensityMixer;
pub use state::KsState;

use crate::config::{InitialDensity, Scf as ScfConfig};
use crate::density::CanonicalDensityEvaluator;
use crate::error::{Result, ScrapboxError};
use crate::hamiltonian::KohnShamHamiltonian;
use crate::spectrum::SpectrumSource;
use crate::xc::ExchangeCorrelation;

/// The canonical-ensemble finite-T DFT solver.
#[derive(Debug, Clone)]
pub struct CanonicalThermalDFTSolver {
    /// Layer 0 — physical model.
    pub hamiltonian: KohnShamHamiltonian,
    /// Layer 1 — xc functional.
    pub xc: ExchangeCorrelation,
    /// Layer 3 — spectrum source.
    pub spectrum: SpectrumSource,
    /// Layer 4 — density evaluator.
    pub density: CanonicalDensityEvaluator,
    /// Layer 5 — mixing strategy and tolerances.
    pub scf_config: ScfConfig,
}

impl CanonicalThermalDFTSolver {
    /// Construct the solver.
    #[must_use]
    pub fn new(
        hamiltonian: KohnShamHamiltonian,
        xc: ExchangeCorrelation,
        spectrum: SpectrumSource,
        density: CanonicalDensityEvaluator,
        scf_config: ScfConfig,
    ) -> Self {
        Self {
            hamiltonian,
            xc,
            spectrum,
            density,
            scf_config,
        }
    }

    /// Run the SCF loop until convergence or maximum iterations.
    pub fn solve(&self) -> Result<KsState> {
        let num_sites = self.hamiltonian.num_sites;
        let mut density = self.initial_density()?;
        let mut mixer = DensityMixer::from_config(&self.scf_config.mixing);

        let mut residual = f64::INFINITY;
        for iteration in 0..self.scf_config.max_iterations {
            let hxc = self.xc.evaluate(&density);
            let ks_matrix = self.hamiltonian.build_ks_matrix(&hxc);
            let eigen = self.spectrum.diagonalize(&ks_matrix)?;
            let result = self.density.evaluate(
                &eigen,
                self.hamiltonian.num_electrons_per_spin,
                self.hamiltonian.beta,
            )?;

            residual = max_abs_diff(&density, &result.densities);

            if residual < self.scf_config.tolerance {
                let free_energy = -result.partition_function.ln() / self.hamiltonian.beta;
                let _ = num_sites; // (reserved for future per-iteration tracing)
                return Ok(KsState {
                    densities: result.densities,
                    eigen,
                    partition_function_per_spin: result.partition_function_per_spin,
                    partition_function: result.partition_function,
                    free_energy,
                    hxc_potential: hxc,
                    iterations: iteration + 1,
                    residual,
                });
            }

            mixer.mix_in_place(&mut density, &result.densities);
            let _ = num_sites; // unused-binding suppression for future logging
        }

        Err(ScrapboxError::ScfDivergence {
            iterations: self.scf_config.max_iterations,
            last_residual: residual,
        })
    }

    fn initial_density(&self) -> Result<Vec<f64>> {
        let l = self.hamiltonian.num_sites;
        let n_per_spin = self.hamiltonian.num_electrons_per_spin;
        match &self.scf_config.initial_density {
            InitialDensity::Uniform => {
                // Total electrons = 2 · n_per_spin. Per site = 2 · n_per_spin / L.
                let n_site = 2.0 * (n_per_spin as f64) / (l as f64);
                Ok(vec![n_site; l])
            }
            InitialDensity::Explicit { values } => {
                if values.len() != l {
                    return Err(ScrapboxError::ConfigValidation {
                        message: format!(
                            "scf.initial_density.values length {} != num_sites {}",
                            values.len(),
                            l
                        ),
                    });
                }
                Ok(values.clone())
            }
        }
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ExternalPotential, Hamiltonian as HamiltonianConfig, Mixing};

    fn dimer_solver() -> CanonicalThermalDFTSolver {
        let h_cfg = HamiltonianConfig {
            model: "hubbard_1d_inhomogeneous".into(),
            num_sites: 2,
            hopping_j: 1.0,
            on_site_interaction: 4.0,
            spinful: true,
            num_electrons_per_spin: 1,
            beta: 2.0,
            units: "j_units".into(),
            external_potential: ExternalPotential::Uniform { amplitude: 0.0 },
        };
        let h = KohnShamHamiltonian::from_config(&h_cfg).unwrap();
        let xc = ExchangeCorrelation::hubbard_lda(4.0, 2.0);
        let spectrum = SpectrumSource::DenseDiag;
        let density = CanonicalDensityEvaluator::pratt_default();
        let scf_config = ScfConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            mixing: Mixing::Linear { alpha: 0.5 },
            initial_density: InitialDensity::Uniform,
        };
        CanonicalThermalDFTSolver::new(h, xc, spectrum, density, scf_config)
    }

    #[test]
    fn dimer_half_filling_symmetric_density() {
        // Half-filling, no external potential, particle-hole symmetric →
        // converged density should be exactly (1, 1).
        let solver = dimer_solver();
        let state = solver.solve().expect("SCF should converge");
        assert!(
            (state.densities[0] - 1.0).abs() < 1e-8,
            "n_0 = {} differs from 1.0",
            state.densities[0]
        );
        assert!(
            (state.densities[1] - 1.0).abs() < 1e-8,
            "n_1 = {} differs from 1.0",
            state.densities[1]
        );
        // Σ n_i = N = 2.
        let total: f64 = state.densities.iter().sum();
        assert!((total - 2.0).abs() < 1e-10);
        // Z_N > 0 and F finite.
        assert!(state.partition_function > 0.0);
        assert!(state.free_energy.is_finite());
        // Should converge in just a few iterations.
        assert!(
            state.iterations < 50,
            "took {} iterations",
            state.iterations
        );
    }
}
