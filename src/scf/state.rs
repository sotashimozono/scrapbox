//! Output of a converged SCF run.

use crate::spectrum::Eigendecomposition;

/// Result of one converged self-consistent SCF calculation.
#[derive(Debug, Clone)]
pub struct KsState {
    /// Converged thermal density `{n_i^β}`, length `num_sites`.
    pub densities: Vec<f64>,
    /// Converged eigendecomposition of `H^KS`.
    pub eigen: Eigendecomposition,
    /// Per-spin canonical partition function `Z_{N_σ}(β)`.
    pub partition_function_per_spin: f64,
    /// Total canonical partition function `Z_N(β)`.
    pub partition_function: f64,
    /// Helmholtz free energy `F = -β⁻¹ ln Z_N`.
    pub free_energy: f64,
    /// Final converged HXC potential.
    pub hxc_potential: Vec<f64>,
    /// Number of SCF iterations executed.
    pub iterations: usize,
    /// Final density-change residual.
    pub residual: f64,
}
