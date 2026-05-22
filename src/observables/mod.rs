//! Layer 6 — quantum-thermodynamics observables bundle.

use serde::{Deserialize, Serialize};

/// Observable bundle dumped by the runner.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservableReport {
    /// `F = -β⁻¹ ln Z_N`.
    pub free_energy: Option<f64>,
    /// Total canonical partition function `Z_N(β)`.
    pub partition_function: Option<f64>,
    /// `<W>` for the requested sudden quench.
    pub mean_work: Option<f64>,
    /// `<S_irr> = β(<W> − ΔF)`.
    pub irreversible_entropy: Option<f64>,
    /// `<W²>` second moment of the work distribution (Palamara eq 28+30).
    pub mean_work_sq: Option<f64>,
    /// Work variance `σ_w² = <W²> − <W>²`.
    pub work_variance: Option<f64>,
    /// Quantum correction `Θ_2[{a^β,0}]` (Palamara eq 30). `0.0` in the
    /// commuting-Hamiltonian limit; `"zero"` method always emits `0.0`.
    pub theta_2: Option<f64>,
    /// Generalized-FDR residual (Palamara eq 31):
    /// `<S_irr> − (β²/2)(σ_w² − Θ_2)`. Used as a self-consistency probe.
    pub fdr_residual: Option<f64>,
}
