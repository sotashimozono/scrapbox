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
}
