//! Layer 4 — canonical density evaluators.

pub mod gce_projection;
pub mod pratt;

use crate::config::{DensityEvaluator as DensityConfig, PrattParams};
use crate::error::Result;
use crate::spectrum::Eigendecomposition;

/// Output of a single density-evaluator invocation.
#[derive(Debug, Clone)]
pub struct DensityResult {
    /// Site occupations summed over spin: `n_i = n_↑_i + n_↓_i`.
    pub densities: Vec<f64>,
    /// Per-spin canonical partition function `Z_{N_σ}(β)`.
    pub partition_function_per_spin: f64,
    /// Total canonical partition function `Z_N = (Z_{N_σ})^2`.
    pub partition_function: f64,
}

/// Dispatching enum over density-evaluator variants.
#[derive(Debug, Clone)]
pub enum CanonicalDensityEvaluator {
    /// Pratt recursion (exact for non-interacting fermions).
    PrattRecursion(pratt::PrattParamsRuntime),
    /// Grand-canonical fugacity-circle quadrature + Fourier projection.
    GcePlusProjection(gce_projection::GceProjectionParamsRuntime),
}

impl CanonicalDensityEvaluator {
    /// Build from config.
    pub fn from_config(cfg: &DensityConfig) -> Result<Self> {
        match cfg {
            DensityConfig::PrattRecursion { params } => Ok(Self::PrattRecursion(
                pratt::PrattParamsRuntime::from(params),
            )),
            DensityConfig::GcePlusProjection { params } => Ok(Self::GcePlusProjection(
                gce_projection::GceProjectionParamsRuntime::from(params),
            )),
        }
    }

    /// Convenience: build with default Pratt parameters.
    #[must_use]
    pub fn pratt_default() -> Self {
        Self::PrattRecursion(pratt::PrattParamsRuntime::from(&PrattParams::default()))
    }

    /// Evaluate `({n_i}, Z_N)` from the supplied KS eigendecomposition.
    pub fn evaluate(
        &self,
        eigen: &Eigendecomposition,
        num_electrons_per_spin: usize,
        beta: f64,
    ) -> Result<DensityResult> {
        match self {
            Self::PrattRecursion(params) => {
                pratt::evaluate(eigen, num_electrons_per_spin, beta, params)
            }
            Self::GcePlusProjection(params) => {
                gce_projection::evaluate(eigen, num_electrons_per_spin, beta, params)
            }
        }
    }
}
