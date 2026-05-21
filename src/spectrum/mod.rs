//! Layer 3 — eigendecomposition of the single-particle KS matrix.

pub mod dense_diag;

use crate::config::SpectrumSource as SpectrumConfig;
use crate::error::Result;
use faer::Mat;

/// Output of an eigendecomposition.
#[derive(Debug, Clone)]
pub struct Eigendecomposition {
    /// Sorted ascending. Length `num_sites`.
    pub eigenvalues: Vec<f64>,
    /// Column k holds eigenvector k. Shape `(num_sites, num_sites)`.
    pub eigenvectors: Mat<f64>,
}

/// Dispatching enum over spectrum-source variants.
#[derive(Debug, Clone)]
pub enum SpectrumSource {
    /// Dense LAPACK-style.
    DenseDiag,
}

impl SpectrumSource {
    /// Build from config.
    pub fn from_config(cfg: &SpectrumConfig) -> Result<Self> {
        match cfg {
            SpectrumConfig::DenseDiag => Ok(Self::DenseDiag),
        }
    }

    /// Diagonalize the supplied real-symmetric matrix.
    pub fn diagonalize(&self, matrix: &Mat<f64>) -> Result<Eigendecomposition> {
        match self {
            Self::DenseDiag => dense_diag::diagonalize(matrix),
        }
    }
}
