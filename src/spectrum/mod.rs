//! Layer 3 — eigendecomposition of the single-particle KS matrix.

pub mod dense_diag;
pub mod lanczos;
pub mod linear_operator;

use crate::config::SpectrumSource as SpectrumConfig;
use crate::error::Result;
use faer::Mat;

/// Output of an eigendecomposition.
#[derive(Debug, Clone)]
pub struct Eigendecomposition {
    /// Sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Column k holds eigenvector k. Shape `(num_sites, effective_m)`.
    pub eigenvectors: Mat<f64>,
}

/// Dispatching enum over spectrum-source variants.
#[derive(Debug, Clone)]
pub enum SpectrumSource {
    /// Dense LAPACK-style via `faer`.
    DenseDiag,
    /// Lanczos tridiagonalization + dense diag of `T_m`.
    Lanczos(lanczos::LanczosParams),
}

impl SpectrumSource {
    /// Build from config.
    pub fn from_config(cfg: &SpectrumConfig) -> Result<Self> {
        match cfg {
            SpectrumConfig::DenseDiag => Ok(Self::DenseDiag),
            SpectrumConfig::Lanczos {
                krylov_dim,
                max_iter,
                tol,
            } => Ok(Self::Lanczos(lanczos::LanczosParams {
                krylov_dim: *krylov_dim,
                max_iter: *max_iter,
                tol: *tol,
            })),
        }
    }

    /// Diagonalize the supplied real-symmetric matrix.
    pub fn diagonalize(&self, matrix: &Mat<f64>) -> Result<Eigendecomposition> {
        match self {
            Self::DenseDiag => dense_diag::diagonalize(matrix),
            Self::Lanczos(params) => lanczos::diagonalize(matrix, params),
        }
    }
}
