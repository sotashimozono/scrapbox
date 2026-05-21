//! Layer 1 — exchange-correlation functionals.

pub mod hubbard_lda;
pub mod non_interacting;

use crate::config::{HubbardLdaParams, XcFunctional as XcConfig};
use crate::error::Result;
use crate::hamiltonian::KohnShamHamiltonian;

/// Dispatching enum over xc functional variants.
#[derive(Debug, Clone)]
pub enum ExchangeCorrelation {
    /// Single-site analytical Hubbard LDA.
    HubbardLda(hubbard_lda::HubbardLda),
    /// `λ^{h-xc} = 0` everywhere.
    NonInteracting,
}

impl ExchangeCorrelation {
    /// Build from config + Hamiltonian context (needs `U`, `β`).
    pub fn from_config(cfg: &XcConfig, hamiltonian: &KohnShamHamiltonian) -> Result<Self> {
        match cfg {
            XcConfig::HubbardLda { params } => Ok(Self::HubbardLda(hubbard_lda::HubbardLda::new(
                hamiltonian.on_site_interaction,
                hamiltonian.beta,
                params.clone(),
            ))),
            XcConfig::NonInteracting => Ok(Self::NonInteracting),
        }
    }

    /// Evaluate the site-wise Hartree-XC potential `λ^{h-xc}_i[n]`.
    #[must_use]
    pub fn evaluate(&self, density: &[f64]) -> Vec<f64> {
        match self {
            Self::HubbardLda(lda) => lda.evaluate(density),
            Self::NonInteracting => vec![0.0; density.len()],
        }
    }

    /// Convenience constructor for v0.1 default.
    #[must_use]
    pub fn hubbard_lda(u: f64, beta: f64) -> Self {
        Self::HubbardLda(hubbard_lda::HubbardLda::new(
            u,
            beta,
            HubbardLdaParams::default(),
        ))
    }
}
