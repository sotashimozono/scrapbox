//! Layer 1 — exchange-correlation functionals.

pub mod balda;
pub mod balda_finite_t;
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
    /// BALDA — Bethe-ansatz local density approximation (Lima 2003).
    Balda(balda::Balda),
    /// BALDA finite-T dispatch shim (v0.13 beta).
    BaldaFiniteT(balda_finite_t::BaldaFiniteT),
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
            XcConfig::Balda { params } => Ok(Self::Balda(balda::Balda::new(
                hamiltonian.on_site_interaction / hamiltonian.hopping_j,
                params.clone(),
            ))),
            XcConfig::BaldaFiniteT { params } => {
                Ok(Self::BaldaFiniteT(balda_finite_t::BaldaFiniteT::new(
                    hamiltonian.on_site_interaction / hamiltonian.hopping_j,
                    hamiltonian.beta,
                    params.clone(),
                )))
            }
            XcConfig::NonInteracting => Ok(Self::NonInteracting),
        }
    }

    /// Evaluate the site-wise Hartree-XC potential `λ^{h-xc}_i[n]`.
    #[must_use]
    pub fn evaluate(&self, density: &[f64]) -> Vec<f64> {
        match self {
            Self::HubbardLda(lda) => lda.evaluate(density),
            Self::Balda(b) => b.evaluate(density),
            Self::BaldaFiniteT(b) => b.evaluate(density),
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
