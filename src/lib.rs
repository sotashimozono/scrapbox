//! # scrapbox
//!
//! Reference implementation of **canonical-ensemble finite-temperature density
//! functional theory** for lattice fermion systems (1D Hubbard family in
//! v0.1), with quantum-thermodynamics observables of sudden quenches.
//!
//! The library is driven end-to-end by a single `config.toml`; see
//! [`config`] and the project's `notes/discipline/CONFIG.md` for the schema.
//!
//! ## Layers
//!
//! | Layer | Module | Concept |
//! |---|---|---|
//! | 0 — Inputs | [`hamiltonian`] | `KohnShamHamiltonian` and lattice models |
//! | 1 — XC | [`xc`] | `ExchangeCorrelation` |
//! | 3 — Spectrum | [`spectrum`] | Eigendecomposition of `H^KS` |
//! | 4 — Density | [`density`] | `CanonicalDensityEvaluator` (Pratt recursion) |
//! | 5 — SCF | [`scf`] | `CanonicalThermalDFTSolver` |
//! | 6 — Observables | [`observables`] | Mean work, irreversible entropy, ... |
//!
//! Layer 2 (KS Hamiltonian assembly) lives inside [`hamiltonian`].
//!
//! ## Naming
//!
//! Per `notes/discipline/CONVENTIONS.md`, names follow the physics literally:
//! `KohnShamHamiltonian`, `PrattRecursion`, `SuddenQuenchEvaluator`. No
//! mathematical camouflage.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod config;
pub mod density;
pub mod error;
pub mod hamiltonian;
pub mod observables;
pub mod output;
pub mod quench;
pub mod scf;
pub mod spectrum;
pub mod sweep;
pub mod validation;
pub mod xc;

pub use crate::error::{Result, ScrapboxError};
