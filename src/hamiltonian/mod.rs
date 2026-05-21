//! Layer 0 — lattice Hamiltonians.
//!
//! v0.1 implements only the 1D inhomogeneous Hubbard model. The
//! [`KohnShamHamiltonian`] type carries the data needed by every downstream
//! layer (kinetic hopping, on-site interaction, external + HXC potentials,
//! particle number, inverse temperature).

pub mod hubbard_1d;

pub use hubbard_1d::KohnShamHamiltonian;
