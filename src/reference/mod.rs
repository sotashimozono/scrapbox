//! Exact / analytic / closed-form references for cross-checking the
//! canonical thermal DFT solver.
//!
//! Each submodule provides reference values that the solver pipelines
//! must reproduce under their stated assumptions. Tests across the
//! crate consume these as ground truth (Level-3 cross-checks in the
//! consistency matrix discussion).
//!
//! ## Module map
//!
//! - [`dimer`] - analytic closed-form spectrum and observables for the
//!   `L = 2` Hubbard dimer in the half-filled `(N_up=1, N_dn=1)` sector.
//! - [`free_chain`] - non-interacting (`U = 0`) 1D chain reference:
//!   single-particle spectrum (OBC + PBC), canonical thermal partition
//!   function, ground-state energy with the thermodynamic-limit
//!   `e(n) = -(4t/pi) sin(pi*n/2)` per-site formula.
//! - [`atomic`] - atomic limit (`J = 0`) closed forms: per-site
//!   grand-canonical partition function and density.
//! - [`ed`] - generic many-body exact diagonalisation: bitmask basis
//!   enumerator, Jordan-Wigner signed hopping, full Hubbard Hamiltonian
//!   builder, canonical thermal density / partition function /
//!   free energy. Practically resource-limited to `L <= 6`.

pub mod atomic;
pub mod dimer;
pub mod ed;
pub mod free_chain;
