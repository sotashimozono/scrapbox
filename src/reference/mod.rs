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
//!   function in log-space, thermodynamic-limit
//!   `e(n) = -(4t/pi) sin(pi*n/2)`.
//! - [`atomic`] - atomic limit (`J = 0`) closed forms: per-site
//!   grand-canonical partition function and density.
//! - [`ed`] - generic many-body exact diagonalisation: bitmask basis
//!   enumerator, Jordan-Wigner signed hopping, full Hubbard Hamiltonian
//!   builder, canonical thermal density / partition function /
//!   free energy. Practically resource-limited to `L <= 6`.
//! - [`bethe`] - Bethe-ansatz integrals: Lieb-Wu half-filling
//!   ground-state energy `e_h(u) = -4 int J_0 J_1 / [x(1 + exp(ux/2))]`.
//! - [`heisenberg`] - Large-U effective Heisenberg model: superexchange
//!   `J_H = 4t^2/U` and the per-site Bethe constant `-ln 2 + 1/4`.
//! - [`high_t_expansion`] - Beta -> 0 leading free energy `-T ln D`
//!   where `D = C(L, N_up) * C(L, N_dn)` is the Hilbert dimension.
//! - [`identities`] - Closure-based verifiers for thermodynamic
//!   identities (Hellmann-Feynman, `d(beta F)/dbeta = E`, Mermin
//!   minimum). Generic over any free-energy/density evaluator.

pub mod atomic;
pub mod bethe;
pub mod dimer;
pub mod ed;
pub mod free_chain;
pub mod heisenberg;
pub mod high_t_expansion;
pub mod identities;
pub mod tpq;
