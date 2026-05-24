# MILESTONE-v5 - scrapbox v0.5

Status: **shipped 2026-05-24** - minimum-scope gates green.

This is the v0.5 contract. v0.5 is the **first-cut quantum-correction**
milestone: BALDA finally has a non-trivial `Theta_2` dispatch
(closing the FDR for non-commuting quenches), the canonical density
gains a typicality-based estimator that scales beyond ED-feasible
sizes, and the Lanczos kernel is decoupled from dense matrix storage
so future matrix-free Hubbard backends can plug in without touching
the iterative core.

## What v0.5 adds over v0.4

- **`theta_2.method = "lda"` dispatch** (PR #24, batch α, partial #22)
  `src/observables/theta_2_lda.rs` provides a first-cut LDA estimate
  of the Palamara 2024 eq 31 quantum correction:

  ```text
  Theta_2_LDA = sum_i [delta V_i - mean(delta V)]^2
              * n_i (2 - n_i) / 2
              * alpha_BALDA(u),
  alpha_BALDA(u) = (2 - beta_Lima(u)) / 2.
  ```

  Symmetries are explicit:
  - vanishes for uniform `delta V` (commuting quench)
  - vanishes at `U=0` (free electrons)
  - vanishes at `n=0` and `n=2` (empty / full bands)

  Only dispatches when `xc_functional.kind = "balda"`; other XC kinds
  return a `ConfigValidation` error pointing at the coupling
  requirement. Numerical agreement with the exact `Theta_2` is **not**
  claimed; refinement to the full Palamara Sec III.3 derivation
  remains tracked on #22.

  E2E: `configs/dimer_balda_quench.toml` + `tests/lda_theta_2_e2e.rs`
  exercise the dispatcher and verify the LDA `Theta_2` improves FDR
  closure relative to `method = "zero"` baseline.

- **TPQ canonical-thermal sampler** (PR #25, batch β)
  `reference::tpq::tpq_density(ed, beta, n_samples, seed)` implements
  a Typical Pure Quantum state sampler in the ED energy basis. For a
  random pure state `|psi_0> = sum_k c_k |k>` with real Gaussian
  `c_k`, `|psi_beta> = e^{-beta H / 2} |psi_0>` is evolved and the
  per-site occupation `<psi_beta| n_i |psi_beta> / <psi_beta|psi_beta>`
  averaged across samples gives the canonical density.

  Self-averaging gates:
  - L=2 dimer: < 0.1 site error at 50 samples (dim = 4)
  - L=4 SU(2): < 0.05 site error at 500 samples (dim = 36)
  - L=6 SU(2): < 0.02 site error at 100 samples (dim = 400)
  - Deterministic with fixed seed; distinct seeds produce distinct
    estimates.

  Implemented entirely on top of `reference::ed::EdResult`, so the
  L=6 ED gold from v0.4 is the cross-check anchor.

- **`LinearOperator` trait + `SparseMatrix` CSR** (PR #26, batch γ)
  `src/spectrum/linear_operator.rs` introduces

  ```rust
  pub trait LinearOperator {
      fn dim(&self) -> usize;
      fn apply(&self, x: &[f64], y: &mut [f64]);
  }
  ```

  with impls for `faer::Mat<f64>` (dense, preserves v0.2 numerics
  bit-for-bit) and `SparseMatrix` (CSR built via `from_triples`,
  sums duplicates, drops zeros, exposes `nnz()`).

  `src/spectrum/lanczos.rs::diagonalize` is generic over
  `<O: LinearOperator>`, so the same iterative kernel will consume
  matrix-free Hubbard JW Hamiltonian applications when that lands.

The CLI surface is unchanged from v0.4. Schema remains `0.2` (no
breaking config changes).

## Config-key surface (delta from v0.4)

The only new behaviour key is `observables.theta_2.method = "lda"`,
which already existed in the schema as an enum variant but previously
returned a not-implemented error. v0.5 makes it functional when paired
with BALDA xc; other XC kinds error with an actionable message.

## Acceptance gates

- **α.1 - LDA Theta_2 symmetries**: zero at `U=0`, zero for uniform
  quench, zero at empty band, zero at saturated band, half-filling
  kernel-maximal, large-u alpha saturation (8 symmetry tests in
  `theta_2_lda`).
- **α.2 - dispatcher rejects bad inputs**: returns `ConfigValidation`
  for non-BALDA XC (coupling requirement), negative `U/t` (BALDA
  domain), and non-physical density outside `[0, 2]` (upstream bug).
- **α.3 - BALDA dimer non-commuting quench**: LDA Theta_2 reduces
  `|FDR residual|` relative to `method = "zero"` baseline
  (`tests/lda_theta_2_e2e.rs`).
- **β.1 - TPQ dimer convergence**: 50 samples bring site density to
  within 0.1 of ED at dim = 4.
- **β.2 - TPQ L=4 cross-check**: 500 samples reach 0.05 agreement
  with `reference::ed::thermal_density` at dim = 36.
- **β.3 - TPQ L=6 self-averaging**: 100 samples reach 0.02 agreement
  at dim = 400.
- **β.4 - TPQ determinism**: same `(beta, n_samples, seed)` gives
  identical output across calls.
- **γ.1 - LinearOperator dense parity**: `Mat<f64>` impl matches
  in-tree dense matvec (Lanczos dimer + L=4 KS unchanged).
- **γ.2 - SparseMatrix cross-check**: CSR matvec matches dense on
  3x3 symmetric test (`linear_operator::tests::sparse_matches_dense_matvec`).
- **γ.3 - Lanczos generic refactor**: `diagonalize<O>` consumes
  both `Mat<f64>` and `SparseMatrix` without numeric drift.

## Tests at tag time

*(Snapshot - authoritative source is `cargo test --release`.)*

- 154 tests (133 unit + 21 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.6 or later)

- **Exact `Theta_2`**: full Palamara Sec III.3 derivation (beyond
  LDA placeholder). Refinement tracked on #22.
- **Matrix-free Hubbard JW operator**: γ added the trait, but the
  many-body JW Hamiltonian is not yet implemented as a
  `LinearOperator` consumer. ED still builds dense matrices.
- **TPQ work observables**: TPQ currently provides density only; FDR
  pipeline (`<W>`, `sigma_w^2`, `Theta_2`) still routes through
  exact diagonalisation.
- **Complex-Gaussian TPQ**: real Gaussian suffices for the real
  symmetric lattice Hubbard H; complex-amplitude variant would lower
  variance further.
- **True finite-T BALDA**: v0.5 keeps the T=0 xc approximation
  evaluated at the finite-T density (carryover from v0.4).

## Source-of-truth references

- `notes/discipline/PHASES.md` - milestone definitions.
- `notes/discipline/ACCEPTANCE.md` - mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) - per-batch log.
- Palamara, Plata, Pekola, Goold, *Phys. Rev. Lett.* 133, 207101 (2024) - generalized FDR + `Theta_2` definition.
- Lima, Silva, Capelle, PRL 90, 146402 (2003) - BALDA + `beta_Lima(u)`.
- Sugiura, Shimizu, PRL 108, 240401 (2012); 111, 010401 (2013) - TPQ
  formulation.
- Lanczos, J. Res. Natl. Bur. Stand. 45 (1950) - tridiagonalisation
  consumed via the `LinearOperator` apply contract.
