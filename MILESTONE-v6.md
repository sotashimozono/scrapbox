# MILESTONE-v6 - scrapbox v0.6

Status: **shipped 2026-05-24** - minimum-scope gates green.

This is the v0.6 contract. v0.6 is the **scale-out** milestone: the
trait abstraction introduced in v0.5 γ finally has a real consumer
(matrix-free Hubbard JW operator), and the typicality sampler grows
from per-site density to the full sudden-quench work pipeline plus a
complex-Gaussian variance-reduction path. End-to-end, observables that
previously required exact diagonalisation can now in principle be
estimated on Hilbert spaces too large to materialise.

## What v0.6 adds over v0.5

- **Matrix-free Hubbard JW operator** (PR #28, batch α)
  `src/spectrum/hubbard_jw.rs` introduces `JwHubbard`, an `impl
  LinearOperator` over the 1D inhomogeneous Hubbard Hamiltonian in the
  Jordan-Wigner occupation-bitmask basis. The same Lanczos kernel that
  consumed `Mat<f64>` in v0.5 now consumes `JwHubbard` without ever
  materialising the `dim x dim` matrix.

  Basis convention is shared with `reference::ed` so cross-checks
  compare apples-to-apples. The pre-computed diagonal vector amortises
  the `U * doubles + sum_i v_ext[i] * occ_i` work across `apply` calls;
  hop branches enumerate per-spin nearest-neighbour moves with JW sign
  via `reference::ed::fermion_sign`.

- **TPQ canonical work statistics** (PR #29, batch β)
  `reference::tpq::tpq_work_statistics(ed_init, ed_final, beta,
  n_samples, seed)` extends the typicality sampler from density to the
  full sudden-quench work pipeline. Returns
  `TpqWorkStats { mean_w, work_variance, mean_w_stderr, n_samples }`
  with `<W>` and `sigma_W^2 = <W^2> - <W>^2` pooled across samples
  (Sugiura-Shimizu unbiased ratio estimator) plus a per-sample standard
  error of `mean_w`. Internal `apply_hamiltonian(psi, ed)` helper uses
  the eigen-decomp to act with H without exposing the dense matrix.

- **Complex-Gaussian TPQ variance reduction** (PR #30, batch γ)
  Opt-in complex-amplitude variants:
  - `tpq_density_complex(...)`
  - `tpq_work_statistics_complex(...)`
  Both draw `c_k = (re + i im) / sqrt(2)` with `re, im ~ N(0, 1)`.
  Theoretical asymptotic factor `1 / sqrt(2)` on `mean_w_stderr`;
  empirically verified `< 0.9` at `L=4, n_samples=800`.
  `box_muller_pair` returns both Gaussian samples per call (no work
  wasted on the sin partner). The real-amplitude defaults are unchanged
  to preserve numeric reproducibility for existing callers.

The CLI surface is unchanged from v0.5 (`run / validate / sweep / bench
/ doctor`). Schema remains `0.2` (no breaking config changes).

## Config-key surface

No new keys in v0.6. The matrix-free JW operator is a library-side
abstraction; CLI dispatching it through `spectrum_source` is deferred
to a future batch.

## Acceptance gates

- **α.1 - JW matvec parity at L=2 dimer**: `jw.apply(e_col)` matches
  `H_ed * e_col` reconstructed from ED eigen-decomp to 1e-10.
- **α.2 - JW matvec parity at L=4 inhomogeneous**: same, with
  `v_ext = [0.1, -0.2, 0.3, -0.1]`.
- **α.3 - JW matvec parity in spin-polarised sector**: same, with
  `n_up != n_dn`.
- **α.4 - Lanczos through trait at L=4**: GS via `diagonalize(&jw, ...)`
  matches ED GS to 1e-9.
- **α.5 - Lanczos through trait at L=6 half-filling**: three lowest
  eigenvalues match ED to 1e-7 at Hilbert dim 400.

- **β.1 - TPQ work zero-quench identity**: `ed_init == ed_final` gives
  `mean_w` and `work_variance` within 1e-10 of zero.
- **β.2 - TPQ work matches ED at L=4 sudden quench**: 600 samples,
  comb-V quench `v0 = 0.3`; `<W>` within 0.05 of ED, `sigma_W^2`
  within 0.1.
- **β.3 - TPQ work determinism**: same `(beta, n_samples, seed)` gives
  identical output across calls.

- **γ.1 - Complex TPQ density correctness**: 500 samples at L=4, same
  0.05 tolerance band as the real path (no correctness regression).
- **γ.2 - Complex TPQ work correctness**: same L=4 quench gates as
  β.2.
- **γ.3 - Complex variance reduction**: at fixed `n_samples = 800`,
  `mean_w_stderr_complex / mean_w_stderr_real < 0.9` (theoretical
  asymptote `~ 1/sqrt(2) ~ 0.71`).
- **γ.4 - Complex determinism**: same seed gives identical density.

## Tests at tag time

*(Snapshot - authoritative source is `cargo test --release`.)*

- 166 tests (145 unit + 21 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.7+)

- **Exact `Theta_2`**: full Palamara Sec III.3 derivation (beyond LDA
  placeholder shipped in v0.5 α). Refinement still tracked on #22.
- **CLI dispatch of matrix-free JW**: `JwHubbard` is library-side
  only; wiring it through `spectrum_source` so configs can request
  matrix-free Hubbard solves needs schema extension.
- **Sparse Hubbard CSR builder**: a `SparseMatrix::from_hubbard(L,
  n_up, n_dn, J, U, v_ext)` would complement `JwHubbard` for cases
  where building the sparse matrix once is cheaper than per-apply
  on-the-fly enumeration.
- **TPQ at scale**: `tpq_density(_complex)` and
  `tpq_work_statistics(_complex)` currently take an `EdResult`. A
  matrix-free TPQ path (driven by `LinearOperator + e^{-beta H/2}`
  via Lanczos exponentiation) is the natural follow-up that closes the
  scale-out loop.
- **True finite-T BALDA**: carryover from v0.4 / v0.5.

## Source-of-truth references

- `notes/discipline/PHASES.md` - milestone definitions.
- `notes/discipline/ACCEPTANCE.md` - mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) - per-batch log.
- Sugiura, Shimizu, *Phys. Rev. Lett.* 108, 240401 (2012); 111, 010401
  (2013) - TPQ formulation and complex-amplitude variance argument.
- Palamara, Plata, Pekola, Goold, *Phys. Rev. Lett.* 133, 207101
  (2024) - generalized FDR and `Theta_2` (consumed by α dispatch in
  v0.5 and by β work pipeline in v0.6).
- Jordan, Wigner, *Z. Phys.* 47, 631 (1928) - fermion-to-spin mapping
  underlying the JW occupation basis.
- Lanczos, *J. Res. Natl. Bur. Stand.* 45 (1950) - iterative kernel
  consumed via the v0.5 γ `LinearOperator` apply contract by the v0.6
  α `JwHubbard` implementation.
