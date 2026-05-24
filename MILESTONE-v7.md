# MILESTONE-v7 - scrapbox v0.7

Status: **shipped 2026-05-24** - minimum-scope gates green.

This is the v0.7 contract. v0.7 is the **user-facing scale-out**
milestone: the matrix-free `LinearOperator` machinery introduced in
v0.5 / v0.6 finally reaches the CLI, gains a CSR backend, and gets
glued to the typicality sampler via a Krylov subspace exponentiation.
End-to-end, `scrapbox ed` now exposes three solvers (dense, matrix-
free Lanczos, sparse CSR Lanczos) and `reference::tpq` can estimate
canonical density without ever building the dense Hamiltonian.

## What v0.7 adds over v0.6

- **`scrapbox ed` subcommand + `[ed]` config section** (PR #32, batch α)
  New CLI dispatcher consuming a Hubbard `[hamiltonian]` config plus an
  optional `[ed]` section (`solver` + `num_eigenvalues`). Three
  backends behind a single TOML knob:
  - `"dense"`: full spectrum via `reference::ed::canonical_thermal`.
  - `"matrix_free_lanczos"`: low-k spectrum via Lanczos on the v0.6 α
    `JwHubbard` (no dim x dim matrix in memory).
  - `"sparse_lanczos"`: reserved here, wired in batch β.
  Writes `<output.directory>/ed_spectrum.json` with `solver`, `dim`,
  `num_eigenvalues_returned`, `eigenvalues`, and `wall_time_ms`.

- **`SparseMatrix::from_hubbard` CSR builder** (PR #33, batch β)
  One-shot construction of the Hubbard `H` as CSR triples (shared
  basis convention with `reference::ed` and `JwHubbard`). Wires
  `EdSolver::SparseLanczos` through to a working dispatch path.
  Complements matrix-free for workloads where many matvecs amortise
  the build cost (the regime where Lanczos `krylov_dim * 4` iterations
  multiply the per-iter cost).

- **Krylov `expm_apply` + matrix-free TPQ** (PR #34, batch γ)
  New `src/spectrum/krylov.rs::expm_apply<O: LinearOperator>(op,
  scale, psi, m)` computes `exp(scale * H) psi` via an m-step Lanczos
  + dense eigendecomposition of the m x m tridiagonal block + back-
  projection. `reference::tpq::tpq_density_matrix_free(jw, beta,
  n_samples, seed, krylov_m)` consumes it to estimate canonical
  density without ED, closing the scale-out loop: TPQ now runs on
  Hilbert spaces too large for `EdResult`.

The CLI surface gained `scrapbox ed` (now 6 subcommands:
`run / validate / sweep / bench / doctor / ed`). Schema remains 0.2
(additive `[ed]` section is `#[serde(default)]`).

## Config-key surface (delta from v0.6)

New optional section:

```toml
[ed]
solver = "dense" | "matrix_free_lanczos" | "sparse_lanczos"
num_eigenvalues = 4   # optional; dense defaults to full spectrum,
                      # iterative backends default to 8
```

No other key changes. Existing v0.6 configs continue to load.

## Acceptance gates

- **α.1 - scrapbox ed dense full spectrum at L=4**: emits 36 ascending
  eigenvalues to `ed_spectrum.json`.
- **α.2 - scrapbox ed matrix-free vs dense parity at L=4**: low-k=4
  spectrum agrees to 1e-6 (Lanczos truncation bound at dim=36).
- **β.1 - SparseMatrix::from_hubbard symmetry preservation**: CSR
  built once, consumed by Lanczos; sparse spectrum matches dense at
  L=4 (E2E via `scrapbox ed`).
- **γ.1 - Krylov expm_apply correctness**: matches dense eigen-decomp
  `U diag(exp(scale * lambda)) U^T psi` on a 5x5 SPD to 1e-10.
- **γ.2 - Krylov edge cases**: zero input -> zero; scale=0 -> identity.
- **γ.3 - tpq_density_matrix_free matches ED-path tpq_density**:
  500 samples at L=4, beta=2, comb-V give per-site density within
  0.05 of ED.
- **γ.4 - tpq_density_matrix_free dimer half-filling**: density ~ 1
  within 0.15 at 60 samples.
- **γ.5 - Determinism**: same `(beta, n_samples, seed, krylov_m)`
  reproduces identical output.

## Tests at tag time

*(Snapshot - authoritative source is `cargo test --release`.)*

- 175 tests (151 unit + 24 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.8+)

- **Exact `Theta_2`**: full Palamara Sec III.3 derivation (beyond the
  v0.5 α LDA placeholder). Tracked on #22.
- **Matrix-free TPQ work observables**: `tpq_work_statistics(_complex)`
  still takes `EdResult`. A matrix-free `<W>`, `sigma_W^2` path needs
  exp(-beta H / 2) applied twice (initial + final Hamiltonians).
- **True finite-T BALDA**: carryover.
- **CLI dispatch of matrix-free TPQ**: the new
  `tpq_density_matrix_free` is library-only; surfacing it through
  `scrapbox run`'s observables section would need schema extension.
- **Krylov adaptive m**: current `krylov_m` is a fixed user parameter.
  Adaptive selection based on residual norm is a follow-up.

## Source-of-truth references

- `notes/discipline/PHASES.md` - milestone definitions.
- `notes/discipline/ACCEPTANCE.md` - mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) - per-batch log.
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, SIAM
  (2003), Chapter 13 - Krylov subspace methods for matrix functions
  including `exp(A) v`.
- Sugiura, Shimizu, *Phys. Rev. Lett.* 108, 240401 (2012); 111,
  010401 (2013) - TPQ formulation consumed by both ED and matrix-free
  paths.
- Jordan, Wigner, *Z. Phys.* 47, 631 (1928) - fermion-to-spin
  mapping underlying the Hubbard JW basis used by both matrix-free
  (JwHubbard, v0.6 α) and CSR (SparseMatrix::from_hubbard, v0.7 β)
  backends.
- Palamara, Plata, Pekola, Goold, *Phys. Rev. Lett.* 133, 207101
  (2024) - generalized FDR carryover for v0.8 exact Theta_2 work.
