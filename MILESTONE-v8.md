# MILESTONE-v8 - scrapbox v0.8

Status: **shipped 2026-05-24** - minimum-scope gates green.

This is the v0.8 contract. v0.8 is the **matrix-free FDR pipeline**
milestone: TPQ canonical work statistics now run end-to-end through
the LinearOperator trait, the CLI exposes all four TPQ analysis modes
under a single subcommand, and the Krylov subspace exponentiation
gains a posteriori residual-based adaptive stopping. End-to-end,
`<W>`, `sigma_W^2`, and `mean_w_stderr` are computable on Hilbert
spaces too large for exact diagonalisation.

## What v0.8 adds over v0.7

- **`tpq_work_statistics_matrix_free`** (PR #36, batch alpha)
  Mirrors v0.6 beta `tpq_work_statistics` (which takes `EdResult`s) but
  runs entirely matrix-free: canonical thermal state via v0.7 gamma
  `expm_apply` on `H_init`, work operator `W = H_final - H_init`
  applied as two direct matvecs on `psi_beta`. Generic over both
  operators so mixed backends (e.g., `JwHubbard` initial + `SparseMatrix
  ::from_hubbard` final) are supported. Output `TpqWorkStats` identical
  to ED-path so downstream tooling swaps variants without API change.

- **`scrapbox tpq` subcommand + `[tpq]` config section** (PR #37, batch beta)
  All four TPQ modes exposed under one CLI knob:
  - `kind = density, source = ed` -> `tpq_density(EdResult)`
  - `kind = density, source = matrix_free` -> `tpq_density_matrix_free(JwHubbard)`
  - `kind = work_statistics, source = ed` -> `tpq_work_statistics(EdResult x 2)`
  - `kind = work_statistics, source = matrix_free` -> `tpq_work_statistics_matrix_free(JwHubbard x 2)`
  `work_statistics` requires `[quench]` for the final Hamiltonian
  (reuses existing `Quench` enum). Optional `[tpq].beta` override;
  defaults to `hamiltonian.beta`. `krylov_m` defaults to 30 for
  matrix-free paths. Writes `<output.directory>/tpq_report.json`
  (tagged variant: `Density` or `WorkStatistics`).

- **`expm_apply_adaptive` with Saad residual stopping** (PR #38, batch gamma)
  Quality upgrade for v0.7 gamma `expm_apply`. Selects the Krylov
  subspace dim `m` on the fly using Saad 1992 eq 3.6 posteriori bound
  `|beta_{m+1} * (exp(scale * T_m))[m, 1]| * ||psi|| < tol`. Returns
  `(y, m_used)` so callers can log or assert on actual subspace size.
  The existing fixed-`m` `expm_apply` is unchanged so v0.7 / v0.8
  alpha / beta callers keep numeric reproducibility.

The CLI surface gained `scrapbox tpq` (now 7 subcommands:
`run / validate / sweep / bench / doctor / ed / tpq`). Schema remains
0.2 (additive `[tpq]` section is `#[serde(default)]`).

## Config-key surface (delta from v0.7)

New optional section:

```toml
[tpq]
kind = "density" | "work_statistics"
source = "ed" | "matrix_free"
n_samples = 500
seed = 7
beta = 2.0       # optional; defaults to hamiltonian.beta
krylov_m = 30    # optional; matrix_free only; default 30
```

`work_statistics` requires `[quench]` as well. No other key changes.

## Acceptance gates

- **alpha.1 - tpq_work_matrix_free zero quench at L=4**: H_init==H_final
  gives `<W>` and `sigma_W^2` within 1e-9 of zero.
- **alpha.2 - tpq_work_matrix_free vs ED-path at L=4 sudden quench**:
  600 samples, comb-V v0=0.3; `<W>` within 0.05, `sigma_W^2` within 0.15.
- **alpha.3 - determinism**: same seed -> identical output.

- **beta.1 - tpq CLI density matrix-free vs ed at L=4**: agree within
  0.1 per site at 500 samples.
- **beta.2 - tpq CLI work matrix-free vs ed at L=4**: agree on `<W>`
  within 0.05, `sigma_W^2` within 0.15.

- **gamma.1 - expm_apply_adaptive matches fixed at L=5 SPD**: with
  tol=1e-10 the adaptive output matches fixed-m=5 to 1e-9.
- **gamma.2 - early termination monotonicity**: `m_used(loose) <=
  m_used(tight)` for any pair of tolerances.
- **gamma.3 - zero input -> m_used=0**: handled as the trivial case.

## Tests at tag time

*(Snapshot - authoritative source is `cargo test --release`.)*

- 183 tests (157 unit + 26 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.9+)

- **Exact `Theta_2`**: full Palamara Sec III.3 derivation (beyond the
  v0.5 alpha LDA placeholder). Tracked on #22.
- **True finite-T BALDA**: carryover.
- **Adaptive `krylov_m` from CLI**: `[tpq].krylov_m` is still a fixed
  user knob; wiring `expm_apply_adaptive` through to the dispatcher
  would close this gap.
- **Matrix-free TPQ density complex-Gaussian variant**: the v0.6
  gamma `tpq_density_complex` / `tpq_work_statistics_complex` paths
  do not yet have matrix-free counterparts.
- **Block Krylov for multi-RHS**: would amortise `apply` cost when
  evaluating many quenches against the same `H_init`.

## Source-of-truth references

- `notes/discipline/PHASES.md` - milestone definitions.
- `notes/discipline/ACCEPTANCE.md` - mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) - per-batch log.
- Saad, Y. *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) -
  posteriori residual bound consumed by `expm_apply_adaptive`.
- Sugiura, Shimizu, *Phys. Rev. Lett.* 108, 240401 (2012); 111,
  010401 (2013) - TPQ formulation, consumed by every TPQ path.
- Palamara, Plata, Pekola, Goold, *Phys. Rev. Lett.* 133, 207101
  (2024) - generalized FDR + `Theta_2` definition; v0.8 alpha closes
  the matrix-free part of the work-statistics pipeline this references.
- Jordan, Wigner, *Z. Phys.* 47, 631 (1928) - JW basis underlying
  both `JwHubbard` and `SparseMatrix::from_hubbard`.
