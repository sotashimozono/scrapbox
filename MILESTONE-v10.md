# MILESTONE-v10 — scrapbox v0.10

Status: **shipped 2026-05-24** — minimum-scope gates green.

This is the v0.10 contract. v0.10 is the **TPQ analysis suite
completion + adaptive Krylov CLI** milestone: the v0.9 exact
`Theta_2` reaches both the `scrapbox tpq` subcommand and the
matrix-free `LinearOperator` path, and the v0.8 γ
`expm_apply_adaptive` becomes user-selectable from TOML via the new
`[tpq].krylov_tol` knob.

## What v0.10 adds over v0.9

- **`scrapbox tpq kind = "theta_2"` mode** (PR #45, batch α)
  Extends the v0.8 β `scrapbox tpq` subcommand to a third analysis
  kind. One TOML knob (`[tpq].kind` + `[tpq].source`) now covers
  `density` / `work_statistics` / `theta_2`. Output JSON gains a
  `Theta2 { source, dim, beta, theta_2, wall_time_ms }` tagged variant.

- **Matrix-free `exact_theta_2` via low-K Lanczos** (PR #46, batch β)
  `reference::ed::exact_theta_2_matrix_free<O1, O2>(h_init, h_final,
  beta, k_states, krylov_m) -> f64`. Lanczos computes the K lowest
  eigenpairs of `H_init` matrix-free; the Palamara trace is then
  evaluated truncated to those states. `K = dim` recovers the exact
  ED result to ~1e-7. Closes the `scrapbox tpq` 6-cell matrix
  (3 kinds × 2 sources). Wired through `[tpq].theta_2_k_states`.

- **Adaptive `krylov_tol` from CLI** (PR #47, batch γ)
  `pub enum KrylovSpec { Fixed { m }, Adaptive { tol, max_m } }` plus
  `expm_apply_with_spec` dispatcher. Four matrix-free TPQ entry points
  refactored to take `spec: KrylovSpec` instead of `krylov_m: usize`.
  CLI: `[tpq].krylov_tol` (when `Some`) makes per-sample
  `exp(-β H / 2)` pick the subspace size on the fly via the v0.8 γ
  Saad residual bound; `krylov_m` becomes the upper bound `max_m`.

The CLI surface is unchanged from v0.9 (`run / validate / sweep /
bench / doctor / ed / tpq`). Schema remains `0.2`.

## Config-key surface (delta from v0.9)

`[observables].theta_2.method` is unchanged. `[tpq]` gains:

```toml
[tpq]
kind = "density" | "work_statistics" | "theta_2"   # was: density | work_statistics
source = "ed" | "matrix_free"
theta_2_k_states = 16    # optional; matrix_free + theta_2 only; default 16
krylov_tol = 1.0e-10     # optional; matrix_free only; switches to adaptive Krylov
krylov_m = 30            # fixed m when krylov_tol is None; max_m when krylov_tol is set
```

## Acceptance gates

- **α.1 — `scrapbox tpq` `kind = theta_2` `source = ed` matches `scrapbox run` `theta_2.method = exact`**: per-config `theta_2` values agree to **1e-12** at L=4.
- **α.2 — `source = matrix_free` reservation** (until β): proper `ConfigValidation` error.

- **β.1 — `K = dim` matrix-free recovers ED-path**: `exact_theta_2_matrix_free` at K=36 vs `exact_theta_2` at L=4 agree to **1e-7**.
- **β.2 — K convergence monotonic**: `K=16 error ≤ K=8 error`; K=8 within 50% of exact.
- **β.3 — L=6 smoke**: dim=400 Lanczos low-K runs and returns finite non-negative.
- **β.4 — CLI K=dim parity**: `scrapbox tpq kind=theta_2 source=matrix_free K=36` matches `source=ed` at L=4 to **1e-7**.

- **γ.1 — adaptive matches fixed via CLI**: `scrapbox tpq` with `krylov_tol = 1e-10` produces densities within **1e-6** of `krylov_m = 30` fixed at L=4, same seed.

## Tests at tag time

*(Snapshot — authoritative source is `cargo test --release`.)*

- 206 tests (170 unit + 36 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.11+)

- **True finite-T BALDA** — long-running carryover (since v0.4).
- **Block Krylov for multi-RHS quench sweeps** — perf optimisation.
- **TpqMethod typed enum** — string-match dispatch in `bin_support.rs::compute_observables` could be parsed once at config time.
- **Krylov diagnostics in `tpq_report.json`** — currently the adaptive path drops `m_used`; surfacing it would help users tune `krylov_tol`.
- **Sweep configs for the new `theta_2` mode** — `configs/` lacks a demo of `scrapbox tpq kind = theta_2`.

## Source-of-truth references

- `notes/discipline/PHASES.md` — milestone definitions.
- `notes/discipline/ACCEPTANCE.md` — mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) — per-batch log.
- Palamara, Plata, Pekola, Goold, *Phys. Rev. Lett.* 133, 207101
  (2024) — Sec III.3 exact `Theta_2` consumed by α, β.
- Saad, Y., *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) —
  posteriori residual bound consumed by `KrylovSpec::Adaptive` in γ.
