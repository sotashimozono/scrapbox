# MILESTONE-v2 — scrapbox v0.2

Status: **shipped 2026-05-22** — all acceptance gates green.

This is the v0.2 contract extracted from
[`notes/discipline/PHASES.md`](notes/discipline/PHASES.md), with each
gate cross-referenced to the CHANGELOG entries that satisfied it.

## What v0.2 adds over v0.1

A second-generation harness with:

- **Second moment of work** (`<W²>_c`) via finite-difference
  susceptibility, and the **generalized fluctuation-dissipation
  relation** `<S_irr> = (β²/2)(σ_w² − Θ_2)` (Palamara 2024 eq 31),
  with the `Θ_2 = 0` route enabled for v0.2 and `"lda"` reserved for
  v0.3+ (BALDA).
- **Pulay/DIIS mixing** as an alternative to linear, unlocking the
  BALDA Mott regime where linear mixing requires `α = 0.05` to creep
  to convergence.
- **Lanczos tridiagonalization** as an alternative `SpectrumSource`,
  reproducing the dense pipeline to 1e-10 on the dimer.
- **GCE-plus-projection** density evaluator (fugacity-circle quadrature
  + Fourier projection), as an algorithmic cross-check on the Pratt
  recursion.
- **Sweep mode** (`scrapbox sweep`) — cartesian product over
  `[sweep].axes` with dotted-key TOML overrides and per-cell subdirs.

The CLI surface grows to:

```
scrapbox run      <config.toml>   -- single SCF + observables + dump
scrapbox validate <config.toml>   -- run + reference-dataset comparison
scrapbox sweep    <config.toml>   -- cartesian-product over [sweep].axes
```

`scrapbox bench` and `scrapbox doctor` remain reserved.

## Config-key surface (deltas from v0.1)

```
schema_version = "0.2"

[scf]
mixing.kind = "linear" | "pulay"
mixing.history_depth = N        # Pulay only

[spectrum_source]
kind = "dense_diag" | "lanczos"
krylov_dim = M                  # Lanczos, optional (default = full)
max_iter = ...
tol = ...

[density_evaluator]
kind = "pratt_recursion" | "gce_plus_projection"
params.num_quadrature_points = M  # GCE only, default 64

[observables]
work_variance = bool
theta_2.method = "zero"   # "lda" gated to v0.3+

[sweep]
subdir_template = "..."
parallelism = 1
[[sweep.axes]]
key = "hamiltonian.<dotted.path>"
label = "<short alias>"
values = [a, b, c, ...]
```

Every unknown key remains a hard parse error; `schema_version` mismatch
remains a hard error with a migration message.

## Acceptance gates (per ACCEPTANCE.md)

- **§1.1 — `<W²>` computation lands**:
  Batch 11 (commit `e6ce585`).
- **§1.2 — generalized FDR consistency to 1% on commuting quench**:
  Batch 12 (commit `e072318`), residual at machine precision (~1e-32).
- **§1.3 — Pulay mixing converges BALDA Mott regime**:
  Batch 10 (commit `4880036`), L=6 comb in 84 iter at `α=0.1`, depth 8.
- **§1.4 — Lanczos reproduces dense pipeline**:
  Batch 13 (commit `0e3d749`), dimer F matches to 1e-10.
- **§1.5 — GCE+projection cross-validates Pratt**:
  Batch 14 (commit `c0afc14`), 1e-9 agreement on synthetic spectra.
- **§1.6 — sweep mode produces one subdir per cell**:
  Batch 15 (commit `60096ee`), L=2 dimer × 4 U values verified.

## Tests at tag time

- **51 unit + 10 integration = 61 tests**, all green
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  clean (pedantic + nursery, with file-level `clippy::similar_names`
  allows on two physics-naming-heavy modules)
- `cargo fmt --check` clean

## Out of scope (deferred to later milestones)

- BALDA xc functional and `Θ_2.method = "lda"` — **v0.3**
- Parallel sweep workers (`parallelism > 1`) — **v0.3**
- `scrapbox bench` and `scrapbox doctor` bodies — **v0.3+**
- Independent ED validation beyond the dimer — **v0.3**

## Source-of-truth references

- `notes/discipline/PHASES.md` §v0.2 — milestone definitions
- `notes/discipline/ACCEPTANCE.md` — mechanical gates
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) — per-batch log
- `notes/Zettelkasten/PermanentNote/theta-2-quantum-correction.md` —
  Palamara eq 30 derivation
- `notes/Zettelkasten/LiteratureNote/palamara-2024.md` — eqs 23, 25, 28,
  31 and Sec III.3 / IV.1
