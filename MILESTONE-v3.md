# MILESTONE-v3 — scrapbox v0.3

Status: **shipped 2026-05-22** — all acceptance gates green.

This is the v0.3 contract extracted from
[`notes/discipline/PHASES.md`](notes/discipline/PHASES.md), with each
gate cross-referenced to the CHANGELOG entries that satisfied it.

## What v0.3 adds over v0.2

- **BALDA xc functional** (Lima, Silva, Capelle PRL 90, 146402, 2003).
  Solves β(u) via the Lieb-Wu Bethe-ansatz integral identity, builds
  `V_HXC^BALDA(n)` piecewise across n = 1 with optional Mott-gap
  smoothing for SCF convergence on inhomogeneous configs.
- **Parallel sweep workers**. `[sweep].parallelism > 1` now dispatches
  per-cell solves through a `rayon::ThreadPoolBuilder`. Output dirs
  remain distinct so per-cell I/O is thread-safe.
- **Independent ED at L=4**. Full 36×36 Hubbard Hamiltonian builder
  with programmatic basis + Jordan-Wigner signs; reusable for L=6/8.
- **`scrapbox bench`**. Wall-clock timing harness with `[bench]` config
  (warmup + measured), emits `bench_report.json`
  (min/median/p95/mean/samples in ms).
- **`scrapbox doctor`**. Parses the config and constructs every layer
  without running SCF; emits `doctor_report.json`. Cheap validation.

The CLI surface now covers all five subcommands defined in HARNESS.md:

```
scrapbox run      <config.toml>
scrapbox validate <config.toml>
scrapbox sweep    <config.toml>
scrapbox bench    <config.toml>
scrapbox doctor   <config.toml>
```

## Config-key surface (deltas from v0.2)

```
[xc_functional]
kind = "hubbard_lda" | "balda" | "non_interacting"
params.mott_gap_smoothing_width = δ   # BALDA only, default 0.02
params.lieb_simpson_intervals   = N   # BALDA only, default 4096
params.beta_max_bisect_iter     = N   # BALDA only, default 80
params.beta_tol                 = ε   # BALDA only, default 1e-13

[sweep]
parallelism = N                       # any usize >= 1

[bench]
warmup   = N                          # default 1
measured = N                          # default 5
```

Schema version remains `0.2`; v0.3 is backward-compatible at the schema
layer (only adds new optional sections and variants).

## Acceptance gates

- **§2.1 — BALDA xc dispatches and converges**:
  Batch 17 (`7b4773a`). 9 lib tests + dimer E2E with Pulay mixing.
- **§2.2 — Parallel sweep produces N cells with shared semantics**:
  Batch 18 (`ba84ccc`). 8 U-axis cells × parallelism = 4.
- **§2.3 — Independent ED at L=4**:
  Batch 19 (`e14dc91`). 36×36 ED matches KS-DFT density at uniform V
  to 1e-10; CDW response symmetry confirmed.
- **§2.4 — `scrapbox bench` emits JSON timing**:
  Batch 20 (`8397899`). `min <= median <= p95` invariant verified.
- **§2.5 — `scrapbox doctor` validates without SCF**:
  Batch 21 (`cde9e88`). Parses + constructs every layer, emits report.

## Tests at tag time

- **63 unit + 16 integration = 79 tests**, all green.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  clean (pedantic + nursery, with file-scoped allows for `similar_names`
  in physics-naming-heavy modules and for `cast_*` lints in the
  benchmark statistics).
- `cargo fmt --check` clean.

## Out of scope (deferred to later milestones)

- **`Θ_2.method = "lda"`** — needs the BALDA homogeneous susceptibility
  and is gated to v0.4.
- **True finite-T BALDA** — the v0.3 functional is the standard "T = 0
  xc" approximation evaluated at the finite-T density.
- **TPQ engine** — typicality (TPQ state) sampler for the canonical
  density. Gated to v0.4+.
- **Lanczos sparse matrix backing** — current Lanczos still consumes a
  dense `Mat<f64>` matrix-vector product; a true sparse backend is v0.4.

## Source-of-truth references

- `notes/discipline/PHASES.md` §v0.3 — milestone definitions.
- `notes/discipline/ACCEPTANCE.md` — mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) — per-batch log.
- Lima, Silva, Capelle, PRL 90, 146402 (2003) — BALDA reference paper.
- Lieb, Wu, PRL 20, 1445 (1968) — Bethe-ansatz exact integral.
