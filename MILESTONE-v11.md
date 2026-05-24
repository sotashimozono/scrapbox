# MILESTONE-v11 — scrapbox v0.11

Status: **shipped 2026-05-24** — minimum-scope gates green.

This is the v0.11 contract. v0.11 is the **diagnostics + CLI hygiene
+ demos** milestone: the adaptive Krylov path (v0.8 γ + v0.10 γ)
gets per-sample `m_used` reporting in `tpq_report.json`, the
`[observables].theta_2.method` string field becomes a typed
`TpqMethod` enum (typo detection at parse time), and the `configs/`
suite gains three demos exercising the matrix-free TPQ paths.

## What v0.11 adds over v0.10

- **`KrylovStats` + `m_used` diagnostic in `tpq_report.json`** (PR #49, batch α)
  `pub struct KrylovStats { min_m, max_m, mean_m: f64 }` + a new
  `expm_apply_with_spec_m` returning `(Vec<f64>, usize)`. The four
  matrix-free TPQ entry points are refactored to return
  `(T, KrylovStats)`; `tpq_report.json` gains `krylov_stats` (skipped
  for ed-path output). `bin/scrapbox.rs` prints
  `[krylov m: min=N, max=N, mean=N.N]` for matrix-free runs.

- **`TpqMethod` typed enum** (PR #50, batch β)
  `[observables].theta_2.method` was `String` since v0.5 α (mirrors
  the v0.10 α `TpqKind` typing). v0.11 β makes it
  `pub enum TpqMethod { Zero (default), Lda, Exact }` so serde
  catches typos like `"exatc"` at `Config::from_toml_str` time
  instead of dispatcher-time. `bin_support.rs` match arms become
  exhaustive; the `other =>` fallback is removed.

- **Matrix-free TPQ + theta_2 demo configs** (PR #51, batch γ)
  Three new `configs/*.toml` exercising `scrapbox tpq` paths the
  v0.7–v0.10 work introduced:
  - `tpq_density_matrix_free_l6.toml` — adaptive Krylov density at
    L=6 (`krylov_tol = 1e-10`); `krylov_stats` surfaces the
    auto-chosen `m`.
  - `tpq_work_matrix_free_quench_l4.toml` — sudden-quench work
    statistics via matrix-free TPQ.
  - `tpq_theta_2_matrix_free_l6.toml` — Palamara III.3 exact
    `Theta_2` via Lanczos low-K (v0.10 β).
  `configs/README.md` gains a "Matrix-free TPQ + theta_2 demos
  (v0.11)" section.

The CLI surface is unchanged from v0.10 (`run / validate / sweep /
bench / doctor / ed / tpq`). Schema remains `0.2`.

## Config-key surface (delta from v0.10)

`[observables].theta_2.method` is now an enum (`zero`/`lda`/`exact`)
rather than a free-form string; serde rejects any other value at parse
time.

No new `[tpq]` keys.

## Acceptance gates

- **α.1 — adaptive `krylov_stats` in JSON**: `scrapbox tpq` with
  `krylov_tol = 1e-10` emits `krylov_stats { min_m, max_m, mean_m }`
  in `tpq_report.json` with `1 <= min_m <= mean_m <= max_m <= krylov_m`.
- **α.2 — fixed `krylov_stats` collapses**: `Fixed { m: N }` gives
  `min_m == max_m == N`.
- **α.3 — JSON output unchanged for ed-path**: `krylov_stats` is
  `Option`-skipped when `source = "ed"`.

- **β.1 — typo rejected at parse**: `theta_2.method = "exatc"` fails
  `Config::from_toml_str` (`config::tests::theta_2_method_unknown_string_rejected_at_parse`).
- **β.2 — three valid values accepted**: `"zero" / "lda" / "exact"`
  parse cleanly (`config::tests::theta_2_method_accepts_zero_lda_exact`).
- **β.3 — Display for `TpqMethod`**: round-trips to original string
  spelling for `doctor` report formatting.

- **γ.1 — three demos parse**: `tests/config_smoke::every_sample_config_parses`
  exercises all three new TOMLs (auto-included).
- **γ.2 — three demos smoke-run on Panza**: each produces a valid
  `tpq_report.json` end-to-end (manual verification at PR time).

## Tests at tag time

*(Snapshot — authoritative source is `cargo test --release`.)*

- 209 tests (172 unit + 37 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.12+)

- **True finite-T BALDA** — long carryover (since v0.4). Dedicated
  research sprint.
- **Block Krylov for multi-RHS quench sweeps** — perf optimisation;
  amortises Lanczos build across samples in batches.
- **Krylov diagnostics for matrix-free `theta_2`** — v0.10 β
  `exact_theta_2_matrix_free` uses Lanczos for eigenstates, not
  `expm_apply`, so it currently emits `krylov_stats: None`. Surfacing
  the Lanczos `krylov_dim` actually consumed would close that
  diagnostic gap.
- **Sweep demos** — `[sweep]` grids over the new matrix-free configs
  would let users compare adaptive `m_used` across parameters.

## Source-of-truth references

- `notes/discipline/PHASES.md` — milestone definitions.
- `notes/discipline/ACCEPTANCE.md` — mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) — per-batch log.
- Saad, Y., *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) —
  posteriori bound consumed by `KrylovSpec::Adaptive`; v0.11 α makes
  the stopping-`m` it picks visible to users.
