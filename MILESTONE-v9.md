# MILESTONE-v9 — scrapbox v0.9

Status: **shipped 2026-05-24** — minimum-scope gates green.

This is the v0.9 contract. v0.9 is the **exact Theta_2 + matrix-free
TPQ polish** milestone: the BALDA-only LDA placeholder Theta_2 from
v0.5 α is superseded by the Palamara 2024 III.3 exact formula
(xc-agnostic, runs on any XC choice), the new exact path is wired
through `scrapbox run` as `theta_2.method = "exact"`, and the
matrix-free TPQ feature matrix is completed with complex-Gaussian
variants.

## What v0.9 adds over v0.8

- **`exact_theta_2` in `reference::ed`** (PR #40, batch α)
  Palamara 2024 III.3 quantum correction = off-diagonal contribution
  to the work variance in `H_init`'s eigenbasis:

  ```text
  Theta_2_exact = (1/Z_init) Σ_n p_n ( <n|W^2|n> - W_nn^2 )
  ```

  Vanishes when `H_init` and `H_final` share an eigenbasis; strictly
  positive otherwise. Library API:
  `reference::ed::exact_theta_2(ed_init, ed_final, beta) -> f64`.

- **`scrapbox run` dispatches `theta_2.method = "exact"`** (PR #41, β)
  New arm in `bin_support.rs` builds `ed_init` and `ed_final` via
  `reference::ed::canonical_thermal` for the current sector and quench
  final potential, then calls `exact_theta_2`. **xc-agnostic** (unlike
  v0.5 α `"lda"` which is BALDA-only). Error message updated to list
  `"zero", "lda", "exact"`.

- **matrix-free complex-Gaussian TPQ variants** (PR #42, γ)
  Closes the TPQ feature matrix:

  ```text
  tpq_density_matrix_free_complex(jw, beta, n, seed, krylov_m)
  tpq_work_statistics_matrix_free_complex<O1, O2>(h_init, h_final,
                                                  beta, n, seed,
                                                  krylov_m)
  ```

  Variance reduction of `~ 1/sqrt(2)` on `mean_w_stderr` in the
  matrix-free regime (χ² with 2 dof vs 1 dof). Two real `Vec`
  workspaces carry re/im amplitudes, no `num_complex` dep added.

The CLI surface is unchanged from v0.8 (`run / validate / sweep /
bench / doctor / ed / tpq`). Schema remains `0.2`.

## Config-key surface (delta from v0.8)

No new keys. The existing `[observables].theta_2.method` enum gains a
third permitted value:

```toml
[observables]
theta_2.method = "zero" | "lda" | "exact"
```

`"lda"` continues to require `xc_functional.kind = "balda"`; `"exact"`
works for any XC choice.

## Acceptance gates

- **α.1 — `exact_theta_2` trivial quench**: `H_init == H_final` gives
  `|Theta_2| < 1e-10` (`exact_theta_2_trivial_quench_is_zero`).
- **α.2 — `exact_theta_2` small quench positive at L=4**: comb-V
  quench `v0 = 0.3` at L=4, β=2, U=4 gives Theta_2 in (0, 1)
  (`exact_theta_2_small_quench_is_positive_at_l4`).
- **α.3 — Theta_2 ≤ full σ_W²**: cross-check that Theta_2 is bounded
  by the full work variance, recomputed inline
  (`exact_theta_2_differs_from_diagonal_part_at_l4`).
- **α.4 — U=0 trivial quench**: degenerate case still gives 0
  (`exact_theta_2_zero_u_zero_quench_is_zero`).

- **β.1 — exact tightens FDR vs LDA at BALDA dimer quench**:
  `|fdr_residual(exact)| <= |fdr_residual(lda)|`
  (`exact_theta_2_tightens_fdr_versus_lda_at_balda_dimer_quench`).
- **β.2 — unknown method errors with updated list**
  (`exact_theta_2_unknown_method_errors`).

- **γ.1 — mf complex density matches ED at L=4**: 500 samples,
  0.05 tolerance.
- **γ.2 — mf complex work matches ED at L=4 sudden quench**:
  600 samples, `<W>` within 0.05, σ_W² within 0.15.
- **γ.3 — mf complex stderr < mf real stderr**: at fixed
  `n_samples = 800`, ratio < 0.85 (asymptote `1/sqrt(2) ≈ 0.71`).
- **γ.4 — mf complex determinism**.

## Tests at tag time

*(Snapshot — authoritative source is `cargo test --release`.)*

- 193 tests (165 unit + 28 integration), all green
- `cargo clippy --release --all-targets -- -D warnings` clean
- `cargo fmt --all -- --check` clean

## Out of scope (deferred to v0.10+)

- **True finite-T BALDA** — long-running carryover.
- **Matrix-free `exact_theta_2`** — current exact path requires full
  ED of `H_init` and `H_final` to extract eigenstates and diagonal
  matrix elements. A matrix-free version would need either (a) low-
  temperature truncation to a few Lanczos states, or (b) a stochastic
  estimator similar to TPQ.
- **Adaptive `krylov_m` from CLI** — `[tpq].krylov_m` is still a fixed
  user knob; auto-selection via v0.8 γ `expm_apply_adaptive` bound
  would close this gap.
- **Block Krylov for multi-RHS quench sweeps** — perf optimisation.
- **`scrapbox tpq` Theta_2 mode** — current modes: density,
  work_statistics; adding Theta_2 would complete the analysis suite.

## Source-of-truth references

- `notes/discipline/PHASES.md` — milestone definitions.
- `notes/discipline/ACCEPTANCE.md` — mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) — per-batch log.
- Palamara, Plata, Pekola, Goold, *Phys. Rev. Lett.* 133, 207101
  (2024) — Sec III.3 exact `Theta_2` derivation consumed by α and β;
  closes the LDA placeholder from v0.5 α.
- Sugiura, Shimizu, *Phys. Rev. Lett.* 108, 240401 (2012); 111,
  010401 (2013) — TPQ formulation; γ extends to the matrix-free +
  complex-Gaussian path.
