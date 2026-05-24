# MILESTONE-v12 — scrapbox v0.12

Status: **shipped 2026-05-24** — minimum-scope gates green.

This is the v0.12 contract. v0.12 is the **diagnostic completeness +
Krylov subspace reuse + adaptive-m observability** milestone: the
`theta_2` matrix-free path catches up to density/work_statistics by
reporting `krylov_stats`, the `expm_apply` core is refactored into a
reusable `KrylovSubspace` so β-sweeps amortise their matvec cost, and
a small sweep harness lets users see how adaptive `m` responds to
`(β, krylov_tol)`. Also unblocks the `typos` CI check that has been
red since v0.11 β.

## What v0.12 adds over v0.11

- **`KrylovStats` for `theta_2` matrix-free + `typos` allowlist** (PR #53, batch α)
  `src/spectrum/lanczos.rs` gets a `diagonalize_with_effective_m`
  wrapper returning `(Eigendecomposition, usize)`. The matrix-free
  `exact_theta_2_matrix_free` return type changes from `f64` to
  `(f64, usize)`; the `scrapbox tpq theta_2 source = matrix_free`
  dispatch captures the `effective_m` and emits
  `krylov_stats { min_m, max_m, mean_m }` with all three equal to the
  single Lanczos run's `effective_m`. Closes the symmetry gap left by
  v0.11 α where density/work emitted stats but theta_2 did not.

  `.typos.toml` allowlist gains `exatc` (intentional fixture from
  v0.11 β parse-time rejection tests) and `Numer` (SIAM J. Numer.
  Anal. journal abbreviation in Saad 1992 citation).

- **`KrylovSubspace` reuse + `expm_apply_multi_scale`** (PR #54, batch β)
  The Lanczos build phase of `expm_apply` becomes a reusable
  `pub struct KrylovSubspace { norm, q_vecs, alphas, betas }` with
  `KrylovSubspace::apply_expm(scale)`. The original `expm_apply`
  collapses to a 1-line shim. New `pub fn expm_apply_multi_scale(op,
  psi, scales, m) -> Vec<Vec<f64>>` builds the subspace once and
  evaluates at each scale for ~free. For a TPQ β sweep at fixed
  `(H, psi)` this turns matvec cost from `O(K * m)` to `O(m)`.

  Scope note: original v0.12 plan listed "block Krylov for multi-RHS
  quench sweeps", but TPQ samples use independent random start
  vectors (no shared Krylov subspace) and quench sweeps vary
  `H_final` (different operator per point). Subspace reuse across β
  at fixed `(H, psi)` is the actually-shared structure, so this
  primitive is shipped. Multi-RHS block Lanczos deferred.

- **Adaptive-m sweep configs + analyzer** (PR #55, batch γ)
  `configs/sweeps/beta_*.toml` (3 files) and `configs/sweeps/tol_*.toml`
  (3 files): TPQ matrix-free density at L=6 dim=400, holding all
  parameters fixed except `[hamiltonian].beta` resp. `[tpq].krylov_tol`.
  `scripts/analyze_adaptive_m.py` (Python 3.8+, no third-party deps)
  reads the resulting `tpq_report.json` files and prints a table of
  `min_m / max_m / mean_m / wall_ms` vs the swept parameter. The
  swept value is parsed out of the run-dir basename, so the analyzer
  has zero TOML dependency.

  `configs/README.md` gets a new section walking through both sweeps
  with the invocation and a captured live output table.

## Live output samples

### theta_2 matrix-free now reports krylov_stats (α)

```
scrapbox tpq: kind = theta_2, source = matrix_free, dim = 36, beta = 2,
  theta_2 = 0.146497 [krylov m: min=16, max=16, mean=16.0] (wall 1 ms)
  -> runs/tpq_theta_2_mf_e2e_stats
```

### Adaptive-m sweep tables (γ)

```
      beta  min_m  max_m   mean_m  wall_ms
--------------------------------------------
       0.5     16     17     16.0       78
       2.0     27     27     27.0      172
      10.0     57     63     59.3     1888

       tol  min_m  max_m   mean_m  wall_ms
--------------------------------------------
     1e-14     31     38     32.1      213
     1e-10     27     27     27.0      147
     1e-06     21     22     22.0       96
```

Deep-cold (large β) needs ~4x larger Krylov subspace; wall scales
~24x because the `m^3` tridiagonal EVD dominates at `m ~ 60`.

## Acceptance gates (executed on Panza, 2026-05-24)

- **α.1 — theta_2 matrix-free emits krylov_stats**: new e2e test
  `tpq_theta_2_matrix_free_emits_krylov_stats_in_json` verifies the
  JSON payload has `krylov_stats { min_m, max_m, mean_m }` with
  `min_m == max_m` and `(mean_m - min_m_as_f64).abs() < 1e-12`.
- **α.2 — typos CI unblocked**: `/tmp/typos . --config ./.typos.toml`
  exits 0 (was failing 9x in a row before this PR).
- **β.1 — subspace reuse matches per-scale**: new lib test
  `expm_apply_multi_scale_matches_per_scale_expm_apply` verifies that
  `expm_apply_multi_scale(op, psi, scales, m)[i]` matches
  `expm_apply(op, scales[i], psi, m)` to `< 1e-12` for 5 scales on a
  5x5 sparse operator.
- **β.2 — boundary cases**: `build_krylov_subspace_dim_clamps_to_op_dim`
  + `build_krylov_subspace_zero_input_yields_empty_alphas` cover the
  m > dim and zero-input edges.
- **γ.1 — sweep configs produce live JSON**: the 6 configs in
  `configs/sweeps/` each emit `runs/sweep_adaptive_m_*/tpq_report.json`
  with `krylov_stats` populated.
- **γ.2 — analyzer reproduces table**: `python3
  scripts/analyze_adaptive_m.py beta 'runs/sweep_adaptive_m_beta_*'`
  prints the β-sweep table above; same for tol.

## Test breakdown

- **Unit (lib)**: 175 (172 from v0.11 + 3 new in v0.12 β:
  `expm_apply_multi_scale_matches_per_scale_expm_apply`,
  `build_krylov_subspace_dim_clamps_to_op_dim`,
  `build_krylov_subspace_zero_input_yields_empty_alphas`).
- **Integration**: 38 (37 from v0.11 + 1 new in v0.12 α:
  `tpq_theta_2_matrix_free_emits_krylov_stats_in_json`).
- **Total**: 213 green.

## Deferred to v0.13+

- True multi-RHS block Lanczos (needs `LinearOperator::apply_batch` +
  block-tridiagonal `T`).
- True finite-T BALDA (carryover from v0.4, now 9 sprints pending).
- Wiring `expm_apply_multi_scale` into a `scrapbox tpq sweep`
  subcommand so beta scans run as one CLI invocation.
- `theta_2` matrix-free `KrylovStats` is degenerate (single Lanczos
  run). When the theta_2 path itself adopts adaptive Lanczos
  diagonalisation, this becomes a meaningful min/max/mean.

## References

- Saad, *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) -
  underpins both `expm_apply_adaptive` (v0.8 γ) and the subspace
  reuse pattern in v0.12 β.
- Palamara et al., *Density functional theory of the quantum
  thermodynamic ensembles*, Phys. Rev. Research (2024) - III.3 still
  drives the matrix-free `theta_2` evaluator threaded with stats in
  v0.12 α.
