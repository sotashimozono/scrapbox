# MILESTONE-v14 — scrapbox v0.14

Status: **shipped 2026-05-25** — minimum-scope gates green.

This is the v0.14 contract. v0.14 is the **dispatch-bodies +
sweep-broadening** milestone: each of v0.13's four dispatch surfaces
gets either a real body or a new consumer, and the `scrapbox tpq`
sweep CLI grows from one supported `(axis, kind, source)` tuple to
five.

## What v0.14 adds over v0.13

- **Leading-T² body for BALDA finite-T** (PR #62, batch α)
  Replaces the v0.13 β placeholder (identical to T=0 BALDA) with
  a Sommerfeld-style additive correction
  `v^{BaldaFiniteT}_i = v^{Balda}_i - c · T² · n_i²` (`c = 0.02`).
  The `n_i²` factor (not the entropy-density-like `n_i (2-n_i)`)
  is asymmetric around `n = 1`, so the canonical Hubbard dimer
  actually sees the T-correction instead of absorbing it as a
  chemical-potential zero-point — an earlier draft using `n(2-n)`
  produced zero density shift on the dimer.

  Still a placeholder pending true Lieb-Wu thermal evaluation, but
  now with observable T-dependence: `1e-5 < max-site-delta < 1e-2`
  vs `balda` at the dimer comb config, and recovers `balda` to
  `< 1e-6` at `β = 1e3` (low-T limit).

- **`expm_apply_block` consuming `LinearOperator::apply_batch`** (PR #63, batch β)
  Wires the v0.13 δ `apply_batch` primitive into a usable consumer:
  `expm_apply_block(op, psis, scale, m) -> Vec<Vec<f64>>` builds N
  independent Lanczos subspaces with each step's matvec dispatched
  through `op.apply_batch` so `SparseMatrix` CSR rows load once per
  step regardless of `psis.len()`. Bit-exact with per-RHS
  `expm_apply` (same arithmetic order).

  Scope note: "block" in the batched-matvec sense only; subspaces
  are not coupled. True block Lanczos with shared subspaces stays
  deferred.

- **Sweep CLI broadening to five `(axis, kind, source)` combos** (PR #64, batch γ)
  `scrapbox tpq` sweep dispatch grows from v0.13 α's single
  `(beta, density, matrix_free)` tuple:

  - `(beta, density, matrix_free)`  — v0.13 α; subspace reuse
  - `(beta, density, ed)`           — v0.14 γ; per-beta ED
  - `(beta, work_statistics, matrix_free)` — v0.14 γ
  - `(krylov_tol, density, matrix_free)` — v0.14 γ; per-tol adaptive
  - `(seed, density, matrix_free)` — v0.14 δ; ensemble summary

  New `TpqSweepAxis::KrylovTol` and `Seed` variants. New output
  variants `DensityKrylovTol`, `WorkStatistics`, `DensitySeedSweep`.

- **Ensemble seed sweep + per-site stderr** (PR #65, batch δ)
  Fifth combo above. Emits one density row per seed plus an
  `ensemble_summary { mean_density, stderr_density }` block
  computing per-site mean and stderr of the mean (sample stddev /
  sqrt(N), N-1 denominator). First in-binary ensemble-statistics
  layer for TPQ.

## Dispatch surface summary (cumulative through v0.14)

| Trigger | Body |
|---|---|
| `[xc_functional].kind = "balda_finite_t"` | T=0 BALDA + leading-T² Sommerfeld correction (v0.14 α) |
| `LinearOperator::apply_batch(xs, ys)` | per-RHS default; SparseMatrix CSR override (v0.13 δ) |
| `expm_apply_block(op, psis, scale, m)` | N-RHS Lanczos sharing batched matvec (v0.14 β) |
| `scrapbox tpq` (sweep, 5 supported combos) | run_sweep dispatcher (v0.13 α + v0.14 γ + v0.14 δ) |

## Live output samples

### BALDA-FT T-correction (α)

```
beta=2 BALDA-FT density:    [0.946..., 1.053...]
beta=2 plain BALDA density: [0.944..., 1.055...]
max delta ~ 2e-3 (T-correction observable, SCF stable)
```

### Sweep output (δ)

```
scrapbox tpq sweep: kind = density (ensemble), source = matrix_free,
  dim = 36, samples = 30, axis = seed, points = 4,
  [krylov m: min=30, max=30, mean=30.0] (wall 73 ms)
  -> runs/tpq_sweep_e2e_seed_ens
```

## Acceptance gates (executed on Panza, 2026-05-25)

- **α.1 — T-correction observable on dimer**: `balda_finite_t` SCF
  density at `β=2, U=4` differs from `balda` by `1e-5 < max
  delta < 1e-2`.
- **α.2 — Low-T limit recovers BALDA**: at `β=1e3` the correction is
  `~5e-9` per site; SCF densities agree to `< 1e-6`.
- **β.1 — Block matches per-RHS bit-exactly on sparse**: 3 RHS on
  5x5 sparse, `expm_apply_block` matches per-RHS `expm_apply` to
  `< 1e-12`.
- **β.2 — Default impl works on dense + edge cases covered**:
  empty batch returns empty; zero-norm RHS produces zero output
  without breaking the nonzero RHS.
- **γ.1 — ed-path sweep emits per-beta rows**: 3-beta `source=ed`
  sweep gives 3 rows with `krylov_stats = {0,0,0.0}`.
- **γ.2 — work_statistics sweep emits work rows**: 2-beta sweep
  emits rows with `mean_w`, `work_variance >= 0`, `mean_w_stderr
  >= 0`.
- **γ.3 — krylov_tol sweep per-row stats**: 3-tol sweep gives
  `m_used` monotone in tol (tighter tol uses larger m).
- **δ.1 — seed-sweep ensemble summary**: 4-seed sweep at L=4 emits
  rows in seed order, `mean_density` sums to N=4 (canonical
  conservation), per-site `stderr_density > 0`.

## Test breakdown

- **Unit (lib)**: 187 (178 from v0.13 + 5 BALDA-FT
  Sommerfeld + 4 block Krylov).
- **Integration**: 46 (42 from v0.13 + 3 sweep broadening + 1
  seed sweep).
- **Total**: 233 green at v0.14 closure HEAD.

## Deferred to v0.15+

- True Lieb-Wu thermal BALDA (v0.14 α placeholder still in flight,
  but the `c·T²·n²` shape is empirical not derived).
- True block Lanczos with shared subspaces (v0.14 β does batched
  matvec but per-RHS subspaces).
- Sweep CLI: `kind = "theta_2"`, multi-axis cartesian sweep,
  batch-statistics for work_statistics across seeds.
- Adaptive Lanczos extended beyond matrix-free `theta_2` to
  ed-path observables.

## References

- Karasiev, Sjostrom, Trickey, *Finite-temperature orbital-free
  DFT molecular dynamics: coupling Profess and Quantum Espresso*,
  PRB 88 161108(R) (2013) — finite-T LDA pattern that the v0.14 α
  shape qualitatively resembles.
- Lima, Silva, Capelle, *Density functionals not based on the
  electron gas: local-density approximation for a Luttinger
  liquid*, PRL 90 146402 (2003) — T=0 BALDA reused as the v0.14 α
  base evaluator.
- Saad, *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) —
  underpins `expm_apply_adaptive` and `expm_apply_block` Lanczos
  re-orthogonalisation patterns.
