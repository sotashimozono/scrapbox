# MILESTONE-v15 — scrapbox v0.15

Status: **shipped 2026-05-25** — minimum-scope gates green.

This is the v0.15 contract. v0.15 is the **sweep-CLI maturity +
finite-T BALDA shape + true block Lanczos** milestone: the
`scrapbox tpq` sweep harness reaches 10 supported
`(axis, kind, source)` tuples (up from v0.14's 5), BALDA-finite-T
gets a U-aware shape, and a true block Krylov primitive
`expm_apply_block_shared` lands alongside the v0.14 β per-RHS
variant. CI grows a dedicated sweep job that runs every
`configs/sweeps/*` config on push and uploads the resulting reports
as artifacts — the first piece of GitHub-compute-as-research-runner
infrastructure.

## What v0.15 adds over v0.14

- **Sweep + `theta_2` kind** (PR #68, batch α)
  Adds `TpqSweepKind::Theta2` and matches `(beta, theta_2, ed)`
  and `(beta, theta_2, matrix_free)` in the dispatcher. First
  sweep `kind` beyond `density` and `work_statistics`. Output
  variant `Theta2Sweep` with per-beta `theta_2_exact` rows;
  matrix-free path carries KrylovStats.

- **Multi-axis cartesian sweep** (PR #69, batch β)
  Adds `TpqSweep::Cartesian { axes: [_; 2], values: [_; 2] }`. Two
  supported tuples `(beta × seed, density, matrix_free)` and
  `(seed × beta, density, matrix_free)` execute the full cartesian
  product as `points = |axis_0| × |axis_1|` rows. Each row carries
  both axis values plus the density vector.

- **Work-statistics seed ensemble sweep** (PR #70, batch γ)
  Adds `(seed, work_statistics, matrix_free)` tuple. Per-seed
  rows plus a scalar `ensemble_summary { mean_w_mean,
  mean_w_stderr_across_seeds }` distinguishing the
  **within-seed** stderr (from TPQ sample fluctuation, already
  on each row) from the **across-seed** stderr (dispersion of
  the within-seed mean estimate across the seed list, N-1
  denominator).

- **KST-inspired κ(n,T) BALDA-FT shape** (PR #71, batch δ)
  Replaces v0.14 α's `c · T² · n²` Sommerfeld correction with a
  Karasiev-Sjostrom-Trickey-inspired form that incorporates the
  interaction strength:

  ```
  v^{BaldaFiniteT}_i = v^{Balda}_i − c · T² · (U / t) · g(n_i),
      g(n) = n² / (1 + n)
  ```

  `g(n)` saturates at `n → 2` instead of growing quadratically
  (closer to the bounded shape KST 2014 proposes for the
  homogeneous electron gas), and the explicit `U / t` factor makes
  the leading thermal correction scale with the interaction scale
  the Mott gap responds to. Still a placeholder, not a derived
  BA-LDA thermal evaluator; true Lieb-Wu thermal integration
  remains deferred to v0.16+.

- **True block Lanczos with shared subspace** (PR #72, batch ε)
  Adds `expm_apply_block_shared` alongside v0.14 β's
  `expm_apply_block`. Builds one shared block Krylov subspace
  `span{ M, AM, A²M, ..., A^{m-1}M }` from the RHS block
  `M = [psi_1 | … | psi_N]` via block Lanczos with full block
  re-orthogonalisation, then evaluates `exp(scale · A) M` against
  the resulting `mN × mN` block tridiagonal. Strictly contains
  the union of per-RHS subspaces, so it captures the same
  accuracy with smaller `m` when RHSs are directionally close
  (nearby TPQ samples around the same beta). v0.15 ε ships the
  primitive only; consumer wiring + a benchmark-driven heuristic
  for when to switch deferred to v0.16+.

- **CI sweep workflow** (this PR, ζ closure)
  New `.github/workflows/sweep.yml` runs every `configs/sweeps/*`
  config on push to `main` (and on `workflow_dispatch`), uploading
  the generated `tpq_sweep_report.json` files as a single artifact.
  Turns the existing free GitHub-hosted runners into a
  research-compute backbone for parameter-sweep experiments. No
  scrapbox code changes — pure infrastructure.

## Dispatch surface summary (cumulative through v0.15)

| Trigger | Body |
|---|---|
| `[xc_functional].kind = "balda_finite_t"` | T=0 BALDA + KST-inspired `c · T² · (U/t) · n² / (1+n)` (v0.15 δ) |
| `LinearOperator::apply_batch(xs, ys)` | per-RHS default; SparseMatrix CSR override (v0.13 δ) |
| `expm_apply_block(op, psis, scale, m)` | N-RHS Lanczos sharing batched matvec (v0.14 β) |
| `expm_apply_block_shared(op, psis, scale, m)` | **single shared block Krylov subspace + `mN × mN` block tridiagonal EVD (v0.15 ε)** |
| `scrapbox tpq` (sweep) | 10 supported `(axis, kind, source)` tuples |

### Sweep tuples supported at v0.15 ζ

1. `(beta, density, matrix_free)`              v0.13 α — subspace reuse
2. `(beta, density, ed)`                       v0.14 γ — per-beta ED
3. `(beta, work_statistics, matrix_free)`      v0.14 γ
4. `(krylov_tol, density, matrix_free)`        v0.14 γ — per-tol adaptive
5. `(seed, density, matrix_free)`              v0.14 δ — ensemble summary
6. `(beta, theta_2, ed)`                       v0.15 α
7. `(beta, theta_2, matrix_free)`              v0.15 α
8. `(beta × seed, density, matrix_free)`       v0.15 β — cartesian
9. `(seed × beta, density, matrix_free)`       v0.15 β — cartesian
10. `(seed, work_statistics, matrix_free)`     v0.15 γ — ensemble work stats

## Acceptance gates (executed on Panza, 2026-05-25)

- **α.1 — theta_2 sweep emits per-beta rows**: `(beta, theta_2, ed)`
  sweep gives N rows of `theta_2_exact`, `(beta, theta_2,
  matrix_free)` adds KrylovStats with `m_used` monotone vs
  configured tolerance.
- **β.1 — cartesian sweep**: 2×3 `(beta, seed)` sweep emits
  `points = 6` rows, axis order respected.
- **γ.1 — work ensemble**: 4-seed `(seed, work_statistics,
  matrix_free)` sweep emits ensemble summary with `mean_w_mean`
  finite and `mean_w_stderr_across_seeds > 0`.
- **δ.1 — T-correction observable & U/t-scaling**: per-site
  BALDA-FT correction at `β=2, U=4` lands in `[1e-4, 5e-3]`;
  doubling `U/t` at fixed `β, n` doubles the correction magnitude
  exactly (unit test `balda_finite_t_correction_scales_with_u_over_t`).
- **ε.1 — N=1 scalar reduction**: `expm_apply_block_shared` with
  N=1 matches `expm_apply` to `< 1e-12` (bit-exact within
  Lanczos arithmetic).
- **ε.2 — N>1 sanity vs dense reference**: n=10 tridiag, N=3
  well-spread RHSs, m=4 shared subspace matches dense `exp(scale·H)
  M` (faer EVD) within `< 1e-6`.
- **ζ.1 — CI sweep workflow runs end-to-end**: `sweep.yml`
  executes the existing `configs/sweeps/*` configs on push, the
  resulting `tpq_sweep_report.json` files are produced and
  uploaded as a single workflow artifact.

## Test breakdown

- **Unit (lib)**: 191 (187 from v0.14 + 1 KST `U/t` scaling
  + 3 block-shared Krylov).
- **Integration**: 51 (46 from v0.14 + 5 sweep extensions
  for α/β/γ).
- **Doctests + binary tests**: 1.
- **Total**: 243 green at v0.15 closure HEAD.

## Live output samples

### Shared block Krylov (ε)

```
expm_apply_block_shared_n3_matches_dense_reference:
  n=10, N=3, m=4 → big = 12 block tridiagonal
  max |shared − dense| < 1e-6
```

### KST BALDA-FT (δ)

```
beta=2 U=4 BALDA-FT correction per site: ~5e-4
beta=2 U=8 BALDA-FT correction per site: ~1e-3  (×2 U/t = ×2 correction)
beta=1e3 BALDA-FT correction per site:    ~5e-10 (T → 0 recovery)
```

### Cartesian sweep (β)

```
scrapbox tpq sweep: kind = density (cartesian), source = matrix_free,
  dim = 36, samples = 30, axes = (beta, seed), cells = 6,
  [krylov m: min=30, max=30, mean=30.0] (wall 211 ms)
  -> runs/tpq_sweep_e2e_cartesian
```

## Deferred to v0.16+

- **True Lieb-Wu thermal BALDA**: KST shape is U-aware and bounded
  but still empirical; needs Bethe-ansatz thermal integration.
- **Consumer wiring for `expm_apply_block_shared`**: ε ships the
  primitive but no production caller switches over yet; need
  benchmark-driven heuristic on `(N, m, matvec cost, n_rhs
  similarity)` to choose between the two `expm_apply_block_*`
  variants per call site.
- **GPU dense EVD path**: candle/burn integration for the `mN × mN`
  block tridiagonal EVD inside `expm_apply_block_shared`, useful
  once `mN` grows past ~1k.
- **`scrapbox tpq` sweep**: ensemble-stat layer for `theta_2`
  similar to v0.15 γ's work-statistics summary; bundle of densely
  spaced beta sweeps for finite-T phase scans.
- **CI sweep workflow**: extend artifact retention, add a job
  matrix for parallelism, post-process artifacts into a
  comparison plot uploaded back to the run.

## References

- Karasiev, Sjostrom, Trickey, *Finite-temperature orbital-free
  DFT molecular dynamics: coupling Profess and Quantum Espresso*,
  PRB 88 161108(R) (2013) — finite-T LDA pattern that the v0.15 δ
  shape reuses qualitatively (with the `n²/(1+n)` reduction
  factor borrowed from the strong-coupling bounded form).
- Lima, Silva, Capelle, *Density functionals not based on the
  electron gas: local-density approximation for a Luttinger
  liquid*, PRL 90 146402 (2003) — T=0 BALDA reused as the v0.15 δ
  base evaluator.
- Saad, *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) —
  per-RHS Lanczos exponentiation reference.
- Golub & Underwood, *The block Lanczos method for computing
  eigenvalues*, in *Mathematical Software III* (1977) — original
  block Lanczos with shared subspace iteration; v0.15 ε
  adapts the construction to `exp(scale·H) M`.
- Palamara, *Lattice density functional theory at finite
  temperature with strongly density-dependent exchange-correlation
  potentials*, PRB 109 (2024) — `theta_2` exact / matrix-free
  contract that v0.15 α exposes via the sweep kind.
