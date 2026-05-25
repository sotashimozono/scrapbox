# MILESTONE-v13 — scrapbox v0.13

Status: **shipped 2026-05-25** — minimum-scope gates green.

This is the v0.13 contract. v0.13 is the **dispatch-surface
expansion** milestone: four orthogonal capabilities each get
landed as a route through an existing dispatcher, so future PRs
can swap the body of any one of them without touching the
surface.

## What v0.13 adds over v0.12

- **`scrapbox tpq sweep` dispatch via `[tpq.sweep]`** (PR #57, batch α)
  When `[tpq.sweep]` is present in the config, `scrapbox tpq`
  switches to a sweep mode and emits `tpq_sweep_report.json`
  (instead of `tpq_report.json`) with one density row per beta in
  `[tpq.sweep].values`. Single-mode behaviour is unchanged when
  `[tpq.sweep]` is absent. v0.13 alpha scope: `axis = "beta"`,
  `kind = "density"`, `source = "matrix_free"` only — other
  combinations return a specific `ConfigValidation` error.

  Built on top of the v0.12 beta `KrylovSubspace` reuse: per
  random `psi_0`, the Lanczos subspace is built once, then
  evaluated at every beta via cached `(Q, T_m)`. Per-beta single-
  mode cross-check matches to `< 1e-10`.

- **`XcFunctional::BaldaFiniteT` dispatch shim** (PR #58, batch β)
  New `kind = "balda_finite_t"` route in the XC dispatcher. The
  route stores `[hamiltonian].beta` and `U / t` but the evaluator
  body currently delegates verbatim to T=0 BALDA. Scope-honesty:
  the original naive scaling `w(beta * U) * v_xc^{T=0}` breaks
  SCF self-consistency (scaled potential is no longer a
  variational derivative), so v0.13 lands the dispatch surface
  without compromising SCF stability. A future PR can swap in a
  Lieb-Wu thermal evaluator without touching the dispatch
  surface.

- **Adaptive Lanczos for matrix-free `theta_2`** (PR #59, batch γ)
  New `[tpq].theta_2_lanczos_tol` config knob switches the
  matrix-free `theta_2` Lanczos diagonalisation from fixed-`m` to
  an adaptive Ritz-residual stopping criterion. Built on a new
  `lanczos::diagonalize_adaptive` primitive that grows the
  Krylov subspace one step at a time and stops at the first
  `m >= k_states` where the top-`k_states` Ritz residuals are
  all below `tol`. `effective_m` is reported via `KrylovStats`
  as before. Self-consistency: tol = 1e-12 and tol = 1e-6 must
  agree on `Theta_2` to 1e-4 with the looser tol using a smaller
  `m_used`.

- **`LinearOperator::apply_batch` trait method** (PR #60, batch δ)
  New batched matvec entry point with per-RHS default impl
  (backward-compatible). `SparseMatrix` overrides with a CSR-
  batched walk that loads each row's non-zero pattern once for
  all RHS. Bit-exact with per-RHS apply. Primitive ships without
  a consumer; future block Krylov / block Lanczos work slots in
  without a trait change.

## Dispatch surface summary

| Route | Trigger | Body |
|---|---|---|
| `scrapbox tpq` (sweep) | `[tpq.sweep]` populated | `run_sweep` → `tpq_density_matrix_free_beta_sweep` |
| `[xc_functional].kind = "balda_finite_t"` | TOML config | `BaldaFiniteT::evaluate` (delegates to T=0 BALDA in v0.13 β) |
| `[tpq].theta_2_lanczos_tol = X` | TOML config | `exact_theta_2_matrix_free_adaptive` → `diagonalize_adaptive` |
| `LinearOperator::apply_batch(xs, ys)` | Trait method | per-RHS default; SparseMatrix CSR override |

## Live output samples

### Sweep CLI (α)

```
scrapbox tpq sweep: source = matrix_free, dim = 36, samples = 40,
  axis = beta, points = 4, [krylov m: min=30, max=30, mean=30.0]
  (wall 31 ms) -> runs/tpq_sweep_e2e_density
```

### Adaptive theta_2 m_used (γ)

For L=4 dim=36 with `k_states=4`:
- `theta_2_lanczos_tol = 1e-12`, `max_m = 36` → `m_used ~ 15-20` (well under cap)
- `theta_2_lanczos_tol = 1e-6`, `max_m = 36` → `m_used` smaller still
- Both adaptive runs agree on `Theta_2` to `< 1e-4`

## Acceptance gates (executed on Panza, 2026-05-25)

- **α.1 — sweep emits one row per beta**: 4-beta sweep at L=4
  emits `tpq_sweep_report.json` with `len(rows) == 4`, beta order
  preserved, per-row `density.sum() ~ 4` (canonical conservation),
  `krylov_stats` bounded by `krylov_m` cap.
- **α.2 — subspace reuse correctness**: sweep at `[0.5, 2.0]` vs
  two single-mode runs with identical `(seed, n_samples,
  krylov_m)`: per-site density agrees to `< 1e-10`.
- **β.1 — BALDA-FT SCF converges**: `balda_finite_t` route on
  dimer + comb + Pulay/0.1 + `mott_gap_smoothing_width = 0.15`
  converges in ~22 iterations to residual `< 1e-8`. Total
  electron count `== 2` (canonical).
- **β.2 — BALDA-FT placeholder matches T=0 BALDA**: cross-check
  that `balda_finite_t` and `balda` routes produce identical
  densities to `< 1e-6` at the same params in v0.13 placeholder.
- **γ.1 — adaptive stops before cap**: L=4 with `k_states=4`,
  `tol=1e-10`, `max_m=60` gives `m_used < 60`.
- **γ.2 — adaptive self-consistency across tols**: `Theta_2` at
  `tol=1e-12` and `tol=1e-6` agree to `< 1e-4`; looser-tol
  `m_used <= tighter-tol m_used`.
- **δ.1 — apply_batch default vs per-RHS**: `Mat<f64>`
  `apply_batch` produces the same vectors as 3 per-RHS `apply`
  calls (default impl path covered).
- **δ.2 — apply_batch CSR override vs per-RHS**: `SparseMatrix`
  `apply_batch` is bit-exact with 3 per-RHS `apply` calls.
- **δ.3 — apply_batch empty batch is no-op**.

## Test breakdown

- **Unit (lib)**: 178 (175 from v0.12 + 3 new across β/γ/δ).
- **Integration**: 42 (38 from v0.12 + 2 sweep e2e + 2 balda-FT
  e2e + 2 adaptive theta_2 e2e).
- **Total**: 219 green at v0.13 closure HEAD.

## Deferred to v0.14+

- True thermal BALDA (Lieb-Wu thermal integration over the
  Bethe-ansatz solution, or Sommerfeld expansion). Currently the
  `balda_finite_t` evaluator delegates to T=0 BALDA. Carryover
  from v0.4 — now 10 sprints pending.
- Block Krylov / block Lanczos consuming
  `LinearOperator::apply_batch`. Will give density-row sweeps and
  multi-quench-final perf wins.
- Sweep CLI broadening: `kind = "work_statistics"`, `kind =
  "theta_2"`, `source = "ed"`, `axis = "krylov_tol"` etc.
- Adaptive Lanczos extended to `theta_2_lda` ED path, not just
  matrix-free.

## References

- Saad, *Analysis of some Krylov subspace approximations to the
  matrix exponential operator*, SIAM J. Numer. Anal. 29 (1992) —
  underpins the adaptive Lanczos Ritz-residual bound used in γ.
- Lima, Silva, Capelle, *Density functionals not based on the
  electron gas: local-density approximation for a Luttinger
  liquid*, PRL 90 146402 (2003) — T=0 BALDA reused as the
  v0.13 β placeholder body.
- Palamara et al., *Density functional theory of the quantum
  thermodynamic ensembles*, Phys. Rev. Research (2024) — III.3
  drives the matrix-free `theta_2` evaluator that γ makes
  adaptive.
