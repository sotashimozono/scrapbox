# Configs

Each `*.toml` here is a self-contained recipe for `scrapbox`. Run via:

```
scrapbox run      <config.toml>     # single SCF + observables
scrapbox validate <config.toml>     # run + reference-dataset comparison
scrapbox sweep    <config.toml>     # cartesian-product over [sweep].axes
scrapbox bench    <config.toml>     # warmup + measured timing
scrapbox doctor   <config.toml>     # parse + layer-construction check
```

## Smoke / regression

| Config | Subcommand | What it does |
|---|---|---|
| `dimer_smoke.toml` | `run` / `doctor` | minimal `L=2` smoke test (1 SCF iter at half-filling; `doctor` exercises parse + dispatch only) |
| `dimer_validate.toml` | `validate` | dimer with `[validation]` reference comparison |
| `dimer_bench.toml` | `bench` | wall-clock baseline (~0.4 ms / solve on a dev box) |

## Pipeline coverage

| Config | Layer exercised |
|---|---|
| `dimer_lanczos.toml` | `SpectrumSource::Lanczos` vs `DenseDiag` |
| `dimer_gce_projection.toml` | `DensityEvaluator::GcePlusProjection` vs `PrattRecursion` |
| `dimer_balda.toml` | `XcFunctional::Balda` with Mott-gap smoothing |
| `dimer_uniform_quench.toml` | sudden quench, machine-precision FDR closure (commuting case) |
| `dimer_sweep_u.toml` | `sweep` mode, sequential |
| `dimer_sweep_parallel.toml` | `sweep` mode, `parallelism = 4` |

## v0.4 demos (physics narratives)

Each demo is a self-contained "what does interaction / boundary / quench do here?" story.

### `mott_crossover_l4.toml` (sweep)

`L = 4` half-filling with a small comb `V = (+0.05, -0.05, +0.05, -0.05)`, sweep `U` from 0 to 3 (the bare HubbardLDA SCF diverges in the Mott regime; deep-Mott demos await BALDA xc). As `U` crosses the bandwidth `~4t`, the Mott gap suppresses charge response: `|n_i - 1|` shrinks roughly as `|V|/U` for `U >> t`. Run:

```
scrapbox sweep configs/mott_crossover_l4.toml
```

Then inspect `runs/mott_crossover_l4/U_*/density.json` to see the density modulation decay.

### `cdw_response_l6.toml` (sweep)

`L = 6` half-filling, fixed `U = 4`, sweep staggered amplitude `v` over `[0, 0.05, 0.1, 0.2, 0.4]`. The CDW order parameter `m_CDW = (1/L) Sum_i (-1)^i n_i` grows from zero in the small-`v` linear-response regime to a saturating value at large `v`.

```
scrapbox sweep configs/cdw_response_l6.toml
```

### `quench_w_squared_l4.toml` (run)

`L = 4`, sudden quench from uniform `V = 0` to staggered `V' = (+0.05, -0.05, +0.05, -0.05)`. Because `[H_initial, H_final] != 0`, the quench is **non-commuting** and the classical FDR `<S_irr> = (beta^2/2) sigma_w^2` has a non-trivial residual that the quantum correction `Theta_2` is supposed to close. With `theta_2.method = "zero"` the residual is what an LDA-corrected `Theta_2` would need to match.

```
scrapbox run configs/quench_w_squared_l4.toml
cat runs/quench_w_squared_l4/observables.json
```

Compare with the commuting `dimer_uniform_quench.toml` where the residual closes to machine precision.

## Cross-checks against exact references

Tests in `tests/` use the `reference::` module as ground truth:

- `tests/ed_dimer.rs` - L=2 KS-DFT vs analytic dimer (`reference::dimer`) + generic ED (`reference::ed`)
- `tests/ks_ed_consistency.rs` - L=4 KS-DFT vs ED at uniform V (machine precision) and comb V (LDA error budget)
- `tests/ks_ed_l6.rs` - same at L=6 (400-state Hilbert)

## Adding a config

1. Copy the closest existing file as a template.
2. Update `[meta]` (name, description, created, tags).
3. Pick `[scf].mixing` carefully - Pulay (`alpha=0.2-0.3`, `history_depth=6-8`) works on smooth potentials; large-U or Mott-regime needs linear (`alpha=0.05`, `max_iterations=2000+`).
4. Run `scrapbox doctor configs/your.toml` first - catches schema mismatches before any SCF cost.

## Matrix-free TPQ + theta_2 demos (v0.11)

Three demo configs showing the v0.7-v0.10 matrix-free TPQ machinery exposed through `scrapbox tpq`. Each writes `tpq_report.json` in its own `runs/` directory.

### `tpq_density_matrix_free_l6.toml`

`L = 6`, dim = 400, half-filling. Drives `tpq_density_matrix_free` via Krylov `exp(-beta H/2)` on `JwHubbard`. Adaptive Krylov (`krylov_tol = 1e-10`, `krylov_m = 60` as upper bound) — `tpq_report.json` includes `krylov_stats` showing the per-sample `m` spread.

```
scrapbox tpq configs/tpq_density_matrix_free_l6.toml
cat runs/tpq_density_matrix_free_l6/tpq_report.json
```

### `tpq_work_matrix_free_quench_l4.toml`

`L = 4`, sudden quench `v_init = 0 -> v_final = (+0.3, -0.3, +0.3, -0.3)`. Drives `tpq_work_statistics_matrix_free` (v0.8 alpha). Fixed `krylov_m = 30`, `n_samples = 500` — gives `<W>` and `sigma_W^2` plus `mean_w_stderr` for FDR diagnostics.

```
scrapbox tpq configs/tpq_work_matrix_free_quench_l4.toml
```

### `tpq_theta_2_matrix_free_l6.toml`

`L = 6`, Palamara III.3 exact `Theta_2` via Lanczos low-K truncation (v0.10 beta). `theta_2_k_states = 16` keeps only the 16 lowest eigenpairs of `H_init`, sufficient at `beta = 2.0` where the thermal weight concentrates on low-energy states.

```
scrapbox tpq configs/tpq_theta_2_matrix_free_l6.toml
```


## Adaptive Krylov m sweeps (v0.12)

`configs/sweeps/` holds bundles of TPQ matrix-free runs that hold all parameters fixed except one, exposing how the adaptive Krylov subspace size `m` (Saad 1992 stopping) responds to that knob. `scripts/analyze_adaptive_m.py` reads the resulting `tpq_report.json` files and prints `min_m / max_m / mean_m / wall_ms` per row.

### Beta sweep (`configs/sweeps/beta_*.toml`)

L = 6, dim = 400, fixed `krylov_m = 80` cap, `krylov_tol = 1e-10`, `n_samples = 100`. Varies `[hamiltonian].beta` over `{0.5, 2.0, 10.0}`:

```
for f in configs/sweeps/beta_*.toml; do scrapbox tpq "$f"; done
python3 scripts/analyze_adaptive_m.py beta 'runs/sweep_adaptive_m_beta_*'
```

Observed:

```
      beta  min_m  max_m   mean_m  wall_ms
--------------------------------------------
       0.5     16     17     16.0       78
       2.0     27     27     27.0      172
      10.0     57     63     59.3     1888
```

Deep-cold (large `beta`) needs ~4x larger Krylov subspace; wall time scales ~24x because the `m^3` tridiagonal EVD dominates at `m ~ 60`.

### Tolerance sweep (`configs/sweeps/tol_*.toml`)

L = 6, dim = 400, fixed `beta = 2.0`, `krylov_m = 80` cap, `n_samples = 100`. Varies `[tpq].krylov_tol` over `{1e-14, 1e-10, 1e-6}`:

```
for f in configs/sweeps/tol_*.toml; do scrapbox tpq "$f"; done
python3 scripts/analyze_adaptive_m.py tol 'runs/sweep_adaptive_m_tol_*'
```

Observed:

```
       tol  min_m  max_m   mean_m  wall_ms
--------------------------------------------
     1e-14     31     38     32.1      213
     1e-10     27     27     27.0      147
     1e-06     21     22     22.0       96
```

Looser `tol` lets adaptive Krylov stop earlier (~20-25%); wall time scales accordingly.
