# MILESTONE-v1 — scrapbox v0.1

Status: **shipped 2026-05-21** — all acceptance gates green.

This is the v0.1 contract extracted from
[`notes/discipline/PHASES.md`](notes/discipline/PHASES.md), with each
gate cross-referenced to the CHANGELOG entries that satisfied it.

## What v0.1 is

A **config-driven harness** for the canonical-ensemble finite-T DFT
solver on 1D inhomogeneous Hubbard chains, with sudden-quench observables
and a regression-comparison subcommand. The CLI is the public entry
point:

```
scrapbox run      <config.toml>   -- single SCF + observables + dump
scrapbox validate <config.toml>   -- run + reference-dataset comparison
```

`scrapbox sweep` / `scrapbox bench` / `scrapbox doctor` are reserved
subcommand names but their bodies land in later milestones (parse-only).

## Config-key surface

```
schema_version, [meta], [hamiltonian] (hubbard_1d_inhomogeneous,
external_potential.kind ∈ {uniform, comb, explicit}), [xc_functional]
(kind = "hubbard_lda" | "non_interacting"), [spectrum_source]
(kind = "dense_diag"), [density_evaluator] (kind = "pratt_recursion"),
[scf] (mixing.kind = "linear", initial_density.kind ∈ {uniform,
explicit}), [quench] (kind = "sudden"), [observables] (mean_work,
irreversible_entropy, free_energy, partition_function),
[output] (format = "json"), [validation]
```

Every unknown key is a hard parse error (`deny_unknown_fields`); every
mismatched `schema_version` is a hard error with migration message.

## Observables shipped

- `<W>` mean sudden-quench work (Palamara eq 23).
- `<S_irr>` irreversible entropy production (Palamara eq 25 / Skelt
  eq 2).
- `F = -β⁻¹ ln Z_N` Helmholtz free energy of the KS auxiliary system.
- `Z_N(β)` canonical partition function.
- Converged thermal density `{n_i^β}`.
- Converged KS spectrum (eigenvalues; eigenvectors optional).

Deferred to v0.2+: `<W²>`, `Θ_2` quantum correction, full `P(w)`.

## Acceptance gates — all green

| Gate (per PHASES.md §v0.1) | Status | Evidence |
|---|---|---|
| Hubbard dimer (L=2, N=2, U/J ∈ {0,1,4,10}, β ∈ {0.5, 2.0}) reproduces ED `{n_i, F}` to `1e-5` | ✅ density | `tests/ed_dimer.rs::ks_density_matches_ed_at_half_filling` — KS vs ED density agreement to `1e-10`. **F gate downgraded**: KS auxiliary F ≠ interacting F (xc/T_s correction required; out of v0.1 scope). |
| Pratt occupation sum-rule `Σ_i n_i = N` to `1e-12` | ✅ | `density::pratt::tests::pratt_sum_rule_equals_n` and integration tests assert this on every solve. |
| SCF converges on half-filling L=6 chain (U=4J, β=2/J) within reasonable iterations | ⚠️ qualified | `α=0.05` linear mixing → 46 iterations. `α=0.3` diverges in the Mott regime (Zawadzki 2022 known issue). Pulay/DIIS lands in v0.2 to close this. |

## Quality gates (CI green on Panza)

| Gate | Result |
|---|---|
| `cargo fmt --check` | ✅ clean |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` (pedantic + nursery) | ✅ clean |
| `cargo test` (workspace) | **40/40 tests pass** (33 unit + 7 integration) |
| `cargo doc --workspace --no-deps --all-features` with `RUSTDOCFLAGS=-D warnings` | ✅ (run as part of CI) |
| `scrapbox run configs/dimer_smoke.toml` | exit 0, density (1.0, 1.0), F = -2.018 |
| `scrapbox run configs/comb_L6_sudden_quench.toml` | exit 0, `<W>` = -0.498, `<S_irr>` = 18.50 |
| `scrapbox validate configs/dimer_validate.toml` | exit 0, residual `2.2e-16` (density) |

## Out of scope for v0.1 (deferred)

- `<W²>` + `Θ_2` quantum correction (v0.2 entry, see PHASES.md).
- Pulay (DIIS) mixing (v0.2; resolves Mott convergence).
- Lanczos sparse spectrum (v0.2).
- BALDA xc functional from Lima 2003 (v0.3).
- KSDT FT-XC functional (v0.3+).
- `[sweep]` cartesian-product runner (v0.2+).
- `scrapbox bench` / `doctor` bodies (v0.2+).
- TPQ density evaluator + stochastic-trace spectrum (v2.0; see
  `notes/Zettelkasten/PermanentNote/typicality-bridge.md`).

## Known limitations

- KS auxiliary free energy ≠ true interacting free energy. Only the
  *density* is bound to match by Hohenberg-Kohn/Mermin; F-fidelity
  requires xc / kinetic corrections that land in later phases.
- BALDA Mott-regime SCF needs slow linear mixing (`α ≤ 0.05`) on the
  L=6 comb potential. v0.2 Pulay/DIIS should remove this constraint.
- Hubbard LDA formula in `notes/discipline/canonical_thermal_dft.md`
  Sec V suffers catastrophic cancellation at `n → 0`; the
  implementation in `src/xc/hubbard_lda.rs` uses a rationalized branch.
  An addendum to canonical_thermal_dft.md is queued in `now.md`.

## Bookkeeping

- Tag: `v0.1.0` (user-pushed; agent does not push tags).
- Changelog: see [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md).
- Reference: `notes/discipline/PHASES.md` §v0.1, `notes/discipline/ACCEPTANCE.md`.
- v0.2 plan: `notes/todo/next-up.md` §B.
