# MILESTONE-v4 - scrapbox v0.4

Status: **shipped 2026-05-24** - minimum-scope gates green.

This is the v0.4 contract. v0.4 was a **consolidation** milestone:
no new solver layers, but the `reference::` module (introduced in v0.3
PR #17) gained a complete coverage matrix, the L=4 / L=6 ED gold is
locked in as cross-check infrastructure, and the demonstration configs
turn the harness into a teaching artefact.

## What v0.4 adds over v0.3

- **L=6 ED coverage** (PR #20, batch γ)
  `reference::ed::canonical_thermal` was already generic over L;
  v0.4 validates and locks in L=6 (400-state Hilbert) with three
  cross-method tests:
  - half-filling V=0 density is 1 (symmetry)
  - U=0 free-energy matches the single-particle Pratt recursion on
    the OBC spectrum (cross-method consistency at the U=0 limit)
  - L=6 KS-vs-ED integration test (`tests/ks_ed_l6.rs`) for both
    uniform V (machine precision) and small comb V (LDA error <2%).

- **Demonstration configs + index** (PR #21, batch δ)
  Three "what does the physics do here" configs and a single
  `configs/README.md` index for all 15 configs:
  - `mott_crossover_l4.toml`: U sweep showing Mott response
    suppression
  - `cdw_response_l6.toml`: staggered V amplitude sweep, CDW
    order parameter growth
  - `quench_w_squared_l4.toml`: non-commuting sudden quench
    surfacing the non-trivial FDR residual

- **Reference module is single source of truth** (PR #19, follow-up
  to v0.3)
  `src/xc/balda.rs` no longer carries its own `lieb_wu_integral`;
  it calls `reference::bethe::lieb_wu_half_filling_energy_with_params`.
  `tests/ed_l4.rs` deleted; its content folded into
  `tests/ks_ed_consistency.rs` and `reference::ed::tests`.

The CLI surface is unchanged from v0.3 (`run / validate / sweep /
bench / doctor`). Schema remains `0.2` (no breaking config changes).

## Config-key surface (no deltas from v0.3)

v0.4 adds no new config keys. Existing sections (sweep, bench, quench,
etc.) work as documented in MILESTONE-v3.

## Acceptance gates

- **γ.1 - L=6 ED Hilbert dim**: `C(6, 3)^2 = 400` (PR #20).
- **γ.2 - L=6 free chain cross-method consistency**: `reference::ed` at
  U=0 matches `reference::free_chain::free_energy` to 1e-9 (PR #20).
- **γ.3 - L=6 KS-vs-ED at uniform V**: machine precision (PR #20).
- **γ.4 - L=6 KS-vs-ED at small comb V**: LDA error < 2% (PR #20).
- **δ.1 - mott crossover sweep**: 6 U cells converge with Pulay (PR #21).
- **δ.2 - cdw response sweep**: 5 v cells converge (PR #21).
- **δ.3 - non-commuting quench**: FDR residual `~ -3.4e-4` recorded
  in `runs/quench_w_squared_l4/observables.json` (PR #21).

## Tests at tag time

*(Snapshot - authoritative source is `cargo test`.)*

- ~134 tests (109 unit + 25 integration), all green
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  clean
- `cargo fmt --check` clean

## Out of scope (deferred to v0.5)

- **α - BALDA `Theta_2 = "lda"` route**: needs research-quality
  derivation of the LDA quantum-correction kernel. Tracked in #22.
- **TPQ engine**: typicality sampler for the canonical density. The
  L=6 ED reference (γ) will give the cross-check anchor when this
  lands.
- **Sparse Lanczos**: current Lanczos still consumes dense `Mat<f64>`
  matvec; a true sparse backend awaits.
- **True finite-T BALDA**: v0.4 keeps the T=0 xc approximation
  evaluated at the finite-T density.

## Source-of-truth references

- `notes/discipline/PHASES.md` - milestone definitions.
- `notes/discipline/ACCEPTANCE.md` - mechanical gates.
- [`notes/todo/CHANGELOG.md`](notes/todo/CHANGELOG.md) - per-batch log.
- Lima, Silva, Capelle, PRL 90, 146402 (2003) - BALDA.
- Lieb, Wu, PRL 20, 1445 (1968) - Bethe-ansatz exact integral.
- Palamara 2024 - generalized FDR and `Theta_2` definition.
