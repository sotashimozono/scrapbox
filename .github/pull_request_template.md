## Description

Related issue / spec: # (if any)

## Type of change

- [ ] ✨ Feature (`enhancement` / `feature`)
- [ ] 🐛 Bug fix (`bug` / `fix`)
- [ ] ⚡ Performance (`performance` / `optimization`)
- [ ] 📖 Documentation (`documentation` / `docs`)
- [ ] 🧰 Refactor / maintenance (`chore` / `refactor`)
- [ ] 💥 Breaking change (`breaking`)

## Changes

<!--
Describe the modified files / modules and the design decisions concisely.
Per discipline/canonical_thermal_dft.md and discipline/CONVENTIONS.md, use
physics-faithful terminology (Kohn-Sham Hamiltonian, Pratt recursion,
thermal density, sudden quench, TPQ, etc.). NO mathematical camouflage.
-->

-

## Discipline checklist

- [ ] Public API names follow [`discipline/CONVENTIONS.md`](../notes/discipline/CONVENTIONS.md) (physics-faithful, no single-letter `pub` fields).
- [ ] Any new config key is documented in [`discipline/CONFIG.md`](../notes/discipline/CONFIG.md) and rejected when missing.
- [ ] Any new `kind = "..."` enum variant has at least one config-driven integration test.
- [ ] Phase scope respected: this PR does not silently land a feature listed as a later milestone in [`discipline/PHASES.md`](../notes/discipline/PHASES.md).
- [ ] Acceptance criteria from [`discipline/ACCEPTANCE.md`](../notes/discipline/ACCEPTANCE.md) are met for any new code path.

## Test plan

- [ ] `cargo fmt --check` clean.
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.
- [ ] `cargo test --workspace --all-features` passes locally.
- [ ] `cargo doc --workspace --no-deps --all-features` clean with `RUSTDOCFLAGS=-D warnings`.
- [ ] Coverage does not regress (Codecov check on the PR).
- [ ] If a reference dataset is touched, `scrapbox validate configs/<benchmark>.toml` still passes.
- [ ] No paper PDF, HTML/CSS/JS report, or other non-tracked file (per `.gitignore`) is accidentally included.

## Notes for reviewer

<!-- Pitfalls, open questions, things to verify. -->
