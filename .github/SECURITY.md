# Security policy

Thank you for taking the time to report a security issue in `scrapbox`.

## Supported versions

The repository is pre-release (no tagged version yet). All security work
targets the current `main` branch. Once a `v1.0.0` is cut, this section
will be updated to track the supported release line.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive
reports. Instead:

1. Open a **private vulnerability report** via the GitHub Security tab on
   this repository (`Security` → `Report a vulnerability`).
2. Include enough detail to reproduce the issue (commit SHA, config, OS,
   Rust toolchain version).
3. If possible, suggest a remediation; otherwise we will work out a fix
   with you.

We will acknowledge receipt within **7 business days**, and aim to have a
mitigation or remediation plan within **30 days** for high-severity issues.

## Scope

In-scope for this project:

- Vulnerabilities in the `scrapbox` crate and binary.
- Configuration handling that could lead to unintended computation
  (e.g., directory traversal in `[output].directory`).

Out of scope (please report upstream):

- Vulnerabilities in third-party crates (report to the relevant
  maintainer; we will track via `cargo audit`).
- Generic GitHub infrastructure issues.

## Disclosure

By default we coordinate disclosure with you; we will not publish a fix
referencing your report until you confirm timing, unless an actively-
exploited issue forces our hand.
