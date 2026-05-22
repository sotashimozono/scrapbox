//! Layer 7 — parameter-grid sweep runner.
//!
//! Consumes `[sweep]` from a config and runs the cartesian product of
//! `axes` as independent `scrapbox run` calls into per-cell subdirs.
//! Full impl lands in Batch 15.

use crate::config::Config;
use crate::error::{Result, ScrapboxError};

/// Execute a parameter sweep. Placeholder until Batch 15 wires the full
/// cartesian-product runner.
pub fn run(_cfg: &Config) -> Result<()> {
    Err(ScrapboxError::ConfigValidation {
        message: "scrapbox sweep is not implemented yet — Batch 15 of v0.2.".into(),
    })
}
