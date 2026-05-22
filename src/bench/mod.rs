#![allow(unknown_lints, clippy::manual_is_multiple_of)]
#![allow(
    clippy::similar_names,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
//! Layer 7 - wall-clock benchmark runner.
//!
//! Solves the configured pipeline `warmup + measured` times, discarding
//! the warmup phase, and emits a `bench_report.json` with timing stats
//! (min, median, p95, mean, samples in ms).

use std::path::Path;
use std::time::Instant;

use crate::bin_support;
use crate::config::Config;
use crate::error::{Result, ScrapboxError};

/// Execute `[bench]` from the config at `config_path`.
pub fn run(config_path: &Path) -> Result<()> {
    let cfg = Config::from_file(config_path)?;
    let bench = cfg
        .bench
        .clone()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: "[bench] section missing -- scrapbox bench requires it".into(),
        })?;
    if bench.measured == 0 {
        return Err(ScrapboxError::ConfigValidation {
            message: "[bench].measured must be >= 1".into(),
        });
    }

    eprintln!(
        "scrapbox bench: warmup={}, measured={}",
        bench.warmup, bench.measured
    );

    for i in 0..bench.warmup {
        eprintln!("  warmup {}/{}...", i + 1, bench.warmup);
        bin_support::solve_and_write(&cfg)?;
    }

    let mut samples_ms = Vec::with_capacity(bench.measured);
    for i in 0..bench.measured {
        let t0 = Instant::now();
        bin_support::solve_and_write(&cfg)?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  measured {}/{}: {:.3} ms", i + 1, bench.measured, dt_ms);
        samples_ms.push(dt_ms);
    }

    let stats = summarize(&samples_ms);
    eprintln!(
        "  -> min={:.3} ms  median={:.3} ms  p95={:.3} ms  max={:.3} ms  mean={:.3} ms",
        stats.min_ms, stats.median_ms, stats.p95_ms, stats.max_ms, stats.mean_ms
    );

    let out_dir = bin_support::resolve_output_dir(&cfg);
    std::fs::create_dir_all(&out_dir).map_err(|source| ScrapboxError::Artifact {
        path: out_dir.clone(),
        message: format!("create bench output dir: {source}"),
    })?;
    let report_path = out_dir.join("bench_report.json");
    let report_tmp = out_dir.join("bench_report.json.tmp");
    let report = BenchReport {
        warmup: bench.warmup,
        measured: bench.measured,
        samples_ms,
        min_ms: stats.min_ms,
        max_ms: stats.max_ms,
        median_ms: stats.median_ms,
        p95_ms: stats.p95_ms,
        mean_ms: stats.mean_ms,
    };
    // Atomic write: serialise to .tmp then rename, so a crash mid-write
    // cannot leave a partially-formed bench_report.json on disk.
    let file = std::fs::File::create(&report_tmp).map_err(|source| ScrapboxError::Artifact {
        path: report_tmp.clone(),
        message: format!("create bench_report.json.tmp: {source}"),
    })?;
    serde_json::to_writer_pretty(file, &report).map_err(|source| ScrapboxError::Artifact {
        path: report_tmp.clone(),
        message: format!("write bench_report.json.tmp: {source}"),
    })?;
    std::fs::rename(&report_tmp, &report_path).map_err(|source| ScrapboxError::Artifact {
        path: report_path.clone(),
        message: format!("rename bench_report.json.tmp -> .json: {source}"),
    })?;
    eprintln!("scrapbox bench: report -> {}", report_path.display());
    Ok(())
}

#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy)]
struct Stats {
    min_ms: f64,
    max_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
}

fn summarize(samples_ms: &[f64]) -> Stats {
    let mut sorted: Vec<f64> = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN timings"));
    let n = sorted.len();
    let min_ms = sorted[0];
    let max_ms = sorted[n - 1];
    let median_ms = if n % 2 == 0 {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    } else {
        sorted[n / 2]
    };
    // Linear-interpolation p95 (numpy/scipy default). For small n it
    // approaches max_ms; we surface max_ms separately so consumers do
    // not silently mistake p95 = max for a true tail estimate. p95 is
    // only statistically meaningful for roughly n >= 20.
    let p95_ms = if n == 1 {
        sorted[0]
    } else {
        let pos = ((n - 1) as f64) * 0.95;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - (lo as f64);
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    };
    let mean_ms = sorted.iter().sum::<f64>() / (n as f64);
    Stats {
        min_ms,
        max_ms,
        median_ms,
        p95_ms,
        mean_ms,
    }
}

#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, serde::Serialize)]
struct BenchReport {
    warmup: usize,
    measured: usize,
    samples_ms: Vec<f64>,
    min_ms: f64,
    max_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_single_sample_equals_value() {
        let s = summarize(&[42.0]);
        assert!((s.min_ms - 42.0).abs() < 1e-12);
        assert!((s.max_ms - 42.0).abs() < 1e-12);
        assert!((s.median_ms - 42.0).abs() < 1e-12);
        assert!((s.p95_ms - 42.0).abs() < 1e-12);
        assert!((s.mean_ms - 42.0).abs() < 1e-12);
    }

    #[test]
    fn summarize_p95_linear_interpolation_under_max() {
        // For [1, 2, 3, 4, 5] linear-interp p95 = sorted[3] + 0.8*(sorted[4]-sorted[3])
        // = 4 + 0.8*1 = 4.8, distinctly less than max (5.0).
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s.p95_ms - 4.8).abs() < 1e-12, "p95 = {}", s.p95_ms);
        assert!((s.max_ms - 5.0).abs() < 1e-12);
    }

    #[test]
    fn summarize_odd_count() {
        let s = summarize(&[1.0, 3.0, 2.0, 5.0, 4.0]);
        assert!((s.min_ms - 1.0).abs() < 1e-12);
        assert!((s.median_ms - 3.0).abs() < 1e-12);
        assert!((s.mean_ms - 3.0).abs() < 1e-12);
    }

    #[test]
    fn summarize_even_count() {
        let s = summarize(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s.median_ms - 2.5).abs() < 1e-12);
        assert!((s.mean_ms - 2.5).abs() < 1e-12);
    }

    #[test]
    fn summarize_p95_picks_high_tail_via_max_ms() {
        // With linear interpolation p95 is between sorted[3] and sorted[4]
        // and approaches max only for large n. max_ms exposes the true tail.
        let s = summarize(&[1.0, 1.0, 1.0, 1.0, 100.0]);
        assert!((s.max_ms - 100.0).abs() < 1e-12);
        assert!(s.p95_ms > s.median_ms);
        assert!(s.p95_ms < s.max_ms);
    }
}
