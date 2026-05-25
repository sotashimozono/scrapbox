#![allow(clippy::doc_markdown)]
//! Rust-side end-to-end test for the v0.15 zeta `.github/workflows/sweep.yml`
//! workflow. Iterates `configs/sweeps/*.toml`, invokes
//! `scrapbox tpq <cfg>` for each, and asserts that one of
//! `tpq_report.json` or `tpq_sweep_report.json` lands under the
//! expected `runs/<meta.name>/` directory.
//!
//! The CI workflow is the user-visible artifact; this test keeps the
//! existing sweep configs honest at PR time. If a config breaks
//! (renamed field, dropped variant, etc.), this test fires before
//! the workflow does on `push: main`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

fn invoke(config_path: &Path) -> std::process::ExitStatus {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "tpq",
            config_path.to_str().expect("config path utf8"),
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox tpq invocation")
}

fn extract_meta_name(cfg_path: &Path) -> Option<String> {
    let src = std::fs::read_to_string(cfg_path).ok()?;
    let mut in_meta = false;
    for raw_line in src.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_meta = line == "[meta]";
            continue;
        }
        if in_meta {
            if let Some(rest) = line.strip_prefix("name") {
                if let Some(eq) = rest.find('=') {
                    let value = rest[eq + 1..].trim();
                    let unquoted = value.trim_matches(|c| c == '"' || c == '\'');
                    return Some(unquoted.to_string());
                }
            }
        }
    }
    None
}

fn report_under(run_dir: &Path, min_mtime: SystemTime) -> Option<PathBuf> {
    if !run_dir.exists() {
        return None;
    }
    for candidate in &["tpq_sweep_report.json", "tpq_report.json"] {
        let p = run_dir.join(candidate);
        if let Ok(meta) = std::fs::metadata(&p) {
            if let Ok(mtime) = meta.modified() {
                if mtime + Duration::from_secs(1) >= min_mtime {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[test]
fn sweep_harness_runs_all_sweeps_configs_to_completion() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let sweeps_dir = Path::new(manifest).join("configs/sweeps");
    let mut configs: Vec<PathBuf> = std::fs::read_dir(&sweeps_dir)
        .expect("configs/sweeps must exist")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    configs.sort();
    assert!(
        !configs.is_empty(),
        "configs/sweeps/ must contain at least one .toml; the v0.15 zeta workflow has nothing to run otherwise"
    );

    let runs_root = Path::new(manifest).join("runs");
    let mut total_bytes = 0_u64;
    let mut reported_kinds: Vec<String> = Vec::with_capacity(configs.len());

    for cfg in &configs {
        let name = extract_meta_name(cfg)
            .unwrap_or_else(|| panic!("config {} has no [meta].name", cfg.display()));
        let run_dir = runs_root.join(&name);
        let start = SystemTime::now();

        let status = invoke(cfg);
        assert!(
            status.success(),
            "scrapbox tpq {} exit {:?}",
            cfg.display(),
            status.code()
        );

        let report = report_under(&run_dir, start).unwrap_or_else(|| {
            panic!(
                "no fresh tpq_report.json or tpq_sweep_report.json under {}",
                run_dir.display()
            )
        });
        let bytes = std::fs::metadata(&report).expect("report metadata").len();
        assert!(
            bytes > 0,
            "report {} should have non-zero size",
            report.display()
        );
        total_bytes += bytes;
        reported_kinds.push(
            report
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or("?")
                .to_string(),
        );
    }

    // Sanity: at least one report appeared per config, and total bytes
    // are non-trivial (every config produces JSON, not an empty file).
    assert_eq!(reported_kinds.len(), configs.len());
    assert!(
        total_bytes > (configs.len() as u64) * 32,
        "expected > 32 bytes per report on average, got {total_bytes} across {} configs",
        configs.len()
    );
}
