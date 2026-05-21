//! Integration smoke test — parse every sample config under `configs/` and
//! assert each one is accepted by the schema. This is the leanest
//! `kind = "..."` variant-coverage check (per
//! `notes/discipline/ACCEPTANCE.md` §1.1).

use scrapbox::config::Config;
use std::path::PathBuf;

fn configs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("configs")
}

#[test]
fn every_sample_config_parses() {
    let dir = configs_dir();
    if !dir.is_dir() {
        return;
    }
    let entries = std::fs::read_dir(&dir).expect("read configs/");
    let mut count = 0_usize;
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        let cfg = Config::from_file(&path)
            .unwrap_or_else(|e| panic!("config {} failed: {e}", path.display()));
        assert_eq!(
            cfg.schema_version,
            "0.1",
            "{} schema_version drift",
            path.display()
        );
        count += 1;
    }
    assert!(
        count > 0,
        "expected at least one config/*.toml under {}",
        dir.display()
    );
}
