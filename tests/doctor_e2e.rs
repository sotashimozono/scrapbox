//! `scrapbox doctor` end-to-end: parses a known-good config without
//! running SCF and confirms the report is well-formed.

use std::process::Command;

#[test]
fn doctor_on_dimer_smoke_emits_report() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "doctor",
            "configs/dimer_smoke.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox doctor");
    assert!(status.success(), "scrapbox doctor failed");

    let report_path = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_smoke")
        .join("doctor_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(json["status"].as_str().unwrap(), "ok");
    let lines = json["lines"].as_array().unwrap();
    assert!(lines.len() >= 8, "doctor report should have >= 8 lines");
    let joined: String = lines
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("schema_version: 0.2"));
    assert!(joined.contains("xc: hubbard_lda"));
    assert!(joined.contains("spectrum: dense_diag"));
    assert!(joined.contains("density: pratt_recursion"));
}
