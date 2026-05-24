//! `scrapbox bench` end-to-end: runs `configs/dimer_bench.toml`
//! (warmup = 1, measured = 3) and validates the timing report shape.

use std::process::Command;

#[test]
fn dimer_bench_emits_well_formed_report() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "bench",
            "configs/dimer_bench.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox bench");
    assert!(status.success(), "scrapbox bench failed");

    let report_path = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_bench")
        .join("bench_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(json["warmup"].as_u64().unwrap(), 1);
    assert_eq!(json["measured"].as_u64().unwrap(), 3);
    assert_eq!(json["samples_ms"].as_array().unwrap().len(), 3);
    for stat in ["min_ms", "median_ms", "p95_ms", "mean_ms"] {
        let v = json[stat].as_f64().expect(stat);
        assert!(v.is_finite() && v > 0.0, "{stat} = {v}");
    }
    let min = json["min_ms"].as_f64().unwrap();
    let median = json["median_ms"].as_f64().unwrap();
    let p95 = json["p95_ms"].as_f64().unwrap();
    assert!(min <= median, "min {min} > median {median}");
    assert!(median <= p95, "median {median} > p95 {p95}");
}
