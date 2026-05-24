//! Parallel sweep end-to-end: runs `configs/dimer_sweep_parallel.toml`
//! (`parallelism = 4`) and confirms every cell wrote its output.

use std::process::Command;

#[test]
fn parallel_sweep_produces_all_cells() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "sweep",
            "configs/dimer_sweep_parallel.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox sweep");
    assert!(status.success(), "scrapbox sweep failed");

    let base = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_sweep_parallel");
    for label in ["U_0", "U_1", "U_2", "U_3", "U_4", "U_5", "U_6", "U_8"] {
        let obs_path = base.join(label).join("observables.json");
        assert!(obs_path.exists(), "missing observables.json for {label}");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&obs_path).unwrap()).unwrap();
        let f = json["free_energy"].as_f64().expect("free_energy");
        assert!(f.is_finite(), "non-finite F for {label}: {f}");
    }
}
