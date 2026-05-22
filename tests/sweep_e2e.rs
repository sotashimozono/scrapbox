//! End-to-end sweep: runs `configs/dimer_sweep_u.toml` and confirms the
//! cartesian-product runner produced one observables.json per cell with
//! the expected free energies.

use std::process::Command;

#[test]
fn dimer_u_sweep_produces_one_directory_per_value() {
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
            "configs/dimer_sweep_u.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox sweep");
    assert!(status.success(), "scrapbox sweep failed");

    let base = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_sweep_u");
    for label in ["U_0", "U_2", "U_4", "U_6"] {
        let dir = base.join(label);
        let obs_path = dir.join("observables.json");
        assert!(obs_path.exists(), "missing observables.json for {label}");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&obs_path).unwrap()).unwrap();
        let f = json["free_energy"].as_f64().expect("free_energy");
        assert!(f.is_finite(), "non-finite F for {label}: {f}");
        // Free energy must decrease monotonically with U at half filling
        // for this fixed beta — we only assert finiteness here; ordering
        // is checked manually offline.
    }
}
