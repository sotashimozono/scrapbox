//! End-to-end check that `DensityEvaluator::GcePlusProjection` produces
//! a converged SCF result matching Pratt (verified independently by the
//! `dimer_smoke` pipeline) to 1e-10 on the U=4J, β=2 dimer.

use std::process::Command;

#[test]
fn gce_projection_dimer_matches_known_pratt_values() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let status = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "run",
            "configs/dimer_gce_projection.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("invoke scrapbox run");
    assert!(status.success(), "scrapbox run failed");

    let dir = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_gce_projection");
    let obs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("observables.json")).unwrap())
            .unwrap();
    let density: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("density.json")).unwrap()).unwrap();

    let f = obs["free_energy"].as_f64().unwrap();
    assert!((f - (-2.018_149_927_917_81)).abs() < 1e-10, "F = {f}");
    let z = obs["partition_function"].as_f64().unwrap();
    assert!((z - 56.616_465_672_032_99).abs() < 1e-8, "Z = {z}");

    let n0 = density["site_density"][0].as_f64().unwrap();
    let n1 = density["site_density"][1].as_f64().unwrap();
    assert!((n0 - 1.0).abs() < 1e-10);
    assert!((n1 - 1.0).abs() < 1e-10);
}
