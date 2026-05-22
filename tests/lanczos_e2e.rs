//! End-to-end check that `SpectrumSource::Lanczos` produces a converged
//! SCF result indistinguishable (to 1e-10) from `dense_diag` on a dimer.

use std::process::Command;

#[test]
fn lanczos_dimer_matches_dense_diag_results() {
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
            "configs/dimer_lanczos.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("invoke scrapbox run");
    assert!(status.success(), "scrapbox run failed");

    let dir = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_lanczos");
    let observables: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("observables.json")).unwrap())
            .unwrap();
    let density: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("density.json")).unwrap()).unwrap();

    // Free energy of the symmetric U=4J, β=2 dimer (verified independently
    // by the dense_diag pipeline on dimer_smoke).
    let f = observables["free_energy"].as_f64().unwrap();
    assert!((f - (-2.018_149_927_917_81)).abs() < 1e-10, "F = {f}");

    // Half-filling: n_0 = n_1 = 1.0 by particle-hole symmetry.
    let n0 = density["site_density"][0].as_f64().unwrap();
    let n1 = density["site_density"][1].as_f64().unwrap();
    assert!((n0 - 1.0).abs() < 1e-10);
    assert!((n1 - 1.0).abs() < 1e-10);
}
