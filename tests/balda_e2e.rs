//! BALDA SCF end-to-end: runs `configs/dimer_balda.toml` and confirms
//! convergence with the Mott-gap smoothing active.

use std::process::Command;

#[test]
fn dimer_balda_converges_with_smoothing() {
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
            "configs/dimer_balda.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox run");
    assert!(status.success(), "scrapbox run failed");

    let dir = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_balda");
    let density: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("density.json")).unwrap()).unwrap();

    let n0 = density["site_density"][0].as_f64().unwrap();
    let n1 = density["site_density"][1].as_f64().unwrap();
    assert!((n0 + n1 - 2.0).abs() < 1e-8, "n_0 + n_1 = {}", n0 + n1);
    assert!((n0 - 1.0).abs() < 0.2, "n_0 = {n0} too far from 1");
    assert!(
        n0 < n1,
        "comb breaks symmetry: site 0 (high V) should be depleted"
    );
}
