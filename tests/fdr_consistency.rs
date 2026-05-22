//! Generalized fluctuation-dissipation consistency (Palamara 2024 eq 31)
//! for commuting Hamiltonians.
//!
//! For a uniform-V shift `V → V + δ` the pre- and post-quench
//! Hamiltonians commute (the perturbation is proportional to the
//! conserved `N̂`), so `Θ_2` is identically zero and the classical
//! FDR holds: `<S_irr> = (β²/2) σ_w²`. All three quantities are zero
//! to machine precision.

use std::process::Command;

#[test]
fn uniform_v_quench_fdr_closes_to_machine_precision() {
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
            "configs/dimer_uniform_quench.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("invoke scrapbox run");
    assert!(status.success(), "scrapbox run failed");

    let obs_path = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_uniform_quench")
        .join("observables.json");
    let raw = std::fs::read_to_string(&obs_path).expect("read observables.json");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("parse json");

    let mean_work = json["mean_work"].as_f64().expect("mean_work");
    let s_irr = json["irreversible_entropy"].as_f64().expect("s_irr");
    let sigma_w_sq = json["work_variance"].as_f64().expect("work_variance");
    let theta_2 = json["theta_2"].as_f64().expect("theta_2");
    let fdr_residual = json["fdr_residual"].as_f64().expect("fdr_residual");

    // Uniform shift δ=0.5, N=2 → <W> = δ·N = 1.0.
    assert!((mean_work - 1.0).abs() < 1e-8, "mean_work {mean_work}");
    // Commuting quench: <S_irr> = β(<W> − ΔF) = 0.
    assert!(s_irr.abs() < 1e-8, "s_irr {s_irr}");
    // Variance is at numerical-noise level (χ from finite differences).
    assert!(sigma_w_sq.abs() < 1e-4, "sigma_w_sq {sigma_w_sq}");
    assert!(theta_2.abs() < f64::EPSILON, "theta_2 {theta_2}");
    assert!(
        fdr_residual.abs() < 1e-3,
        "FDR residual {fdr_residual} exceeded tolerance in commuting-quench case"
    );
}
