#![allow(clippy::doc_markdown, clippy::suboptimal_flops)]
//! End-to-end test: BALDA dimer non-commuting quench with
//! `theta_2.method = "lda"` dispatches successfully and produces a
//! Theta_2 value that improves the FDR closure relative to the
//! `method = "zero"` baseline.

use std::process::Command;

// beta = 2.0 from configs/dimer_balda_quench.toml; beta^2/2 = 2.0.
const BETA: f64 = 2.0;

#[test]
fn balda_dimer_lda_theta_improves_fdr_closure() {
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
            "configs/dimer_balda_quench.toml",
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox run");
    assert!(status.success(), "scrapbox run failed");

    let report_path = std::path::Path::new(manifest)
        .join("runs")
        .join("dimer_balda_quench")
        .join("observables.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();

    let theta_2 = json["theta_2"].as_f64().expect("theta_2");
    let sigma_w_sq = json["work_variance"].as_f64().expect("work_variance");
    let s_irr = json["irreversible_entropy"].as_f64().expect("s_irr");
    let fdr_residual = json["fdr_residual"].as_f64().expect("fdr_residual");

    assert!(theta_2 > 0.0, "Theta_2 = {theta_2}, expected > 0");

    let beta_squared_over_two = 0.5 * BETA * BETA;
    let expected_residual = s_irr - beta_squared_over_two * (sigma_w_sq - theta_2);
    assert!(
        (fdr_residual - expected_residual).abs() < 1e-10,
        "reported residual {fdr_residual} vs reconstructed {expected_residual}"
    );

    let baseline_residual = s_irr - beta_squared_over_two * sigma_w_sq;
    assert!(
        fdr_residual.abs() < baseline_residual.abs(),
        "LDA Theta_2 did not close the FDR: |residual| = {} vs baseline |residual| = {}",
        fdr_residual.abs(),
        baseline_residual.abs()
    );
}
