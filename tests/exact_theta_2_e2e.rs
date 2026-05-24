#![allow(clippy::doc_markdown, clippy::suboptimal_flops)]
//! End-to-end test for `theta_2.method = "exact"`: the exact Palamara
//! III.3 path should produce a Theta_2 that tightens the FDR closure
//! versus both the "zero" baseline and the v0.5 alpha LDA placeholder.

use std::process::Command;

fn write_config(path: &std::path::Path, tag: &str, method: &str) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "exact_theta_e2e_{tag}"
description = "v0.9 beta CLI exact Theta_2 dispatch"
created = "2026-05-24"
tags = ["v0.9", "theta_2", "exact"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 2
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 1
beta = 2.0
external_potential.kind = "uniform"
external_potential.amplitude = 0.0

[xc_functional]
kind = "balda"

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 400
tolerance = 1e-10
mixing.kind = "pulay"
mixing.alpha = 0.1
mixing.history_depth = 8
initial_density.kind = "uniform"

[quench]
kind = "sudden"
final_external_potential.kind = "explicit"
final_external_potential.values = [0.3, -0.3]

[observables]
mean_work = true
irreversible_entropy = true
work_variance = true
theta_2.method = "{method}"
free_energy = true
partition_function = true

[output]
directory = "runs/exact_theta_e2e_{tag}"
format = "json"
overwrite = true

[xc_functional.params]
mott_gap_smoothing_width = 0.15
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn invoke_run(config_path: &std::path::Path) -> std::process::ExitStatus {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "run",
            config_path.to_str().unwrap(),
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox run")
}

fn read_obs(tag: &str) -> (f64, f64, f64, f64) {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .join("runs")
        .join(format!("exact_theta_e2e_{tag}"))
        .join("observables.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    (
        json["theta_2"].as_f64().expect("theta_2"),
        json["work_variance"].as_f64().expect("work_variance"),
        json["irreversible_entropy"].as_f64().expect("s_irr"),
        json["fdr_residual"].as_f64().expect("fdr_residual"),
    )
}

#[test]
fn exact_theta_2_tightens_fdr_versus_lda_at_balda_dimer_quench() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg_lda = std::path::Path::new(manifest).join("configs/_test_exact_lda.toml");
    write_config(&cfg_lda, "lda", "lda");
    assert!(invoke_run(&cfg_lda).success());
    let (theta_lda, _, _, residual_lda) = read_obs("lda");

    let cfg_exact = std::path::Path::new(manifest).join("configs/_test_exact_exact.toml");
    write_config(&cfg_exact, "exact", "exact");
    assert!(invoke_run(&cfg_exact).success());
    let (theta_exact, _, _, residual_exact) = read_obs("exact");

    assert!(
        theta_lda > 0.0,
        "LDA placeholder theta_2 should be > 0, got {theta_lda}"
    );
    assert!(
        theta_exact > 0.0,
        "exact theta_2 should be > 0, got {theta_exact}"
    );
    // Exact Palamara III.3 path: |residual| should be smaller than LDA's
    // for the dimer non-commuting quench; tight inequality, not just sign.
    assert!(
        residual_exact.abs() <= residual_lda.abs() + 1e-12,
        "exact |residual| = {} should be <= LDA |residual| = {}",
        residual_exact.abs(),
        residual_lda.abs()
    );
    std::fs::remove_file(&cfg_lda).ok();
    std::fs::remove_file(&cfg_exact).ok();
}

#[test]
fn exact_theta_2_unknown_method_errors() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_exact_unknown.toml");
    write_config(&cfg, "unknown", "unknown_method");
    let status = invoke_run(&cfg);
    assert!(!status.success(), "unknown method should error");
    std::fs::remove_file(&cfg).ok();
}
