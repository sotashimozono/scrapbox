#![allow(clippy::doc_markdown, clippy::suboptimal_flops)]
//! End-to-end tests for `scrapbox tpq`: density + work_statistics x ed + matrix_free.

use std::process::Command;

fn write_density_config(path: &std::path::Path, tag: &str, source: &str) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_cli_e2e_{tag}"
description = "v0.8 beta CLI tpq dispatcher cross-check"
created = "2026-05-24"
tags = ["v0.8", "tpq", "cli"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 4
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 2
beta = 2.0
external_potential.kind = "explicit"
external_potential.values = [0.1, -0.1, 0.1, -0.1]

[xc_functional]
kind = "non_interacting"

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 1
tolerance = 1.0
mixing.kind = "linear"
mixing.alpha = 1.0
initial_density.kind = "uniform"

[observables]

[tpq]
kind = "density"
source = "{source}"
n_samples = 500
seed = 7
krylov_m = 30

[output]
directory = "runs/tpq_cli_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn write_work_config(path: &std::path::Path, tag: &str, source: &str) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_cli_e2e_{tag}"
description = "v0.8 beta CLI tpq work cross-check"
created = "2026-05-24"
tags = ["v0.8", "tpq", "cli", "work"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 4
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 2
beta = 2.0
external_potential.kind = "explicit"
external_potential.values = [0.0, 0.0, 0.0, 0.0]

[xc_functional]
kind = "non_interacting"

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 1
tolerance = 1.0
mixing.kind = "linear"
mixing.alpha = 1.0
initial_density.kind = "uniform"

[observables]

[quench]
kind = "sudden"
final_external_potential.kind = "explicit"
final_external_potential.values = [0.3, -0.3, 0.3, -0.3]

[tpq]
kind = "work_statistics"
source = "{source}"
n_samples = 500
seed = 7
krylov_m = 30

[output]
directory = "runs/tpq_cli_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn invoke_tpq(config_path: &std::path::Path) -> std::process::ExitStatus {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "tpq",
            config_path.to_str().unwrap(),
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox tpq")
}

fn read_density(tag: &str) -> Vec<f64> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .join("runs")
        .join(format!("tpq_cli_e2e_{tag}"))
        .join("tpq_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json["density"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn read_work(tag: &str) -> (f64, f64) {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .join("runs")
        .join(format!("tpq_cli_e2e_{tag}"))
        .join("tpq_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    (
        json["mean_w"].as_f64().unwrap(),
        json["work_variance"].as_f64().unwrap(),
    )
}

#[test]
fn tpq_cli_density_matrix_free_matches_ed_at_l4() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg_ed = std::path::Path::new(manifest).join("configs/_test_tpq_dens_ed.toml");
    write_density_config(&cfg_ed, "dens_ed", "ed");
    assert!(invoke_tpq(&cfg_ed).success());
    let d_ed = read_density("dens_ed");

    let cfg_mf = std::path::Path::new(manifest).join("configs/_test_tpq_dens_mf.toml");
    write_density_config(&cfg_mf, "dens_mf", "matrix_free");
    assert!(invoke_tpq(&cfg_mf).success());
    let d_mf = read_density("dens_mf");

    assert_eq!(d_ed.len(), 4);
    assert_eq!(d_mf.len(), 4);
    for i in 0..4 {
        // Same RNG seed, different psi_beta projection (eigenbasis vs Krylov):
        // statistical agreement only, not exact match.
        assert!(
            (d_ed[i] - d_mf[i]).abs() < 0.1,
            "site {i}: ed = {}, matrix_free = {}",
            d_ed[i],
            d_mf[i]
        );
    }
    std::fs::remove_file(&cfg_ed).ok();
    std::fs::remove_file(&cfg_mf).ok();
}

#[test]
fn tpq_cli_work_matrix_free_matches_ed_at_l4() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg_ed = std::path::Path::new(manifest).join("configs/_test_tpq_work_ed.toml");
    write_work_config(&cfg_ed, "work_ed", "ed");
    assert!(invoke_tpq(&cfg_ed).success());
    let (mw_ed, sw_ed) = read_work("work_ed");

    let cfg_mf = std::path::Path::new(manifest).join("configs/_test_tpq_work_mf.toml");
    write_work_config(&cfg_mf, "work_mf", "matrix_free");
    assert!(invoke_tpq(&cfg_mf).success());
    let (mw_mf, sw_mf) = read_work("work_mf");

    assert!(
        (mw_ed - mw_mf).abs() < 0.05,
        "<W>: ed = {mw_ed}, matrix_free = {mw_mf}"
    );
    assert!(
        (sw_ed - sw_mf).abs() < 0.15,
        "sigma_W^2: ed = {sw_ed}, matrix_free = {sw_mf}"
    );
    std::fs::remove_file(&cfg_ed).ok();
    std::fs::remove_file(&cfg_mf).ok();
}
