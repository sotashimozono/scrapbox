#![allow(clippy::doc_markdown, clippy::suboptimal_flops)]
//! End-to-end tests for `scrapbox ed`: dense vs matrix-free Lanczos
//! produce matching low spectrum on the same L=4 Hubbard config.

use std::process::Command;

fn write_config(path: &std::path::Path, tag: &str, solver: &str, num_eigenvalues: Option<usize>) {
    let num_line = num_eigenvalues
        .map(|k| format!("num_eigenvalues = {k}\n"))
        .unwrap_or_default();
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "ed_cli_e2e_{tag}"
description = "v0.7 alpha CLI ed dispatcher cross-check"
created = "2026-05-24"
tags = ["v0.7", "ed", "cli"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 4
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 2
beta = 2.0
external_potential.kind = "explicit"
external_potential.values = [0.1, -0.2, 0.3, -0.1]

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

[ed]
solver = "{solver}"
{num_line}
[output]
directory = "runs/ed_cli_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn invoke_ed(config_path: &std::path::Path) -> std::process::ExitStatus {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            "ed",
            config_path.to_str().unwrap(),
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox ed")
}

fn read_eigenvalues(out_dir: &str) -> Vec<f64> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest)
        .join(out_dir)
        .join("ed_spectrum.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json["eigenvalues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[test]
fn ed_cli_dense_emits_full_spectrum_at_l4() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_ed_dense_full.toml");
    write_config(&cfg, "dense_full", "dense", None);
    let status = invoke_ed(&cfg);
    assert!(status.success());
    let eigs = read_eigenvalues("runs/ed_cli_e2e_dense_full");
    // L=4, n_up=n_dn=2 -> C(4,2)^2 = 36 states.
    assert_eq!(eigs.len(), 36);
    for w in eigs.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "spectrum not ascending: {} > {}",
            w[0],
            w[1]
        );
    }
    std::fs::remove_file(&cfg).ok();
}

#[test]
fn ed_cli_matrix_free_low_spectrum_matches_dense_at_l4() {
    let manifest = env!("CARGO_MANIFEST_DIR");

    let cfg_dense = std::path::Path::new(manifest).join("configs/_test_ed_dense_cross.toml");
    write_config(&cfg_dense, "dense_cross", "dense", Some(4));
    assert!(invoke_ed(&cfg_dense).success());
    let dense = read_eigenvalues("runs/ed_cli_e2e_dense_cross");

    let cfg_mf = std::path::Path::new(manifest).join("configs/_test_ed_mf_cross.toml");
    write_config(&cfg_mf, "mf_cross", "matrix_free_lanczos", Some(4));
    assert!(invoke_ed(&cfg_mf).success());
    let mf = read_eigenvalues("runs/ed_cli_e2e_mf_cross");

    assert_eq!(dense.len(), 4);
    assert_eq!(mf.len(), 4);
    for k in 0..4 {
        assert!(
            (dense[k] - mf[k]).abs() < 1e-6,
            "level {k}: dense = {}, matrix-free = {}",
            dense[k],
            mf[k]
        );
    }
    std::fs::remove_file(&cfg_dense).ok();
    std::fs::remove_file(&cfg_mf).ok();
}

#[test]
fn ed_cli_rejects_sparse_lanczos_until_beta() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_ed_sparse_unimpl.toml");
    write_config(&cfg, "sparse_unimpl", "sparse_lanczos", Some(4));
    let status = invoke_ed(&cfg);
    assert!(
        !status.success(),
        "sparse_lanczos must error until v0.7 beta lands"
    );
    std::fs::remove_file(&cfg).ok();
}
