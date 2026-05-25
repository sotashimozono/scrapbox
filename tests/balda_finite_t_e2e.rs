//! BALDA finite-T dispatch end-to-end (v0.13 beta).
//!
//! v0.13 beta scope: the `balda_finite_t` route is a dispatch shim
//! whose evaluator delegates to T=0 BALDA. These tests confirm
//! that the route wires through SCF, converges, and produces the
//! same density as the plain `balda` route at the same parameters.
//! When the thermal evaluator lands in a later sprint, these
//! cross-checks should be replaced with temperature-resolved
//! comparisons (e.g. limit checks against ed-path thermal density).

use std::process::Command;

fn write_config(path: &std::path::Path, tag: &str, kind: &str, beta: f64) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "balda_finite_t_e2e_{tag}"
description = "v0.13 beta BALDA finite-T dispatch e2e"
created = "2026-05-25"
tags = ["v0.13", "balda", "finite_t"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 2
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 1
beta = {beta}
external_potential.kind = "comb"
external_potential.amplitude = 0.3

[xc_functional]
kind = "{kind}"

[xc_functional.params]
mott_gap_smoothing_width = 0.15

[spectrum_source]
kind = "dense_diag"

[density_evaluator]
kind = "pratt_recursion"

[scf]
max_iterations = 400
tolerance = 1e-8
mixing.kind = "pulay"
mixing.alpha = 0.1
mixing.history_depth = 8
initial_density.kind = "uniform"

[observables]

[output]
directory = "runs/balda_finite_t_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn invoke_run(path: &std::path::Path) -> std::process::ExitStatus {
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
            path.to_str().unwrap(),
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox run")
}

fn read_density(run_name: &str) -> Vec<f64> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(manifest).join("runs").join(run_name);
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("density.json")).unwrap()).unwrap();
    v["site_density"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn balda_finite_t_route_runs_and_converges() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_balda_ft_runs.toml");
    write_config(&cfg, "runs", "balda_finite_t", 2.0);
    assert!(
        invoke_run(&cfg).success(),
        "balda_finite_t SCF must converge"
    );
    let density = read_density("balda_finite_t_e2e_runs");
    assert_eq!(density.len(), 2, "L = 2 sites");
    let total: f64 = density.iter().sum();
    assert!(
        (total - 2.0).abs() < 1e-6,
        "total electron count {total} ~ 2"
    );
    std::fs::remove_file(&cfg).ok();
}

#[test]
fn balda_finite_t_differs_from_t0_balda_at_finite_beta_in_v014() {
    // v0.14 alpha: BaldaFiniteT now applies a leading-T^2 additive
    // correction. At beta = 2.0, U = 4.0 the correction is ~5e-3
    // per site at half-filling, so the SCF density should differ
    // from plain BALDA by something larger than 1e-5 but still
    // small (< 1e-2) to keep the route variationally close.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let beta = 2.0_f64;

    let cfg_ft = std::path::Path::new(manifest).join("configs/_test_balda_ft_xref_ft.toml");
    write_config(&cfg_ft, "xref_ft", "balda_finite_t", beta);
    assert!(invoke_run(&cfg_ft).success());
    let d_ft = read_density("balda_finite_t_e2e_xref_ft");

    let cfg_t0 = std::path::Path::new(manifest).join("configs/_test_balda_ft_xref_t0.toml");
    write_config(&cfg_t0, "xref_t0", "balda", beta);
    assert!(invoke_run(&cfg_t0).success());
    let d_t0 = read_density("balda_finite_t_e2e_xref_t0");

    let max_delta = d_ft
        .iter()
        .zip(d_t0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > 1e-5,
        "v0.14 alpha: balda_finite_t SCF density must show T-correction vs plain balda, max delta {max_delta}"
    );
    assert!(
        max_delta < 1e-2,
        "v0.14 alpha: T-correction must remain small for SCF variational stability, max delta {max_delta}"
    );
    std::fs::remove_file(&cfg_ft).ok();
    std::fs::remove_file(&cfg_t0).ok();
}

#[test]
fn balda_finite_t_large_beta_recovers_t0_balda_within_scf_tolerance() {
    // v0.14 alpha: at beta = 1e3 the T^2 correction is ~5e-9 per
    // site, well below the 1e-8 SCF residual tolerance. The two
    // SCFs should converge to densities matching at < 1e-6.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let beta = 1.0e3_f64;

    let cfg_ft = std::path::Path::new(manifest).join("configs/_test_balda_ft_cold_ft.toml");
    write_config(&cfg_ft, "cold_ft", "balda_finite_t", beta);
    assert!(invoke_run(&cfg_ft).success());
    let d_ft = read_density("balda_finite_t_e2e_cold_ft");

    let cfg_t0 = std::path::Path::new(manifest).join("configs/_test_balda_ft_cold_t0.toml");
    write_config(&cfg_t0, "cold_t0", "balda", beta);
    assert!(invoke_run(&cfg_t0).success());
    let d_t0 = read_density("balda_finite_t_e2e_cold_t0");

    for (a, b) in d_ft.iter().zip(d_t0.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "v0.14 alpha low-T limit: balda_finite_t vs balda density {a} vs {b}, delta = {}",
            (a - b).abs()
        );
    }
    std::fs::remove_file(&cfg_ft).ok();
    std::fs::remove_file(&cfg_t0).ok();
}
