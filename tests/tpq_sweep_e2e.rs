#![allow(clippy::doc_markdown, clippy::suboptimal_flops)]
//! End-to-end tests for `scrapbox tpq` sweep dispatch (v0.13 alpha):
//! `[tpq.sweep] axis = "beta"` produces `tpq_sweep_report.json` with
//! one density row per beta, matches per-beta single-mode TPQ, and
//! reports a non-empty `krylov_stats`.

use std::process::Command;

fn write_config(path: &std::path::Path, tag: &str, sweep_block: Option<&str>) {
    let sweep = sweep_block.unwrap_or("");
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_sweep_e2e_{tag}"
description = "v0.13 alpha tpq sweep dispatch e2e"
created = "2026-05-25"
tags = ["v0.13", "tpq", "sweep"]

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

[tpq]
kind = "density"
source = "matrix_free"
n_samples = 40
seed = 7
krylov_m = 30
{sweep}

[output]
directory = "runs/tpq_sweep_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn invoke(config_path: &std::path::Path) -> std::process::ExitStatus {
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

#[test]
fn tpq_sweep_density_matrix_free_emits_per_beta_rows() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_tpq_sweep_density.toml");
    let sweep = "[tpq.sweep]\naxis = \"beta\"\nvalues = [0.5, 1.0, 2.0, 4.0]\n";
    write_config(&cfg, "density", Some(sweep));
    assert!(invoke(&cfg).success(), "sweep CLI must succeed");
    let path =
        std::path::Path::new(manifest).join("runs/tpq_sweep_e2e_density/tpq_sweep_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["kind"], "density");
    assert_eq!(json["axis"], "beta");
    assert_eq!(json["source"], "matrix_free");
    let rows = json["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 4, "one row per beta");
    let mut prev_beta = -1.0_f64;
    for row in rows {
        let beta = row["beta"].as_f64().expect("row beta");
        assert!(beta > prev_beta, "rows must preserve input beta order");
        prev_beta = beta;
        let density = row["density"].as_array().expect("density array");
        assert_eq!(density.len(), 4, "L = 4 sites per row");
        let total: f64 = density.iter().map(|v| v.as_f64().unwrap()).sum();
        assert!((total - 4.0).abs() < 1e-6, "row total {total} ~ 4");
    }
    let stats = json["krylov_stats"].as_object().expect("krylov_stats");
    let min_m = stats["min_m"].as_u64().unwrap();
    let max_m = stats["max_m"].as_u64().unwrap();
    assert!(min_m >= 1, "min_m >= 1");
    assert!(max_m <= 30, "max_m <= krylov_m cap");
    std::fs::remove_file(&cfg).ok();
}

#[test]
fn tpq_sweep_matches_per_beta_single_mode_at_same_seed() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let betas = [0.5_f64, 2.0_f64];

    let cfg_sweep = std::path::Path::new(manifest).join("configs/_test_tpq_sweep_xref_sweep.toml");
    let sweep = "[tpq.sweep]\naxis = \"beta\"\nvalues = [0.5, 2.0]\n";
    write_config(&cfg_sweep, "xref_sweep", Some(sweep));
    assert!(invoke(&cfg_sweep).success());
    let sweep_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(manifest)
                .join("runs/tpq_sweep_e2e_xref_sweep/tpq_sweep_report.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let sweep_rows = sweep_json["rows"].as_array().unwrap();

    for (i, &beta) in betas.iter().enumerate() {
        let tag = format!("xref_single_{i}");
        let cfg_single =
            std::path::Path::new(manifest).join(format!("configs/_test_tpq_sweep_{tag}.toml"));
        let body = format!(
            r#"
schema_version = "0.2"

[meta]
name = "tpq_sweep_e2e_{tag}"
description = "v0.13 alpha tpq sweep cross-check single mode at fixed beta"
created = "2026-05-25"
tags = ["v0.13", "tpq", "sweep"]

[hamiltonian]
model = "hubbard_1d_inhomogeneous"
num_sites = 4
hopping_j = 1.0
on_site_interaction = 4.0
spinful = true
num_electrons_per_spin = 2
beta = {beta}
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

[tpq]
kind = "density"
source = "matrix_free"
n_samples = 40
seed = 7
krylov_m = 30

[output]
directory = "runs/tpq_sweep_e2e_{tag}"
format = "json"
overwrite = true
"#
        );
        std::fs::write(&cfg_single, body).unwrap();
        assert!(invoke(&cfg_single).success());
        let single_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(manifest)
                    .join(format!("runs/tpq_sweep_e2e_{tag}/tpq_report.json")),
            )
            .unwrap(),
        )
        .unwrap();
        let single_density = single_json["density"].as_array().unwrap();
        let sweep_density = sweep_rows[i]["density"].as_array().unwrap();
        for (a, b) in single_density.iter().zip(sweep_density.iter()) {
            let av = a.as_f64().unwrap();
            let bv = b.as_f64().unwrap();
            assert!(
                (av - bv).abs() < 1e-10,
                "beta = {beta}, sweep density {bv} vs single density {av}, delta = {}",
                (av - bv).abs()
            );
        }
        std::fs::remove_file(&cfg_single).ok();
    }
    std::fs::remove_file(&cfg_sweep).ok();
}

fn write_sweep_config_full(
    path: &std::path::Path,
    tag: &str,
    kind: &str,
    source: &str,
    axis: &str,
    values: &str,
    extra: &str,
) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_sweep_e2e_{tag}"
description = "v0.14 gamma sweep broadening e2e"
created = "2026-05-25"
tags = ["v0.14", "tpq", "sweep"]

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

{extra}

[tpq]
kind = "{kind}"
source = "{source}"
n_samples = 30
seed = 7
krylov_m = 30

[tpq.sweep]
axis = "{axis}"
values = {values}

[output]
directory = "runs/tpq_sweep_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

#[test]
fn tpq_sweep_beta_density_ed_path_emits_per_beta_rows() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_tpq_sweep_ed.toml");
    write_sweep_config_full(
        &cfg,
        "ed_beta",
        "density",
        "ed",
        "beta",
        "[0.5, 1.0, 2.0]",
        "",
    );
    assert!(invoke(&cfg).success(), "ed sweep CLI must succeed");
    let path =
        std::path::Path::new(manifest).join("runs/tpq_sweep_e2e_ed_beta/tpq_sweep_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["kind"], "density");
    assert_eq!(json["source"], "ed");
    assert_eq!(json["axis"], "beta");
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 3);
    for row in rows {
        let d = row["density"].as_array().unwrap();
        assert_eq!(d.len(), 4);
        let total: f64 = d.iter().map(|v| v.as_f64().unwrap()).sum();
        assert!((total - 4.0).abs() < 1e-6, "ed canonical total {total} ~ 4");
    }
    let stats = json["krylov_stats"].as_object().expect("krylov_stats");
    assert_eq!(stats["min_m"].as_u64().unwrap(), 0);
    assert_eq!(stats["max_m"].as_u64().unwrap(), 0);
    std::fs::remove_file(&cfg).ok();
}

#[test]
fn tpq_sweep_beta_work_matrix_free_emits_per_beta_work_rows() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_tpq_sweep_work.toml");
    write_sweep_config_full(
        &cfg,
        "work_beta",
        "work_statistics",
        "matrix_free",
        "beta",
        "[0.5, 2.0]",
        "[quench]
kind = \"sudden\"
final_external_potential.kind = \"explicit\"
final_external_potential.values = [0.3, -0.3, 0.3, -0.3]
",
    );
    assert!(invoke(&cfg).success(), "work_statistics sweep must succeed");
    let path =
        std::path::Path::new(manifest).join("runs/tpq_sweep_e2e_work_beta/tpq_sweep_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["kind"], "work_statistics");
    assert_eq!(json["axis"], "beta");
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert!(row["beta"].as_f64().is_some(), "beta present");
        assert!(row["mean_w"].as_f64().is_some(), "mean_w present");
        assert!(
            row["work_variance"].as_f64().unwrap() >= 0.0,
            "variance >= 0"
        );
        assert!(row["mean_w_stderr"].as_f64().unwrap() >= 0.0, "stderr >= 0");
    }
    std::fs::remove_file(&cfg).ok();
}

#[test]
fn tpq_sweep_krylov_tol_density_matrix_free_per_row_krylov_stats() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_tpq_sweep_kt.toml");
    write_sweep_config_full(
        &cfg,
        "kt",
        "density",
        "matrix_free",
        "krylov_tol",
        "[1e-12, 1e-8, 1e-4]",
        "",
    );
    assert!(invoke(&cfg).success(), "krylov_tol sweep must succeed");
    let path = std::path::Path::new(manifest).join("runs/tpq_sweep_e2e_kt/tpq_sweep_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["kind"], "density_krylov_tol");
    assert_eq!(json["axis"], "krylov_tol");
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 3);
    let m_tight = rows[0]["krylov_stats"]["min_m"].as_u64().unwrap();
    let m_mid = rows[1]["krylov_stats"]["min_m"].as_u64().unwrap();
    let m_loose = rows[2]["krylov_stats"]["min_m"].as_u64().unwrap();
    assert!(
        m_tight >= m_mid && m_mid >= m_loose,
        "tighter tol should use >= m: tight={m_tight}, mid={m_mid}, loose={m_loose}"
    );
    std::fs::remove_file(&cfg).ok();
}

#[test]
fn tpq_sweep_seed_density_matrix_free_emits_ensemble_summary() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_tpq_sweep_seed.toml");
    write_sweep_config_full(
        &cfg,
        "seed_ens",
        "density",
        "matrix_free",
        "seed",
        "[7, 19, 42, 101]",
        "",
    );
    assert!(invoke(&cfg).success(), "seed sweep CLI must succeed");
    let path =
        std::path::Path::new(manifest).join("runs/tpq_sweep_e2e_seed_ens/tpq_sweep_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["kind"], "density_seed_sweep");
    assert_eq!(json["axis"], "seed");
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 4);
    let seeds: Vec<u64> = rows.iter().map(|r| r["seed"].as_u64().unwrap()).collect();
    assert_eq!(seeds, vec![7, 19, 42, 101]);
    let summary = json["ensemble_summary"]
        .as_object()
        .expect("ensemble_summary");
    let mean = summary["mean_density"].as_array().expect("mean_density");
    let stderr = summary["stderr_density"]
        .as_array()
        .expect("stderr_density");
    assert_eq!(mean.len(), 4, "L = 4 sites");
    assert_eq!(stderr.len(), 4);
    let total: f64 = mean.iter().map(|v| v.as_f64().unwrap()).sum();
    assert!((total - 4.0).abs() < 1e-6, "mean density sums to N = 4");
    for v in stderr {
        let s = v.as_f64().unwrap();
        assert!(s > 0.0, "stderr must be > 0 across seeds, got {s}");
    }
    std::fs::remove_file(&cfg).ok();
}
