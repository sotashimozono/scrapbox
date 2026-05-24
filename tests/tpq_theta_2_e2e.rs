#![allow(
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::too_many_arguments
)]
//! E2E: `scrapbox tpq` with `kind = "theta_2"` produces the same
//! Palamara III.3 value as `scrapbox run` with `theta_2.method = "exact"`.

use std::process::Command;

fn write_tpq_theta_2_config(path: &std::path::Path, tag: &str) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_theta_2_e2e_{tag}"
description = "v0.10 alpha tpq theta_2 mode e2e"
created = "2026-05-24"
tags = ["v0.10", "tpq", "theta_2"]

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
max_iterations = 400
tolerance = 1e-10
mixing.kind = "linear"
mixing.alpha = 1.0
initial_density.kind = "uniform"

[quench]
kind = "sudden"
final_external_potential.kind = "explicit"
final_external_potential.values = [0.3, -0.3, 0.3, -0.3]

[observables]

[tpq]
kind = "theta_2"
source = "ed"
n_samples = 1
seed = 0

[output]
directory = "runs/tpq_theta_2_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn write_run_exact_config(path: &std::path::Path, tag: &str) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_theta_2_e2e_run_{tag}"
description = "v0.10 alpha cross-check vs scrapbox run exact"
created = "2026-05-24"
tags = ["v0.10", "theta_2", "exact", "run"]

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
max_iterations = 400
tolerance = 1e-10
mixing.kind = "linear"
mixing.alpha = 1.0
initial_density.kind = "uniform"

[quench]
kind = "sudden"
final_external_potential.kind = "explicit"
final_external_potential.values = [0.3, -0.3, 0.3, -0.3]

[observables]
mean_work = true
irreversible_entropy = true
work_variance = true
theta_2.method = "exact"
free_energy = true
partition_function = true

[output]
directory = "runs/tpq_theta_2_e2e_run_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

fn invoke(subcommand: &str, config_path: &std::path::Path) -> std::process::ExitStatus {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--bin",
            "scrapbox",
            "--",
            subcommand,
            config_path.to_str().unwrap(),
        ])
        .current_dir(manifest)
        .status()
        .expect("scrapbox")
}

#[test]
fn tpq_theta_2_ed_matches_run_exact_at_l4() {
    let manifest = env!("CARGO_MANIFEST_DIR");

    let cfg_tpq = std::path::Path::new(manifest).join("configs/_test_tpq_theta_2.toml");
    write_tpq_theta_2_config(&cfg_tpq, "ed");
    assert!(invoke("tpq", &cfg_tpq).success());
    let theta_tpq = {
        let path = std::path::Path::new(manifest).join("runs/tpq_theta_2_e2e_ed/tpq_report.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["theta_2"].as_f64().expect("theta_2")
    };

    let cfg_run = std::path::Path::new(manifest).join("configs/_test_tpq_theta_2_run.toml");
    write_run_exact_config(&cfg_run, "ed");
    assert!(invoke("run", &cfg_run).success());
    let theta_run = {
        let path =
            std::path::Path::new(manifest).join("runs/tpq_theta_2_e2e_run_ed/observables.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["theta_2"].as_f64().expect("theta_2")
    };

    assert!(
        (theta_tpq - theta_run).abs() < 1e-12,
        "tpq kind=theta_2 ({theta_tpq}) should match run theta_2.method=exact ({theta_run})"
    );
    std::fs::remove_file(&cfg_tpq).ok();
    std::fs::remove_file(&cfg_run).ok();
}

fn write_tpq_theta_2_mf_config(path: &std::path::Path, tag: &str, k_states: usize) {
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_theta_2_mf_e2e_{tag}"
description = "v0.10 beta tpq theta_2 matrix-free e2e"
created = "2026-05-24"
tags = ["v0.10", "tpq", "theta_2", "matrix_free"]

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
max_iterations = 400
tolerance = 1e-10
mixing.kind = "linear"
mixing.alpha = 1.0
initial_density.kind = "uniform"

[quench]
kind = "sudden"
final_external_potential.kind = "explicit"
final_external_potential.values = [0.3, -0.3, 0.3, -0.3]

[observables]

[tpq]
kind = "theta_2"
source = "matrix_free"
n_samples = 1
seed = 0
theta_2_k_states = {k_states}

[output]
directory = "runs/tpq_theta_2_mf_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

#[test]
fn tpq_theta_2_matrix_free_k_dim_matches_ed_at_l4() {
    // v0.10 beta: scrapbox tpq with kind=theta_2 source=matrix_free,
    // K = full dim (36 at L=4) must match the ed-path Theta_2 to
    // machine precision (Lanczos converges full Krylov subspace).
    let manifest = env!("CARGO_MANIFEST_DIR");

    let cfg_mf = std::path::Path::new(manifest).join("configs/_test_tpq_theta_2_mf_kdim.toml");
    write_tpq_theta_2_mf_config(&cfg_mf, "kdim", 36);
    assert!(invoke("tpq", &cfg_mf).success());
    let theta_mf = {
        let path =
            std::path::Path::new(manifest).join("runs/tpq_theta_2_mf_e2e_kdim/tpq_report.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["theta_2"].as_f64().expect("theta_2")
    };

    let cfg_ed = std::path::Path::new(manifest).join("configs/_test_tpq_theta_2_ed_xref.toml");
    write_tpq_theta_2_config(&cfg_ed, "ed_xref");
    assert!(invoke("tpq", &cfg_ed).success());
    let theta_ed = {
        let path =
            std::path::Path::new(manifest).join("runs/tpq_theta_2_e2e_ed_xref/tpq_report.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["theta_2"].as_f64().expect("theta_2")
    };

    assert!(
        (theta_mf - theta_ed).abs() < 1e-7,
        "K=dim matrix-free Theta_2 ({theta_mf}) should match ed-path ({theta_ed})"
    );
    std::fs::remove_file(&cfg_mf).ok();
    std::fs::remove_file(&cfg_ed).ok();
}

fn write_tpq_density_mf_config(
    path: &std::path::Path,
    tag: &str,
    krylov_m: Option<usize>,
    krylov_tol: Option<f64>,
) {
    let krylov_m_line = krylov_m
        .map(|m| {
            format!(
                "krylov_m = {m}
"
            )
        })
        .unwrap_or_default();
    let krylov_tol_line = krylov_tol
        .map(|t| {
            format!(
                "krylov_tol = {t}
"
            )
        })
        .unwrap_or_default();
    let body = format!(
        r#"
schema_version = "0.2"

[meta]
name = "tpq_krylov_tol_e2e_{tag}"
description = "v0.10 gamma adaptive krylov dispatch e2e"
created = "2026-05-24"
tags = ["v0.10", "tpq", "krylov_tol"]

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
source = "matrix_free"
n_samples = 200
seed = 42
{krylov_m_line}{krylov_tol_line}
[output]
directory = "runs/tpq_krylov_tol_e2e_{tag}"
format = "json"
overwrite = true
"#
    );
    std::fs::write(path, body).expect("write config");
}

#[test]
fn tpq_adaptive_krylov_density_matches_fixed_at_l4() {
    let manifest = env!("CARGO_MANIFEST_DIR");

    let cfg_fixed = std::path::Path::new(manifest).join("configs/_test_tpq_krylov_fixed.toml");
    write_tpq_density_mf_config(&cfg_fixed, "fixed", Some(30), None);
    assert!(invoke("tpq", &cfg_fixed).success());
    let d_fixed = {
        let path =
            std::path::Path::new(manifest).join("runs/tpq_krylov_tol_e2e_fixed/tpq_report.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["density"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<f64>>()
    };

    let cfg_adaptive =
        std::path::Path::new(manifest).join("configs/_test_tpq_krylov_adaptive.toml");
    write_tpq_density_mf_config(&cfg_adaptive, "adaptive", Some(60), Some(1e-10));
    assert!(invoke("tpq", &cfg_adaptive).success());
    let d_adaptive = {
        let path =
            std::path::Path::new(manifest).join("runs/tpq_krylov_tol_e2e_adaptive/tpq_report.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        json["density"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<f64>>()
    };

    assert_eq!(d_fixed.len(), 4);
    assert_eq!(d_adaptive.len(), 4);
    // Same seed, same n_samples; adaptive at tol=1e-10 should hit the
    // same Krylov accuracy as fixed m=30. Allow loose 1e-6 since the
    // Krylov stopping condition reorders floating-point ops slightly.
    for i in 0..4 {
        assert!(
            (d_fixed[i] - d_adaptive[i]).abs() < 1e-6,
            "site {i}: fixed = {}, adaptive = {}",
            d_fixed[i],
            d_adaptive[i]
        );
    }
    std::fs::remove_file(&cfg_fixed).ok();
    std::fs::remove_file(&cfg_adaptive).ok();
}

#[test]
fn tpq_adaptive_krylov_emits_krylov_stats_in_json() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    // adaptive run: krylov_tol set
    let cfg_adaptive =
        std::path::Path::new(manifest).join("configs/_test_tpq_kstats_adaptive.toml");
    write_tpq_density_mf_config(&cfg_adaptive, "kstats_adaptive", Some(60), Some(1e-10));
    assert!(invoke("tpq", &cfg_adaptive).success());
    let json_adaptive: serde_json::Value = {
        let path = std::path::Path::new(manifest)
            .join("runs/tpq_krylov_tol_e2e_kstats_adaptive/tpq_report.json");
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    };
    let stats = json_adaptive
        .get("krylov_stats")
        .expect("adaptive run must emit krylov_stats");
    let min_m = stats["min_m"].as_u64().expect("min_m");
    let max_m = stats["max_m"].as_u64().expect("max_m");
    let mean_m = stats["mean_m"].as_f64().expect("mean_m");
    assert!(min_m >= 1, "min_m should be >= 1, got {min_m}");
    assert!(
        max_m <= 60,
        "max_m should be <= krylov_m bound 60, got {max_m}"
    );
    assert!(
        mean_m >= min_m as f64 && mean_m <= max_m as f64,
        "mean_m {mean_m} not in [{min_m}, {max_m}]"
    );

    // fixed run: krylov_tol absent -> krylov_stats also present (since matrix_free always emits)
    // but min == max == m for fixed
    let cfg_fixed = std::path::Path::new(manifest).join("configs/_test_tpq_kstats_fixed.toml");
    write_tpq_density_mf_config(&cfg_fixed, "kstats_fixed", Some(30), None);
    assert!(invoke("tpq", &cfg_fixed).success());
    let json_fixed: serde_json::Value = {
        let path = std::path::Path::new(manifest)
            .join("runs/tpq_krylov_tol_e2e_kstats_fixed/tpq_report.json");
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    };
    let stats_f = json_fixed
        .get("krylov_stats")
        .expect("fixed matrix_free run must emit krylov_stats too");
    assert_eq!(stats_f["min_m"].as_u64().unwrap(), 30);
    assert_eq!(stats_f["max_m"].as_u64().unwrap(), 30);

    std::fs::remove_file(&cfg_adaptive).ok();
    std::fs::remove_file(&cfg_fixed).ok();
}

#[test]
fn tpq_theta_2_matrix_free_emits_krylov_stats_in_json() {
    // v0.12 alpha: matrix-free theta_2 must report the actual Lanczos
    // subspace dim it used (effective_m) under `krylov_stats`. The
    // theta_2 path runs a single Lanczos diagonalization, so
    // min_m == max_m == mean_m == effective_m.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cfg = std::path::Path::new(manifest).join("configs/_test_tpq_theta_2_mf_stats.toml");
    write_tpq_theta_2_mf_config(&cfg, "stats", 8);
    assert!(invoke("tpq", &cfg).success());
    let path = std::path::Path::new(manifest).join("runs/tpq_theta_2_mf_e2e_stats/tpq_report.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let stats = json
        .get("krylov_stats")
        .expect("matrix_free theta_2 must emit krylov_stats");
    let min_m = stats["min_m"].as_u64().expect("min_m");
    let max_m = stats["max_m"].as_u64().expect("max_m");
    let mean_m = stats["mean_m"].as_f64().expect("mean_m");
    assert!(min_m >= 1, "min_m must be >= 1, got {min_m}");
    assert_eq!(min_m, max_m, "single Lanczos run: min_m must equal max_m");
    #[allow(clippy::cast_precision_loss)]
    let min_m_f = min_m as f64;
    assert!(
        (mean_m - min_m_f).abs() < 1e-12,
        "mean_m must equal min_m for single Lanczos run: mean_m={mean_m}, min_m={min_m}"
    );
    std::fs::remove_file(&cfg).ok();
}
