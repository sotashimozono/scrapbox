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
