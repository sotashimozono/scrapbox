#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]
//! Many-body Hubbard ED runner exposed via the `scrapbox ed` CLI
//! subcommand.
//!
//! Three backends, selected by `[ed].solver` in the TOML config:
//!
//! - `"dense"`: full spectrum via `reference::ed::canonical_thermal`
//!   (gold standard; limited by `dim^2` memory).
//! - `"matrix_free_lanczos"`: low-`k` spectrum via Lanczos on
//!   `JwHubbard` (the v0.6 batch alpha operator). Scales further in
//!   `L` but only returns the requested number of eigenvalues.
//! - `"sparse_lanczos"`: reserved for v0.7 batch beta; currently
//!   returns `ConfigValidation`.
//!
//! Output JSON shape (no datetime field; pure numerics):
//!
//! ```json
//! {
//!   "solver": "matrix_free_lanczos",
//!   "dim": 400,
//!   "num_eigenvalues_returned": 4,
//!   "eigenvalues": [-7.123, -6.987, -6.41, -6.205],
//!   "wall_time_ms": 142
//! }
//! ```

use crate::config::{Config, EdSolver, EdSpec};
use crate::error::{Result, ScrapboxError};
use crate::reference::ed;
use crate::spectrum::hubbard_jw::JwHubbard;
use crate::spectrum::lanczos::{diagonalize, LanczosParams};
use crate::spectrum::linear_operator::LinearOperator;
use serde::Serialize;
use std::path::Path;
use std::time::Instant;

/// Output payload emitted by `scrapbox ed` to the run directory.
#[derive(Debug, Clone, Serialize)]
pub struct EdSpectrumOutput {
    /// Solver actually used (matches the resolved `EdSolver` variant).
    pub solver: &'static str,
    /// Hilbert dimension `C(L, n_up) * C(L, n_dn)`.
    pub dim: usize,
    /// Number of eigenvalues actually returned.
    pub num_eigenvalues_returned: usize,
    /// Ascending eigenvalues. Dense returns the full spectrum (or the
    /// first `num_eigenvalues` of it); Lanczos returns the lowest
    /// `num_eigenvalues` (clipped to `dim`).
    pub eigenvalues: Vec<f64>,
    /// Wall-clock time spent in the solver (excludes I/O).
    pub wall_time_ms: u128,
}

/// Run the ED subcommand: dispatch the requested solver, write the
/// resulting spectrum JSON to `<output.directory>/ed_spectrum.json`,
/// return the in-memory output.
pub fn run(cfg: &Config) -> Result<EdSpectrumOutput> {
    let ed_cfg = cfg
        .ed
        .as_ref()
        .ok_or_else(|| ScrapboxError::ConfigValidation {
            message: "[ed] section required for `scrapbox ed` (set solver and optional \
                      num_eigenvalues)"
                .into(),
        })?;
    let v_ext = cfg
        .hamiltonian
        .external_potential
        .to_site_values(cfg.hamiltonian.num_sites);
    let n_up = cfg.hamiltonian.num_electrons_per_spin;
    let n_dn = cfg.hamiltonian.num_electrons_per_spin;

    let out = match ed_cfg.solver {
        EdSolver::Dense => run_dense(cfg, &v_ext, n_up, n_dn, ed_cfg),
        EdSolver::MatrixFreeLanczos => run_matrix_free(cfg, &v_ext, n_up, n_dn, ed_cfg)?,
        EdSolver::SparseLanczos => run_sparse(cfg, &v_ext, n_up, n_dn, ed_cfg)?,
    };

    let out_dir = crate::bin_support::resolve_output_dir(cfg);
    std::fs::create_dir_all(&out_dir).map_err(|source| ScrapboxError::Artifact {
        path: out_dir.clone(),
        message: format!("failed to create output dir: {source}"),
    })?;
    let out_path = out_dir.join("ed_spectrum.json");
    write_output_json(&out_path, &out)?;
    Ok(out)
}

fn run_dense(
    cfg: &Config,
    v_ext: &[f64],
    n_up: usize,
    n_dn: usize,
    ed_cfg: &EdSpec,
) -> EdSpectrumOutput {
    let start = Instant::now();
    let result = ed::canonical_thermal(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_ext,
    );
    let elapsed = start.elapsed().as_millis();
    let dim = result.eigenvalues.len();
    let k = ed_cfg.num_eigenvalues.unwrap_or(dim).min(dim);
    EdSpectrumOutput {
        solver: "dense",
        dim,
        num_eigenvalues_returned: k,
        eigenvalues: result.eigenvalues.into_iter().take(k).collect(),
        wall_time_ms: elapsed,
    }
}

fn run_matrix_free(
    cfg: &Config,
    v_ext: &[f64],
    n_up: usize,
    n_dn: usize,
    ed_cfg: &EdSpec,
) -> Result<EdSpectrumOutput> {
    let jw = JwHubbard::new(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_ext,
    );
    let dim = jw.dim();
    let k_request = ed_cfg.num_eigenvalues.unwrap_or(8).min(dim);
    let krylov_dim = (k_request * 10).clamp(20, dim);
    let params = LanczosParams {
        krylov_dim: Some(krylov_dim),
        max_iter: krylov_dim * 4,
        tol: 1e-12,
    };
    let start = Instant::now();
    let eig = diagonalize(&jw, &params)?;
    let elapsed = start.elapsed().as_millis();
    let mut vals = eig.eigenvalues;
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.truncate(k_request);
    Ok(EdSpectrumOutput {
        solver: "matrix_free_lanczos",
        dim,
        num_eigenvalues_returned: vals.len(),
        eigenvalues: vals,
        wall_time_ms: elapsed,
    })
}

fn write_output_json(path: &Path, out: &EdSpectrumOutput) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|source| ScrapboxError::Artifact {
        path: path.to_path_buf(),
        message: format!("failed to write ed_spectrum.json: {source}"),
    })?;
    serde_json::to_writer_pretty(file, out)?;
    Ok(())
}
fn run_sparse(
    cfg: &Config,
    v_ext: &[f64],
    n_up: usize,
    n_dn: usize,
    ed_cfg: &EdSpec,
) -> Result<EdSpectrumOutput> {
    let sparse = crate::spectrum::linear_operator::SparseMatrix::from_hubbard(
        cfg.hamiltonian.num_sites,
        n_up,
        n_dn,
        cfg.hamiltonian.hopping_j,
        cfg.hamiltonian.on_site_interaction,
        v_ext,
    );
    let dim = sparse.dim();
    let k_request = ed_cfg.num_eigenvalues.unwrap_or(8).min(dim);
    let krylov_dim = (k_request * 10).clamp(20, dim);
    let params = LanczosParams {
        krylov_dim: Some(krylov_dim),
        max_iter: krylov_dim * 4,
        tol: 1e-12,
    };
    let start = Instant::now();
    let eig = diagonalize(&sparse, &params)?;
    let elapsed = start.elapsed().as_millis();
    let mut vals = eig.eigenvalues;
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.truncate(k_request);
    Ok(EdSpectrumOutput {
        solver: "sparse_lanczos",
        dim,
        num_eigenvalues_returned: vals.len(),
        eigenvalues: vals,
        wall_time_ms: elapsed,
    })
}
