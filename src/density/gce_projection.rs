#![allow(clippy::similar_names)]
//! Canonical density via grand-canonical-plus-projection.
//!
//! Identity (per spin sector with `N_σ` particles in `n` single-particle
//! levels with energies `{ε_k}` and orbitals `{ψ_k}`):
//!
//! ```text
//! Z_{N_σ}(β) = (1/M) Σ_{j=0}^{M-1} z_j^{-N_σ} ∏_k (1 + z_j e^{-β ε_k})
//! n_i^σ      = (1/Z_{N_σ}) · (1/M) Σ_j z_j^{-N_σ} P_j Σ_k |ψ_k(i)|² f_k(z_j)
//! ```
//!
//! where `z_j = e^{i 2π j / M}` is the fugacity placed on the unit circle
//! and `f_k(z) = z e^{-β ε_k} / (1 + z e^{-β ε_k})` is the twisted
//! Fermi-Dirac factor. For `M ≥ N_σ + 1` the discrete contour integral
//! is exact (modulo round-off).
//!
//! Acts as an algorithmic cross-check against the Pratt recursion.

use super::DensityResult;
use crate::config::GceProjectionParams;
use crate::error::Result;
use crate::spectrum::Eigendecomposition;

/// Runtime mirror of [`GceProjectionParams`].
#[derive(Debug, Clone, Copy)]
pub struct GceProjectionParamsRuntime {
    /// Number of unit-circle fugacity quadrature points.
    pub num_quadrature_points: usize,
    /// Subtract `ε_min` before exponentiation for numerical stability.
    pub spectrum_shift: bool,
}

impl From<&GceProjectionParams> for GceProjectionParamsRuntime {
    fn from(p: &GceProjectionParams) -> Self {
        Self {
            num_quadrature_points: p.num_quadrature_points,
            spectrum_shift: p.spectrum_shift,
        }
    }
}

/// Evaluate `({n_i}, Z_N)` by grand-canonical+projection quadrature.
pub fn evaluate(
    eigen: &Eigendecomposition,
    num_electrons_per_spin: usize,
    beta: f64,
    params: &GceProjectionParamsRuntime,
) -> Result<DensityResult> {
    let num_sites = eigen.eigenvalues.len();
    assert_eq!(
        num_sites,
        eigen.eigenvectors.nrows(),
        "spectrum size mismatch: {num_sites} eigvals vs {} rows",
        eigen.eigenvectors.nrows()
    );
    let n_sigma = num_electrons_per_spin;
    let m = params.num_quadrature_points.max(n_sigma + 1);

    let shift = if params.spectrum_shift && !eigen.eigenvalues.is_empty() {
        eigen.eigenvalues[0]
    } else {
        0.0
    };

    let exp_eps: Vec<f64> = eigen
        .eigenvalues
        .iter()
        .map(|&eps| (-beta * (eps - shift)).exp())
        .collect();

    let mut psi_sq = vec![0.0_f64; eigen.eigenvalues.len() * num_sites];
    for k in 0..eigen.eigenvalues.len() {
        for i in 0..num_sites {
            let v = eigen.eigenvectors[(i, k)];
            psi_sq[k * num_sites + i] = v * v;
        }
    }

    let mut z_acc_re = 0.0_f64;
    let mut z_acc_im = 0.0_f64;
    let mut n_acc_re = vec![0.0_f64; num_sites];
    let mut n_acc_im = vec![0.0_f64; num_sites];

    for j in 0..m {
        let phi = std::f64::consts::TAU * (j as f64) / (m as f64);
        let (z_im, z_re) = phi.sin_cos();

        let mut p_re = 1.0_f64;
        let mut p_im = 0.0_f64;
        let mut grad_re = vec![0.0_f64; num_sites];
        let mut grad_im = vec![0.0_f64; num_sites];

        for (k, &ek) in exp_eps.iter().enumerate() {
            let a_re = z_re.mul_add(ek, 1.0);
            let a_im = z_im * ek;
            let new_p_re = p_re.mul_add(a_re, -p_im * a_im);
            let new_p_im = p_re.mul_add(a_im, p_im * a_re);
            p_re = new_p_re;
            p_im = new_p_im;

            let denom = a_re.mul_add(a_re, a_im * a_im);
            let num_re = z_re * ek;
            let num_im = z_im * ek;
            let f_re = num_re.mul_add(a_re, num_im * a_im) / denom;
            let f_im = num_im.mul_add(a_re, -num_re * a_im) / denom;
            for i in 0..num_sites {
                let w = psi_sq[k * num_sites + i];
                grad_re[i] = w.mul_add(f_re, grad_re[i]);
                grad_im[i] = w.mul_add(f_im, grad_im[i]);
            }
        }

        let arg = -phi * (n_sigma as f64);
        let (zn_im, zn_re) = arg.sin_cos();

        let weight_re = zn_re.mul_add(p_re, -zn_im * p_im);
        let weight_im = zn_re.mul_add(p_im, zn_im * p_re);

        z_acc_re += weight_re;
        z_acc_im += weight_im;
        for i in 0..num_sites {
            let gradi_re = grad_re[i];
            let gradi_im = grad_im[i];
            n_acc_re[i] += weight_re.mul_add(gradi_re, -weight_im * gradi_im);
            n_acc_im[i] += weight_re.mul_add(gradi_im, weight_im * gradi_re);
        }
    }

    let inv_m = 1.0 / (m as f64);
    let z_shifted = z_acc_re * inv_m;
    debug_assert!(
        (z_acc_im * inv_m).abs() < 1.0e-8_f64.max(z_shifted.abs() * 1.0e-8),
        "GCE+projection: imaginary residual {} too large (Z = {z_shifted})",
        z_acc_im * inv_m,
    );

    let mut densities = vec![0.0_f64; num_sites];
    if n_sigma > 0 && z_shifted.abs() > 0.0 {
        for i in 0..num_sites {
            densities[i] = 2.0 * (n_acc_re[i] * inv_m) / z_shifted;
        }
    }

    let n_sigma_f = n_sigma as f64;
    let log_z_per_spin = (n_sigma_f * beta).mul_add(-shift, z_shifted.ln());
    let z_per_spin = log_z_per_spin.exp();
    let z_total = z_per_spin * z_per_spin;

    Ok(DensityResult {
        densities,
        partition_function_per_spin: z_per_spin,
        partition_function: z_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::pratt;
    use faer::Mat;

    fn default_params() -> GceProjectionParamsRuntime {
        GceProjectionParamsRuntime {
            num_quadrature_points: 64,
            spectrum_shift: true,
        }
    }

    fn manual_eigendecomp(eigenvalues: Vec<f64>, eigenvectors: Mat<f64>) -> Eigendecomposition {
        Eigendecomposition {
            eigenvalues,
            eigenvectors,
        }
    }

    fn pratt_runtime() -> pratt::PrattParamsRuntime {
        pratt::PrattParamsRuntime::from(&crate::config::PrattParams::default())
    }

    #[test]
    fn matches_pratt_on_diagonal_l4_n2() {
        let n = 4;
        let mut eigvecs = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eigvecs[(i, i)] = 1.0;
        }
        let eig = manual_eigendecomp(vec![-2.0, -1.0, 1.0, 2.0], eigvecs);
        let p = pratt::evaluate(&eig, 2, 1.0, &pratt_runtime()).unwrap();
        let g = evaluate(&eig, 2, 1.0, &default_params()).unwrap();
        assert!(
            (p.partition_function - g.partition_function).abs() < 1e-9 * p.partition_function.abs(),
            "Z mismatch pratt={} gce={}",
            p.partition_function,
            g.partition_function,
        );
        for i in 0..n {
            assert!(
                (p.densities[i] - g.densities[i]).abs() < 1e-9,
                "n[{i}] mismatch pratt={} gce={}",
                p.densities[i],
                g.densities[i],
            );
        }
    }

    #[test]
    fn matches_pratt_on_random_l5_n3() {
        let n = 5;
        let mut h = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = ((((i * 7 + j * 11 + 3) % 23) as f64) - 11.0) * 0.15;
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        let eig = crate::spectrum::dense_diag::diagonalize(&h).unwrap();
        let p = pratt::evaluate(&eig, 3, 1.5, &pratt_runtime()).unwrap();
        let g = evaluate(&eig, 3, 1.5, &default_params()).unwrap();
        assert!(
            (p.partition_function - g.partition_function).abs() < 1e-9 * p.partition_function.abs(),
        );
        for i in 0..n {
            assert!((p.densities[i] - g.densities[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn n_zero_yields_zero_density() {
        let n = 3;
        let mut eigvecs = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eigvecs[(i, i)] = 1.0;
        }
        let eig = manual_eigendecomp(vec![-1.0, 0.0, 1.0], eigvecs);
        let r = evaluate(&eig, 0, 2.0, &default_params()).unwrap();
        for &n_i in &r.densities {
            assert!(n_i.abs() < 1e-12);
        }
    }
}
