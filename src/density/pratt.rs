//! Pratt-Borrmann-Franke recursion (Layer 4 default).
//!
//! Implements the per-spin canonical density via the analytical-derivative
//! pathway from `notes/discipline/canonical_thermal_dft.md` Sec IV/VII:
//!
//! ```text
//! Z_1(mβ)   = Σ_k exp(-m β ε_k)
//! Z_k       = (1/k) · Σ_{m=1..k} (-1)^{m-1} · Z_1(mβ) · Z_{k-m}     (Z_0 = 1)
//!
//! ∂Z_1(mβ)/∂V_i = -m β · Σ_k exp(-m β ε_k) · |ψ_ki|²
//! ∂Z_k/∂V_i     = (1/k) · Σ_{m=1..k} (-1)^{m-1}
//!                  · ( ∂Z_1(mβ)/∂V_i · Z_{k-m}  +  Z_1(mβ) · ∂Z_{k-m}/∂V_i )
//!
//! n_i^σ     = -(1/β) · (1/Z_{N_σ}) · ∂Z_{N_σ}/∂V_i      (one spin)
//! n_i       = 2 · n_i^σ                                  (paramagnetic)
//! Z_total   = (Z_{N_σ})²
//! F         = -(1/β) · ln Z_total = -(2/β) · ln Z_{N_σ}
//! ```
//!
//! Numerical stability: by default the input spectrum is shifted by
//! `ε_min` before the exponentials are evaluated. The shift cancels in the
//! density (which is a ratio) and is added back into the free energy at the
//! end. See `notes/Zettelkasten/PermanentNote/pratt-recursion.md`.

use super::DensityResult;
use crate::config::PrattParams;
use crate::error::Result;
use crate::spectrum::Eigendecomposition;

/// Runtime mirror of [`PrattParams`].
#[derive(Debug, Clone, Copy)]
pub struct PrattParamsRuntime {
    /// Subtract `ε_min` before recursion to avoid overflow / underflow.
    pub spectrum_shift: bool,
}

impl From<&PrattParams> for PrattParamsRuntime {
    fn from(p: &PrattParams) -> Self {
        Self {
            spectrum_shift: p.spectrum_shift,
        }
    }
}

/// Evaluate the canonical density and partition function from a KS spectrum.
pub fn evaluate(
    eigen: &Eigendecomposition,
    num_electrons_per_spin: usize,
    beta: f64,
    params: &PrattParamsRuntime,
) -> Result<DensityResult> {
    let num_sites = eigen.eigenvalues.len();
    assert_eq!(
        num_sites,
        eigen.eigenvectors.nrows(),
        "spectrum size mismatch: {num_sites} eigvals vs {} rows",
        eigen.eigenvectors.nrows()
    );

    let n_sigma = num_electrons_per_spin;
    let shift = if params.spectrum_shift && !eigen.eigenvalues.is_empty() {
        eigen.eigenvalues[0]
    } else {
        0.0
    };

    // Z_1(mβ) for m = 1..=n_sigma, working with shifted eigenvalues.
    let mut z1 = vec![0.0_f64; n_sigma + 1];
    for m in 1..=n_sigma {
        let m_f = m as f64;
        let mut sum = 0.0;
        for &eps in &eigen.eigenvalues {
            sum += (-m_f * beta * (eps - shift)).exp();
        }
        z1[m] = sum;
    }

    // Pratt-Borrmann-Franke recursion for Z_k.
    let mut z_canon = vec![0.0_f64; n_sigma + 1];
    z_canon[0] = 1.0;
    for k in 1..=n_sigma {
        let mut sum = 0.0;
        for m in 1..=k {
            let sign = if (m - 1) % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * z1[m] * z_canon[k - m];
        }
        z_canon[k] = sum / (k as f64);
    }

    // ∂Z_1(mβ)/∂V_i — stored as flat (m, i) Vec<f64> of length (n_sigma+1)*num_sites.
    let mut dz1 = vec![0.0_f64; (n_sigma + 1) * num_sites];
    for m in 1..=n_sigma {
        let m_f = m as f64;
        for i in 0..num_sites {
            let mut sum = 0.0;
            for (k, &eps) in eigen.eigenvalues.iter().enumerate() {
                let psi = eigen.eigenvectors[(i, k)];
                sum += (-m_f * beta * (eps - shift)).exp() * psi * psi;
            }
            dz1[m * num_sites + i] = -m_f * beta * sum;
        }
    }

    // ∂Z_k/∂V_i recursion.
    let mut dz_canon = vec![0.0_f64; (n_sigma + 1) * num_sites];
    for k in 1..=n_sigma {
        for i in 0..num_sites {
            let mut sum = 0.0;
            for m in 1..=k {
                let sign = if (m - 1) % 2 == 0 { 1.0 } else { -1.0 };
                let dz1_mi = dz1[m * num_sites + i];
                let dzc_kmi = dz_canon[(k - m) * num_sites + i];
                sum += sign * (dz1_mi.mul_add(z_canon[k - m], z1[m] * dzc_kmi));
            }
            dz_canon[k * num_sites + i] = sum / (k as f64);
        }
    }

    // Per-spin density: n_i^σ = -(1/β) · (1/Z_{N_σ}) · ∂Z_{N_σ}/∂V_i.
    // (Shift cancels in this ratio.)
    let z_per_spin_shifted = z_canon[n_sigma];
    let mut densities = vec![0.0_f64; num_sites];
    if n_sigma > 0 {
        let inv_beta_z = -1.0 / (beta * z_per_spin_shifted);
        for i in 0..num_sites {
            let dz_n_sigma = dz_canon[n_sigma * num_sites + i];
            // n_i (total, both spins) = 2 · n_i^σ.
            densities[i] = 2.0 * inv_beta_z * dz_n_sigma;
        }
    }

    // Partition function — restore the shift in the absolute value
    // (densities don't need this since they used a ratio).
    //
    // ln Z_per_spin = ln Z_shifted + (-N_σ β shift)
    //  =>  Z_per_spin = Z_shifted · exp(-N_σ β shift)
    let n_sigma_f = n_sigma as f64;
    let log_z_per_spin = (n_sigma_f * beta).mul_add(-shift, z_per_spin_shifted.ln());
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
    use crate::config::PrattParams;
    use faer::Mat;

    fn params() -> PrattParamsRuntime {
        PrattParamsRuntime::from(&PrattParams::default())
    }

    fn manual_eigendecomp(eigenvalues: Vec<f64>, eigenvectors: Mat<f64>) -> Eigendecomposition {
        Eigendecomposition {
            eigenvalues,
            eigenvectors,
        }
    }

    #[test]
    fn pratt_sum_rule_equals_n() {
        // L=4 sites, N_↑ = N_↓ = 2. Use arbitrary spectrum {-2, -1, 1, 2}
        // and orthonormal eigenvectors (identity matrix → diagonal H).
        let n = 4;
        let mut eigvecs = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eigvecs[(i, i)] = 1.0;
        }
        let eig = manual_eigendecomp(vec![-2.0, -1.0, 1.0, 2.0], eigvecs);
        let r = evaluate(&eig, 2, 1.0, &params()).unwrap();
        let total_n: f64 = r.densities.iter().sum();
        // Σ n_i should equal N = 2 · N_σ = 4 (modulo machine ε).
        assert!(
            (total_n - 4.0).abs() < 1e-10,
            "Pratt sum-rule broke: Σn = {total_n}"
        );
    }

    #[test]
    fn pratt_n_zero_yields_zero_density() {
        let n = 3;
        let mut eigvecs = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eigvecs[(i, i)] = 1.0;
        }
        let eig = manual_eigendecomp(vec![-1.0, 0.0, 1.0], eigvecs);
        let r = evaluate(&eig, 0, 2.0, &params()).unwrap();
        for &n_i in &r.densities {
            assert!(n_i.abs() < 1e-12);
        }
    }

    #[test]
    fn pratt_one_particle_per_spin_fills_lowest() {
        // Single particle per spin at very large β concentrates entirely
        // in the lowest eigenvector. With identity eigenvectors and
        // {-1, 0, 1} the lowest is site 0.
        let n = 3;
        let mut eigvecs = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            eigvecs[(i, i)] = 1.0;
        }
        let eig = manual_eigendecomp(vec![-1.0, 0.0, 1.0], eigvecs);
        let r = evaluate(&eig, 1, 50.0, &params()).unwrap();
        // n_0 → 2 (both spins on site 0), others → 0.
        assert!((r.densities[0] - 2.0).abs() < 1e-6);
        assert!(r.densities[1].abs() < 1e-6);
        assert!(r.densities[2].abs() < 1e-6);
    }
}
