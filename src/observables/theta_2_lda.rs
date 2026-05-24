#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::suboptimal_flops,
    clippy::doc_markdown
)]
//! LDA quantum correction `Theta_2` for the generalized FDR
//! (Palamara 2024 eq 31).
//!
//! This is the v0.5 **first-cut** estimate for `theta_2.method = "lda"`.
//! It dispatches when BALDA xc is active, uses the BALDA `beta(u)`
//! Lima parameter to weight a per-site variance kernel, and produces a
//! number with the right symmetries (vanishes at `U=0` and for
//! commuting quenches). Numerical agreement with the exact `Theta_2`
//! is **not** claimed; refinement to the full Palamara Sec III.3
//! derivation is tracked in issue #22.
//!
//! ## Formula (placeholder)
//!
//! ```text
//! Theta_2_LDA = sum_i [delta V_i - mean(delta V)]^2
//!             * n_i (2 - n_i) / 2
//!             * alpha_BALDA(u),
//!
//! alpha_BALDA(u) = (2 - beta_Lima(u)) / 2.
//! ```

use crate::config::XcFunctional as XcConfig;
use crate::error::{Result, ScrapboxError};
use crate::xc::balda::Balda;

/// Density tolerance: `densities` outside `[-N_TOL, 2 + N_TOL]` are
/// rejected as non-physical. Pratt recursion + Pulay can leak ~1e-12
/// into the negative direction; anything larger is a real upstream
/// bug, not numerical noise.
const N_TOL: f64 = 1.0e-10;

/// LDA estimate of `Theta_2` per the placeholder formula in the module
/// doc.
///
/// # Parameters
/// - `on_site_u_over_t`: dimensionless interaction `u = U / t`. Must be
///   non-negative — BALDA assumes repulsive Hubbard.
/// - `delta_v`: per-site `V_final - V_initial` for the quench, in units
///   of `t`. Centered internally (translation gauge of the one-body
///   potential leaves `H` invariant up to a c-number).
/// - `densities`: per-site spin-summed occupations `n_i in [0, 2]` from
///   the converged KS state.
/// - `xc_cfg`: must be `XcConfig::Balda { .. }`; other variants return
///   `ConfigValidation` because the formula uses BALDA's `beta_Lima(u)`.
///
/// # Caveat
/// First-cut placeholder. Symmetries (zero at `U=0`, zero for commuting
/// quenches, zero at empty/full bands) are exact; the magnitude is not
/// the exact Palamara `Theta_2`. Refinement tracked in issue #22.
pub fn lda_theta_2(
    on_site_u_over_t: f64,
    delta_v: &[f64],
    densities: &[f64],
    xc_cfg: &XcConfig,
) -> Result<f64> {
    assert_eq!(
        delta_v.len(),
        densities.len(),
        "delta_v and densities length mismatch ({} vs {})",
        delta_v.len(),
        densities.len()
    );
    if on_site_u_over_t < 0.0 {
        return Err(ScrapboxError::ConfigValidation {
            message: format!(
                "theta_2.method = \"lda\" requires non-negative U/t \
                 (BALDA is defined for repulsive Hubbard); got {on_site_u_over_t}"
            ),
        });
    }
    let balda_params = match xc_cfg {
        XcConfig::Balda { params } => params.clone(),
        _ => {
            return Err(ScrapboxError::ConfigValidation {
                message: "theta_2.method = \"lda\" requires xc_functional.kind = \"balda\" \
                     (the formula uses BALDA's beta_Lima(u) coupling)"
                    .into(),
            });
        }
    };
    for (i, &n_raw) in densities.iter().enumerate() {
        if !(-N_TOL..=2.0 + N_TOL).contains(&n_raw) {
            return Err(ScrapboxError::ConfigValidation {
                message: format!(
                    "lda_theta_2: density at site {i} = {n_raw} is outside the \
                     physical range [0, 2] (tolerance {N_TOL:e}); upstream \
                     density evaluator is producing non-physical values"
                ),
            });
        }
    }
    let balda = Balda::new(on_site_u_over_t, balda_params);
    let alpha_balda = (2.0 - balda.beta_u) / 2.0;

    let l = delta_v.len() as f64;
    let mean_dv = delta_v.iter().sum::<f64>() / l;

    let mut theta = 0.0_f64;
    for (&dv, &n_raw) in delta_v.iter().zip(densities.iter()) {
        let dv_centered = dv - mean_dv;
        let n = n_raw.clamp(0.0, 2.0);
        let kernel = n * (2.0 - n) / 2.0 * alpha_balda;
        theta += dv_centered * dv_centered * kernel;
    }
    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BaldaParams;

    fn balda_cfg() -> XcConfig {
        XcConfig::Balda {
            params: BaldaParams::default(),
        }
    }

    #[test]
    fn uniform_quench_yields_zero_theta() {
        let theta = lda_theta_2(4.0, &[0.5, 0.5, 0.5, 0.5], &[1.0; 4], &balda_cfg()).unwrap();
        assert!(theta.abs() < 1e-14, "got {theta}");
    }

    #[test]
    fn non_uniform_quench_yields_positive_theta() {
        let theta = lda_theta_2(4.0, &[0.3, -0.3], &[1.0, 1.0], &balda_cfg()).unwrap();
        assert!(theta > 0.0, "expected positive, got {theta}");
    }

    #[test]
    fn u_zero_yields_zero_regardless_of_quench() {
        let theta = lda_theta_2(0.0, &[0.3, -0.3], &[1.0, 1.0], &balda_cfg()).unwrap();
        assert!(theta.abs() < 1e-14, "got {theta}");
    }

    #[test]
    fn empty_band_density_yields_zero() {
        // n=0 -> kernel n*(2-n)/2 = 0 exactly. No clamp residual.
        let theta = lda_theta_2(4.0, &[0.3, -0.3], &[0.0, 0.0], &balda_cfg()).unwrap();
        assert!(theta.abs() < 1e-14, "got {theta}");
    }

    #[test]
    fn saturated_band_density_yields_zero() {
        // n=2 -> kernel n*(2-n)/2 = 0 exactly. Symmetric counterpart of empty band.
        let theta = lda_theta_2(4.0, &[0.3, -0.3], &[2.0, 2.0], &balda_cfg()).unwrap();
        assert!(theta.abs() < 1e-14, "got {theta}");
    }

    #[test]
    fn half_filling_maximizes_kernel() {
        // At n=1 the kernel n*(2-n)/2 = 0.5 is maximal; check theta > 0
        // and strictly larger than the same delta_v at n=0.5 (kernel 0.375).
        let theta_half = lda_theta_2(4.0, &[0.3, -0.3], &[1.0, 1.0], &balda_cfg()).unwrap();
        let theta_quarter = lda_theta_2(4.0, &[0.3, -0.3], &[0.5, 0.5], &balda_cfg()).unwrap();
        assert!(theta_half > theta_quarter && theta_quarter > 0.0);
    }

    #[test]
    fn large_u_saturates_alpha_to_one_half() {
        // beta_Lima(u) -> 1 as u -> infinity, so alpha_BALDA -> 1/2.
        // At u = 100 and n=1, delta=0.3 the expected theta is
        // 2 * (0.3^2) * (1 * 1 / 2) * (1/2) = 0.045 (within solver accuracy).
        let theta = lda_theta_2(100.0, &[0.3, -0.3], &[1.0, 1.0], &balda_cfg()).unwrap();
        assert!((theta - 0.045).abs() < 5e-3, "got {theta}");
    }

    #[test]
    fn errors_when_xc_is_not_balda() {
        let r = lda_theta_2(4.0, &[0.3, -0.3], &[1.0, 1.0], &XcConfig::NonInteracting);
        assert!(r.is_err());
    }

    #[test]
    fn errors_when_u_is_negative() {
        let r = lda_theta_2(-1.0, &[0.3, -0.3], &[1.0, 1.0], &balda_cfg());
        assert!(r.is_err());
    }

    #[test]
    fn errors_when_density_is_unphysical() {
        let r = lda_theta_2(4.0, &[0.3, -0.3], &[2.5, 1.0], &balda_cfg());
        assert!(r.is_err());
    }
}
