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

/// LDA estimate of `Theta_2` per the placeholder formula in the module doc.
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
    let balda = Balda::new(on_site_u_over_t, balda_params);
    let alpha_balda = (2.0 - balda.beta_u) / 2.0;

    let l = delta_v.len() as f64;
    let mean_dv = delta_v.iter().sum::<f64>() / l;

    let mut theta = 0.0_f64;
    for (&dv, &n_raw) in delta_v.iter().zip(densities.iter()) {
        let dv_centered = dv - mean_dv;
        let n = n_raw.clamp(1.0e-12, 2.0 - 1.0e-12);
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
    fn empty_band_density_yields_near_zero() {
        // n=0 is clamped to eta=1e-12; residual ~ delta^2 * alpha * 1e-12 ~ 1e-13.
        let theta = lda_theta_2(4.0, &[0.3, -0.3], &[0.0, 0.0], &balda_cfg()).unwrap();
        assert!(theta.abs() < 1e-10, "got {theta}");
    }

    #[test]
    fn errors_when_xc_is_not_balda() {
        let r = lda_theta_2(4.0, &[0.3, -0.3], &[1.0, 1.0], &XcConfig::NonInteracting);
        assert!(r.is_err());
    }
}
