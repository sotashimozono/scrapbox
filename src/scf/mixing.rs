//! Density mixing strategies for the SCF loop.

use crate::config::Mixing;

/// Mixing dispatcher.
#[derive(Debug, Clone, Copy)]
pub enum DensityMixer {
    /// Simple linear mixing `α n_new + (1-α) n_old`.
    Linear {
        /// Mixing fraction `α ∈ (0, 1]`.
        alpha: f64,
    },
}

impl DensityMixer {
    /// Build from config.
    #[must_use]
    pub fn from_config(cfg: &Mixing) -> Self {
        match cfg {
            Mixing::Linear { alpha } => Self::Linear { alpha: *alpha },
        }
    }

    /// Combine the old and new densities in place into `old`.
    pub fn mix_in_place(&self, old: &mut [f64], new: &[f64]) {
        assert_eq!(old.len(), new.len(), "old and new must have equal length");
        match *self {
            Self::Linear { alpha } => {
                for (o, n) in old.iter_mut().zip(new.iter()) {
                    *o = alpha.mul_add(*n - *o, *o);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_mix_alpha_one_replaces() {
        let mixer = DensityMixer::Linear { alpha: 1.0 };
        let mut old = vec![0.0, 1.0, 2.0];
        mixer.mix_in_place(&mut old, &[10.0, 11.0, 12.0]);
        assert_eq!(old, vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn linear_mix_alpha_zero_freezes() {
        let mixer = DensityMixer::Linear { alpha: 0.0 };
        let mut old = vec![0.0, 1.0, 2.0];
        mixer.mix_in_place(&mut old, &[10.0, 11.0, 12.0]);
        assert_eq!(old, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn linear_mix_half() {
        let mixer = DensityMixer::Linear { alpha: 0.5 };
        let mut old = vec![0.0, 1.0];
        mixer.mix_in_place(&mut old, &[2.0, 3.0]);
        assert_eq!(old, vec![1.0, 2.0]);
    }
}
