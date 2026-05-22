//! Density mixing strategies for the SCF loop.
//!
//! v0.1 shipped `Linear` only. v0.2 adds `Pulay` (DIIS extrapolation
//! over a rolling history) which converges the BALDA Mott regime where
//! linear mixing diverges. See
//! `notes/Zettelkasten/PermanentNote/canonical-ks-construction.md` for
//! the failure-mode discussion this addresses.

use crate::config::Mixing;
use faer::Mat;

/// Mixing dispatcher.
#[derive(Debug, Clone)]
pub enum DensityMixer {
    /// Simple linear mixing `α n_new + (1-α) n_old`.
    Linear {
        /// Mixing fraction `α ∈ (0, 1]`.
        alpha: f64,
    },
    /// Pulay / DIIS mixing with rolling history.
    Pulay(PulayState),
}

/// Internal state of a Pulay mixer — history of (density, residual) pairs.
#[derive(Debug, Clone)]
pub struct PulayState {
    /// Mixing fraction applied to the residual when extrapolating.
    pub alpha: f64,
    /// Maximum number of past pairs retained.
    pub history_depth: usize,
    /// Past density vectors `n_k` (each of length `num_sites`).
    densities: Vec<Vec<f64>>,
    /// Past residuals `r_k = n_new_k - n_k`.
    residuals: Vec<Vec<f64>>,
}

impl PulayState {
    /// Construct an empty Pulay mixer state with the given parameters.
    #[must_use]
    pub fn new(alpha: f64, history_depth: usize) -> Self {
        Self {
            alpha,
            history_depth: history_depth.max(2),
            densities: Vec::new(),
            residuals: Vec::new(),
        }
    }

    fn push(&mut self, density: &[f64], residual: &[f64]) {
        if self.densities.len() >= self.history_depth {
            self.densities.remove(0);
            self.residuals.remove(0);
        }
        self.densities.push(density.to_vec());
        self.residuals.push(residual.to_vec());
    }
}

impl DensityMixer {
    /// Build from config.
    #[must_use]
    pub fn from_config(cfg: &Mixing) -> Self {
        match cfg {
            Mixing::Linear { alpha } => Self::Linear { alpha: *alpha },
            Mixing::Pulay {
                alpha,
                history_depth,
            } => Self::Pulay(PulayState::new(*alpha, *history_depth)),
        }
    }

    /// Mix `new` (Pratt output) into `old` (current SCF density) in place.
    ///
    /// For Pulay this consumes the `(old, new-old)` pair into the rolling
    /// history and produces a DIIS extrapolation. For Linear it is the
    /// usual α-blend.
    pub fn mix_in_place(&mut self, old: &mut [f64], new: &[f64]) {
        assert_eq!(old.len(), new.len(), "old and new must have equal length");
        match self {
            Self::Linear { alpha } => {
                for (o, n) in old.iter_mut().zip(new.iter()) {
                    *o = alpha.mul_add(*n - *o, *o);
                }
            }
            Self::Pulay(state) => pulay_step(state, old, new),
        }
    }
}

/// Perform one Pulay/DIIS step.
fn pulay_step(state: &mut PulayState, old: &mut [f64], new: &[f64]) {
    let residual: Vec<f64> = old.iter().zip(new.iter()).map(|(o, n)| n - o).collect();
    state.push(old, &residual);

    if state.residuals.len() < 2 {
        // Not enough history: fall back to linear mixing.
        for (o, n) in old.iter_mut().zip(new.iter()) {
            *o = state.alpha.mul_add(*n - *o, *o);
        }
        return;
    }

    let Some(coeffs) = solve_pulay_coefficients(&state.residuals) else {
        // Ill-conditioned — fall back to linear.
        for (o, n) in old.iter_mut().zip(new.iter()) {
            *o = state.alpha.mul_add(*n - *o, *o);
        }
        return;
    };

    // n_next = Σ c_i (n_i + α r_i)
    let len = old.len();
    let mut next = vec![0.0_f64; len];
    for (i, c_i) in coeffs.iter().enumerate() {
        let n_i = &state.densities[i];
        let r_i = &state.residuals[i];
        for j in 0..len {
            next[j] += c_i * state.alpha.mul_add(r_i[j], n_i[j]);
        }
    }
    old.copy_from_slice(&next);
}

/// Solve for DIIS coefficients minimizing `||Σ c_i r_i||²` subject to
/// `Σ c_i = 1` (Lagrange multiplier formulation).
///
/// Builds the `(m+1) × (m+1)` block matrix
/// ```text
/// | B    1 | | c |     | 0 |
/// | 1^T  0 | | λ |  =  | 1 |
/// ```
/// where `B[i][j] = <r_i, r_j>`. Returns `None` if the solve is
/// numerically singular.
fn solve_pulay_coefficients(residuals: &[Vec<f64>]) -> Option<Vec<f64>> {
    let m = residuals.len();
    assert!(m >= 2);
    let size = m + 1;

    let mut a = Mat::<f64>::zeros(size, size);
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..residuals[i].len() {
                s += residuals[i][k] * residuals[j][k];
            }
            a[(i, j)] = s;
        }
        a[(i, m)] = 1.0;
        a[(m, i)] = 1.0;
    }
    a[(m, m)] = 0.0;

    let mut rhs = vec![0.0_f64; size];
    rhs[m] = 1.0;

    gauss_solve(&mut a, &mut rhs)?;

    let coeffs = rhs[..m].to_vec();
    if coeffs.iter().any(|c| !c.is_finite()) {
        return None;
    }
    Some(coeffs)
}

/// Tiny in-place Gaussian elimination with partial pivoting on the small
/// `(m+1) × (m+1)` DIIS system. Returns `None` if the pivot collapses.
fn gauss_solve(a: &mut Mat<f64>, b: &mut [f64]) -> Option<()> {
    let n = a.nrows();
    assert_eq!(n, a.ncols());
    assert_eq!(n, b.len());
    for k in 0..n {
        // Partial pivoting.
        let mut pivot = k;
        let mut pivot_val = a[(k, k)].abs();
        for i in (k + 1)..n {
            let v = a[(i, k)].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot = i;
            }
        }
        if pivot_val < 1e-14 {
            return None;
        }
        if pivot != k {
            for j in 0..n {
                let tmp = a[(k, j)];
                a[(k, j)] = a[(pivot, j)];
                a[(pivot, j)] = tmp;
            }
            b.swap(k, pivot);
        }
        for i in (k + 1)..n {
            let factor = a[(i, k)] / a[(k, k)];
            for j in k..n {
                let v = a[(k, j)];
                a[(i, j)] -= factor * v;
            }
            b[i] -= factor * b[k];
        }
    }
    // Back-substitute.
    for k in (0..n).rev() {
        let mut s = b[k];
        for j in (k + 1)..n {
            s -= a[(k, j)] * b[j];
        }
        b[k] = s / a[(k, k)];
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_mix_alpha_one_replaces() {
        let mut mixer = DensityMixer::Linear { alpha: 1.0 };
        let mut old = vec![0.0, 1.0, 2.0];
        mixer.mix_in_place(&mut old, &[10.0, 11.0, 12.0]);
        assert_eq!(old, vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn linear_mix_alpha_zero_freezes() {
        let mut mixer = DensityMixer::Linear { alpha: 0.0 };
        let mut old = vec![0.0, 1.0, 2.0];
        mixer.mix_in_place(&mut old, &[10.0, 11.0, 12.0]);
        assert_eq!(old, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn linear_mix_half() {
        let mut mixer = DensityMixer::Linear { alpha: 0.5 };
        let mut old = vec![0.0, 1.0];
        mixer.mix_in_place(&mut old, &[2.0, 3.0]);
        assert_eq!(old, vec![1.0, 2.0]);
    }

    #[test]
    fn pulay_falls_back_to_linear_until_history_filled() {
        let mut mixer = DensityMixer::Pulay(PulayState::new(0.5, 4));
        let mut old = vec![0.0, 1.0];
        // First step: history len 1 after push -> falls back to linear (alpha=0.5).
        mixer.mix_in_place(&mut old, &[2.0, 3.0]);
        assert_eq!(old, vec![1.0, 2.0]);
    }

    #[test]
    fn pulay_converges_to_fixed_point_on_affine_map() {
        // f(x) = 0.5 * (x + target). Fixed point = target.
        let target = [1.0_f64, -1.0];
        let mut mixer = DensityMixer::Pulay(PulayState::new(0.7, 8));
        let mut x = vec![5.0_f64, 5.0];
        for _ in 0..50 {
            let n: Vec<f64> = x
                .iter()
                .zip(target.iter())
                .map(|(xi, ti)| 0.5 * (xi + ti))
                .collect();
            mixer.mix_in_place(&mut x, &n);
        }
        for (xi, ti) in x.iter().zip(target.iter()) {
            assert!(
                (xi - ti).abs() < 1e-8,
                "Pulay did not converge: {xi} vs {ti}"
            );
        }
    }
}
