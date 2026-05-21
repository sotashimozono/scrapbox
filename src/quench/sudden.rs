//! Sudden quench evaluator — Palamara 2024 eq 23, 25.

/// Inputs needed to evaluate sudden-quench observables.
#[derive(Debug, Clone)]
pub struct SuddenQuenchEvaluator {
    /// Pre-quench external potential `V_i^0`.
    pub initial_potential: Vec<f64>,
    /// Post-quench external potential `V_i^f`.
    pub final_potential: Vec<f64>,
    /// Inverse temperature `β`.
    pub beta: f64,
}

impl SuddenQuenchEvaluator {
    /// Construct an evaluator.
    #[must_use]
    pub fn new(initial_potential: Vec<f64>, final_potential: Vec<f64>, beta: f64) -> Self {
        Self {
            initial_potential,
            final_potential,
            beta,
        }
    }

    /// `<W> = Σ_i (V_i^f − V_i^0) · n_i^β,0`.
    #[must_use]
    pub fn mean_work(&self, initial_density: &[f64]) -> f64 {
        assert_eq!(self.initial_potential.len(), initial_density.len());
        assert_eq!(self.initial_potential.len(), self.final_potential.len());
        self.initial_potential
            .iter()
            .zip(self.final_potential.iter())
            .zip(initial_density.iter())
            .map(|((v0, vf), n)| (vf - v0) * n)
            .sum()
    }

    /// `<S_irr> = β (<W> − ΔF)`.
    #[must_use]
    pub fn irreversible_entropy(
        &self,
        mean_work: f64,
        initial_free_energy: f64,
        final_free_energy: f64,
    ) -> f64 {
        let delta_f = final_free_energy - initial_free_energy;
        self.beta * (mean_work - delta_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_quench_has_zero_work() {
        let e = SuddenQuenchEvaluator::new(vec![0.5, -0.5], vec![0.5, -0.5], 2.0);
        assert!((e.mean_work(&[1.0, 1.0])).abs() < 1e-12);
    }

    #[test]
    fn entropy_uses_beta_times_work_minus_df() {
        let e = SuddenQuenchEvaluator::new(vec![0.0], vec![0.0], 3.0);
        let s = e.irreversible_entropy(2.0, 1.0, 1.5);
        assert!((s - 4.5).abs() < 1e-12);
    }
}
