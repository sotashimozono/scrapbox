//! # scrapbox
//!
//! A generic matrix relaxation solver and iterative fixed-point solver.

pub mod solvers;
pub mod utils;

pub use solvers::{relaxation::RelaxationSolver, fixed_point::FixedPointSolver};

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn test_relaxation_propagation() {
        // Simple diagonal system matrix (H = Identity)
        let matrix = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let solver = RelaxationSolver::new(matrix);

        let mut state = Mat::<f64>::from_fn(2, 1, |_, _| 1.0);
        solver.propagate(&mut state, 1, 0.1);

        // state = state - dt * H * state = 1.0 - 0.1 * 1.0 = 0.9
        assert!((state[(0, 0)] - 0.9).abs() < 1e-9);
        assert!((state[(1, 0)] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_fixed_point_solver() {
        let solver = FixedPointSolver::new(1e-5, 100);
        let initial = Mat::<f64>::from_fn(2, 2, |_, _| 0.0);
        
        let result = solver.solve(&initial).unwrap();
        // Should converge towards the Identity matrix
        assert!((result[(0, 0)] - 1.0).abs() < 1e-4);
        assert!((result[(0, 1)] - 0.0).abs() < 1e-4);
        assert!((result[(1, 1)] - 1.0).abs() < 1e-4);
    }
}
