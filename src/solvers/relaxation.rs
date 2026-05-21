use faer::Mat;

pub struct RelaxationSolver {
    pub system_matrix: Mat<f64>,
}

impl RelaxationSolver {
    pub fn new(matrix: Mat<f64>) -> Self {
        Self { system_matrix: matrix }
    }

    /// Propagates the state vector using relaxation steps (equivalent to imaginary-time evolution).
    pub fn propagate(&self, state: &mut Mat<f64>, _steps: usize, _step_size: f64) {
        let dt = _step_size;
        let mut h_state = Mat::<f64>::zeros(state.nrows(), state.ncols());
        
        // Toy simulation multiplication (H * Psi)
        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                let mut sum = 0.0;
                for k in 0..self.system_matrix.ncols() {
                    sum += self.system_matrix[(i, k)] * state[(k, j)];
                }
                h_state[(i, j)] = sum;
            }
        }

        // Psi = Psi - dt * (H * Psi)
        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                state[(i, j)] -= dt * h_state[(i, j)];
            }
        }
    }
}
