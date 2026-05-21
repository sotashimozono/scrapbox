use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

pub struct GridTransformer {
    pub size: usize,
    fft: Arc<dyn rustfft::Fft<f64>>,
}

impl GridTransformer {
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        Self { size, fft }
    }

    /// Transforms a complex grid representation to momentum/spectral space.
    pub fn transform(&self, buffer: &mut [Complex<f64>]) {
        assert_eq!(buffer.len(), self.size);
        self.fft.process(buffer);
    }
}
