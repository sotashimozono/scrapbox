use faer::Mat;
use rand::Rng;

/// Generates a normalized random probe vector (disguised typical state).
pub fn generate_random_probe(dimension: usize) -> Mat<f64> {
    let mut rng = rand::thread_rng();
    let mut vec = Mat::<f64>::zeros(dimension, 1);
    
    let mut norm_sq = 0.0;
    for i in 0..dimension {
        // Uniform random for demonstration, easily upgraded to Box-Muller Gaussian
        let val: f64 = rng.gen_range(-1.0..1.0);
        vec[(i, 0)] = val;
        norm_sq += val * val;
    }
    
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for i in 0..dimension {
            vec[(i, 0)] /= norm;
        }
    }
    vec
}
