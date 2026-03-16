//! Ptychographic Coherent Diffraction Imaging (CDI).
//!
//! Implements the extended Ptychographic Iterative Engine (ePIE) algorithm
//! for lensless imaging and mask metrology at X-ray wavelengths.

use ndarray::Array2;

use crate::math::fft2d::Fft2D;
use crate::types::Complex64;

/// Ptychography reconstruction configuration.
#[derive(Debug, Clone)]
pub struct PtychographyConfig {
    /// Number of ePIE iterations.
    pub num_iterations: usize,
    /// Object update step size (0 < α ≤ 1).
    pub object_step_size: f64,
    /// Probe update step size (0 < β ≤ 1).
    pub probe_step_size: f64,
}

impl Default for PtychographyConfig {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            object_step_size: 1.0,
            probe_step_size: 1.0,
        }
    }
}

/// Result of ptychographic reconstruction.
#[derive(Debug)]
pub struct PtychographyResult {
    /// Reconstructed object transmittance (complex).
    pub object: Array2<Complex64>,
    /// Refined probe function (complex).
    pub probe: Array2<Complex64>,
    /// Error metric per iteration (sum of squared differences).
    pub error_history: Vec<f64>,
}

/// Simulate diffraction patterns from a known object and probe.
///
/// For each probe position, computes:
///   pattern = |FFT(object_region × probe)|²
pub fn simulate_diffraction_patterns(
    object: &Array2<Complex64>,
    probe: &Array2<Complex64>,
    positions: &[(usize, usize)],
) -> Vec<Array2<f64>> {
    let fft = Fft2D::new();
    let (py, px) = probe.dim();

    positions
        .iter()
        .map(|&(row, col)| {
            let exit_wave = extract_and_multiply(object, probe, row, col, py, px);
            let mut spectrum = exit_wave;
            fft.forward(&mut spectrum);
            spectrum.mapv(|c| c.norm_sqr())
        })
        .collect()
}

/// Reconstruct object and probe from diffraction patterns using ePIE.
///
/// The ePIE algorithm alternates between:
/// 1. Forward propagation: exit_wave = object_region × probe
/// 2. Fourier constraint: replace amplitude with measured √(pattern), keep phase
/// 3. Backward update: adjust object and probe to match corrected exit wave
pub fn epie_reconstruct(
    diffraction_patterns: &[Array2<f64>],
    positions: &[(usize, usize)],
    initial_probe: &Array2<Complex64>,
    object_size: (usize, usize),
    config: &PtychographyConfig,
) -> PtychographyResult {
    let fft = Fft2D::new();
    let (py, px) = initial_probe.dim();
    let (obj_ny, obj_nx) = object_size;

    // Initialize object as uniform transmittance
    let mut object = Array2::from_elem((obj_ny, obj_nx), Complex64::new(1.0, 0.0));
    let mut probe = initial_probe.clone();
    let mut error_history = Vec::with_capacity(config.num_iterations);

    for _iter in 0..config.num_iterations {
        let mut total_error = 0.0;

        for (k, &(row, col)) in positions.iter().enumerate() {
            // 1. Extract object region and compute exit wave
            let exit_wave = extract_and_multiply(&object, &probe, row, col, py, px);

            // 2. Propagate to far field
            let mut psi = exit_wave.clone();
            fft.forward(&mut psi);

            // 3. Apply Fourier magnitude constraint
            let pattern = &diffraction_patterns[k];
            let mut error_k = 0.0;
            for i in 0..py {
                for j in 0..px {
                    let measured_amp = pattern[[i, j]].sqrt();
                    let current_amp = psi[[i, j]].norm();
                    if current_amp > 1e-15 {
                        let phase = psi[[i, j]] / current_amp;
                        error_k += (current_amp - measured_amp).powi(2);
                        psi[[i, j]] = phase * measured_amp;
                    }
                }
            }
            total_error += error_k;

            // 4. Back-propagate
            fft.inverse(&mut psi);

            // 5. Update object and probe (ePIE update rules)
            let probe_max_sq = probe.iter().map(|c| c.norm_sqr()).fold(0.0_f64, f64::max);

            for i in 0..py {
                for j in 0..px {
                    let oi = row + i;
                    let oj = col + j;
                    if oi < obj_ny && oj < obj_nx {
                        let diff = psi[[i, j]] - exit_wave[[i, j]];

                        // Object update
                        if probe_max_sq > 1e-30 {
                            object[[oi, oj]] += config.object_step_size
                                * probe[[i, j]].conj()
                                * diff
                                / probe_max_sq;
                        }
                    }
                }
            }

            // Probe update
            let obj_max_sq: f64 = (0..py)
                .flat_map(|i| (0..px).map(move |j| (i, j)))
                .filter_map(|(i, j)| {
                    let oi = row + i;
                    let oj = col + j;
                    if oi < obj_ny && oj < obj_nx {
                        Some(object[[oi, oj]].norm_sqr())
                    } else {
                        None
                    }
                })
                .fold(0.0_f64, f64::max);

            if obj_max_sq > 1e-30 {
                for i in 0..py {
                    for j in 0..px {
                        let oi = row + i;
                        let oj = col + j;
                        if oi < obj_ny && oj < obj_nx {
                            let diff = psi[[i, j]] - exit_wave[[i, j]];
                            probe[[i, j]] += config.probe_step_size
                                * object[[oi, oj]].conj()
                                * diff
                                / obj_max_sq;
                        }
                    }
                }
            }
        }

        error_history.push(total_error);
    }

    PtychographyResult {
        object,
        probe,
        error_history,
    }
}

/// Extract a region from the object and multiply by the probe.
fn extract_and_multiply(
    object: &Array2<Complex64>,
    probe: &Array2<Complex64>,
    row: usize,
    col: usize,
    py: usize,
    px: usize,
) -> Array2<Complex64> {
    let (obj_ny, obj_nx) = object.dim();
    let mut exit_wave = Array2::zeros((py, px));
    for i in 0..py {
        for j in 0..px {
            let oi = row + i;
            let oj = col + j;
            if oi < obj_ny && oj < obj_nx {
                exit_wave[[i, j]] = object[[oi, oj]] * probe[[i, j]];
            }
        }
    }
    exit_wave
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num::Complex;

    fn make_test_probe(size: usize) -> Array2<Complex64> {
        // Gaussian probe
        let center = size as f64 / 2.0;
        let sigma = size as f64 / 4.0;
        Array2::from_shape_fn((size, size), |(i, j)| {
            let r2 = (i as f64 - center).powi(2) + (j as f64 - center).powi(2);
            Complex64::new((-r2 / (2.0 * sigma * sigma)).exp(), 0.0)
        })
    }

    fn make_test_object(size: usize) -> Array2<Complex64> {
        // Object with a rectangular feature
        Array2::from_shape_fn((size, size), |(i, j)| {
            if i > size / 4 && i < 3 * size / 4 && j > size / 4 && j < 3 * size / 4 {
                Complex64::new(0.5, 0.0) // absorbing feature
            } else {
                Complex64::new(1.0, 0.0) // transparent
            }
        })
    }

    #[test]
    fn test_simulate_diffraction() {
        let probe = make_test_probe(32);
        let object = make_test_object(64);
        let positions = vec![(0, 0), (8, 0), (0, 8), (8, 8)];

        let patterns = simulate_diffraction_patterns(&object, &probe, &positions);
        assert_eq!(patterns.len(), 4);
        for pattern in &patterns {
            assert_eq!(pattern.dim(), (32, 32));
            // All intensities should be non-negative
            assert!(pattern.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn test_epie_convergence() {
        let probe = make_test_probe(16);
        let object = make_test_object(32);
        let positions = vec![
            (0, 0), (4, 0), (8, 0), (12, 0),
            (0, 4), (4, 4), (8, 4), (12, 4),
            (0, 8), (4, 8), (8, 8), (12, 8),
        ];

        // Simulate ground truth patterns
        let patterns = simulate_diffraction_patterns(&object, &probe, &positions);

        // Reconstruct
        let config = PtychographyConfig {
            num_iterations: 20,
            object_step_size: 0.8,
            probe_step_size: 0.8,
        };
        let result = epie_reconstruct(&patterns, &positions, &probe, (32, 32), &config);

        assert_eq!(result.error_history.len(), 20);
        // Error should decrease over iterations
        let first_error = result.error_history[0];
        let last_error = result.error_history[result.error_history.len() - 1];
        assert!(
            last_error <= first_error + 1e-6,
            "Error should decrease: first={:.2e}, last={:.2e}",
            first_error,
            last_error
        );
    }

    #[test]
    fn test_epie_reconstructs_shape() {
        let result_obj_size = (32, 32);
        let probe = make_test_probe(16);
        let object = make_test_object(32);

        let positions: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i * 4, j * 4)))
            .collect();

        let patterns = simulate_diffraction_patterns(&object, &probe, &positions);

        let config = PtychographyConfig {
            num_iterations: 5,
            ..Default::default()
        };
        let result = epie_reconstruct(&patterns, &positions, &probe, result_obj_size, &config);

        assert_eq!(result.object.dim(), result_obj_size);
        assert_eq!(result.probe.dim(), probe.dim());
    }
}
