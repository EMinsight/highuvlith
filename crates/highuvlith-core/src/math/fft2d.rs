use ndarray::Array2;
use num::Complex;
use rustfft::{FftDirection, FftPlanner};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Complex64 = Complex<f64>;

/// Cached 2D FFT engine. Reuses FFT plans for repeated transforms of the same size.
pub struct Fft2D {
    planner: Mutex<FftPlanner<f64>>,
    forward_plans: Mutex<HashMap<usize, Arc<dyn rustfft::Fft<f64>>>>,
    inverse_plans: Mutex<HashMap<usize, Arc<dyn rustfft::Fft<f64>>>>,
}

impl Fft2D {
    pub fn new() -> Self {
        Self {
            planner: Mutex::new(FftPlanner::new()),
            forward_plans: Mutex::new(HashMap::new()),
            inverse_plans: Mutex::new(HashMap::new()),
        }
    }

    fn get_plan(&self, n: usize, direction: FftDirection) -> Arc<dyn rustfft::Fft<f64>> {
        let cache = match direction {
            FftDirection::Forward => &self.forward_plans,
            FftDirection::Inverse => &self.inverse_plans,
        };

        let mut cache_lock = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(plan) = cache_lock.get(&n) {
            return Arc::clone(plan);
        }

        let mut planner = self.planner.lock().unwrap_or_else(|e| e.into_inner());
        let plan = match direction {
            FftDirection::Forward => planner.plan_fft_forward(n),
            FftDirection::Inverse => planner.plan_fft_inverse(n),
        };
        cache_lock.insert(n, Arc::clone(&plan));
        plan
    }

    /// Forward 2D FFT (in-place on rows then columns).
    pub fn forward(&self, data: &mut Array2<Complex64>) {
        let (ny, nx) = data.dim();
        let plan_x = self.get_plan(nx, FftDirection::Forward);
        let plan_y = self.get_plan(ny, FftDirection::Forward);

        // Transform rows
        for mut row in data.rows_mut() {
            let slice = row.as_slice_mut().expect("row not contiguous");
            plan_x.process(slice);
        }

        // Transform columns
        let mut col_buf = vec![Complex64::new(0.0, 0.0); ny];
        for j in 0..nx {
            for i in 0..ny {
                col_buf[i] = data[[i, j]];
            }
            plan_y.process(&mut col_buf);
            for i in 0..ny {
                data[[i, j]] = col_buf[i];
            }
        }
    }

    /// Inverse 2D FFT (in-place, with 1/N normalization).
    pub fn inverse(&self, data: &mut Array2<Complex64>) {
        let (ny, nx) = data.dim();
        let plan_x = self.get_plan(nx, FftDirection::Inverse);
        let plan_y = self.get_plan(ny, FftDirection::Inverse);

        // Transform rows
        for mut row in data.rows_mut() {
            let slice = row.as_slice_mut().expect("row not contiguous");
            plan_x.process(slice);
        }

        // Transform columns
        let mut col_buf = vec![Complex64::new(0.0, 0.0); ny];
        for j in 0..nx {
            for i in 0..ny {
                col_buf[i] = data[[i, j]];
            }
            plan_y.process(&mut col_buf);
            for i in 0..ny {
                data[[i, j]] = col_buf[i];
            }
        }

        // Normalize
        let n = (nx * ny) as f64;
        data.mapv_inplace(|v| v / n);
    }

    /// Forward FFT of real-valued 2D data.
    pub fn forward_real(&self, data: &Array2<f64>) -> Array2<Complex64> {
        let mut complex_data = data.mapv(|v| Complex64::new(v, 0.0));
        self.forward(&mut complex_data);
        complex_data
    }

    /// Apply FFT shift: move zero-frequency component to center.
    pub fn fftshift(data: &Array2<Complex64>) -> Array2<Complex64> {
        let (ny, nx) = data.dim();
        let hx = nx / 2;
        let hy = ny / 2;
        let mut shifted = Array2::zeros((ny, nx));
        for i in 0..ny {
            for j in 0..nx {
                let si = (i + hy) % ny;
                let sj = (j + hx) % nx;
                shifted[[si, sj]] = data[[i, j]];
            }
        }
        shifted
    }

    /// Inverse FFT shift: undo fftshift.
    pub fn ifftshift(data: &Array2<Complex64>) -> Array2<Complex64> {
        let (ny, nx) = data.dim();
        let hx = nx.div_ceil(2);
        let hy = ny.div_ceil(2);
        let mut shifted = Array2::zeros((ny, nx));
        for i in 0..ny {
            for j in 0..nx {
                let si = (i + hy) % ny;
                let sj = (j + hx) % nx;
                shifted[[si, sj]] = data[[i, j]];
            }
        }
        shifted
    }
}

impl Default for Fft2D {
    fn default() -> Self {
        Self::new()
    }
}

// Fft2D is safe to share across threads since all interior mutability
// is behind Mutex.
unsafe impl Send for Fft2D {}
unsafe impl Sync for Fft2D {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_roundtrip() {
        let fft = Fft2D::new();
        let n = 64;
        let mut data = Array2::zeros((n, n));
        // Create a simple pattern
        data[[n / 2, n / 2]] = Complex64::new(1.0, 0.0);

        let original = data.clone();
        fft.forward(&mut data);
        fft.inverse(&mut data);

        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(data[[i, j]].re, original[[i, j]].re, epsilon = 1e-10);
                assert_relative_eq!(data[[i, j]].im, original[[i, j]].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_parseval() {
        let fft = Fft2D::new();
        let n = 64;
        let mut data = Array2::zeros((n, n));
        data[[10, 20]] = Complex64::new(3.0, 1.0);
        data[[30, 40]] = Complex64::new(-2.0, 0.5);

        let energy_spatial: f64 = data.iter().map(|v| v.norm_sqr()).sum();

        let mut freq = data.clone();
        fft.forward(&mut freq);
        let energy_freq: f64 = freq.iter().map(|v| v.norm_sqr()).sum();

        // Parseval: sum|f|^2 = (1/N) * sum|F|^2
        assert_relative_eq!(
            energy_spatial,
            energy_freq / (n * n) as f64,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip() {
        let n = 64;
        let mut data = Array2::zeros((n, n));
        data[[0, 0]] = Complex64::new(1.0, 0.0);
        data[[10, 20]] = Complex64::new(3.0, -1.0);
        data[[50, 30]] = Complex64::new(-2.0, 0.5);

        let shifted = Fft2D::fftshift(&data);
        let restored = Fft2D::ifftshift(&shifted);

        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(restored[[i, j]].re, data[[i, j]].re, epsilon = 1e-15);
                assert_relative_eq!(restored[[i, j]].im, data[[i, j]].im, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_forward_real_matches_forward() {
        let fft = Fft2D::new();
        let n = 64;
        let real_data = Array2::from_shape_fn((n, n), |(i, j)| ((i + j) as f64 * 0.1).sin());

        let result_real = fft.forward_real(&real_data);

        let mut complex_data = real_data.mapv(|v| Complex64::new(v, 0.0));
        fft.forward(&mut complex_data);

        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    result_real[[i, j]].re,
                    complex_data[[i, j]].re,
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    result_real[[i, j]].im,
                    complex_data[[i, j]].im,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_dc_component() {
        let fft = Fft2D::new();
        let n = 64;
        let constant_val = 3.5;
        let real_data = Array2::from_elem((n, n), constant_val);

        let result = fft.forward_real(&real_data);

        // DC component at [0,0] should be N*N * constant_val
        let dc = result[[0, 0]];
        assert_relative_eq!(dc.re, (n * n) as f64 * constant_val, epsilon = 1e-8);
        assert_relative_eq!(dc.im, 0.0, epsilon = 1e-8);

        // All other components should be zero
        for i in 0..n {
            for j in 0..n {
                if i == 0 && j == 0 {
                    continue;
                }
                assert_relative_eq!(result[[i, j]].norm(), 0.0, epsilon = 1e-8);
            }
        }
    }
}
