use ndarray::Array2;
use num::Complex;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{LithographyError, Result};
use crate::mask::Mask;
use crate::math::fft2d::Fft2D;
use crate::mnsl::{MnslConfig, MnslEngine, MnslResult};
use crate::optics::OpticalSystem;
use crate::source::LithographySource;
use crate::types::{Complex64, Grid2D, GridConfig};

/// Aerial image computation engine using Hopkins partially-coherent imaging
/// with TCC (Transmission Cross-Coefficient) eigendecomposition (SOCS).
pub struct AerialImageEngine {
    /// SOCS kernels: (eigenvalue, 2D kernel in frequency domain).
    tcc_kernels: Vec<(f64, Array2<Complex64>)>,
    /// Grid configuration.
    grid: GridConfig,
    /// Wavelength in nm.
    wavelength_nm: f64,
    /// Numerical aperture.
    na: f64,
    /// Flare fraction.
    flare_fraction: f64,
    /// FFT engine.
    fft: Fft2D,
}

impl AerialImageEngine {
    /// Create a new aerial image engine by computing and decomposing the TCC.
    ///
    /// This is the expensive initialization step. Once created, the engine
    /// can efficiently compute aerial images for different masks and defocus values.
    pub fn new(
        source: &(impl LithographySource + ?Sized),
        optics: &(impl OpticalSystem + ?Sized),
        grid: GridConfig,
        max_kernels: usize,
    ) -> Result<Self> {
        let fft = Fft2D::new();
        let wavelength_nm = source.wavelength_nm();
        let na = optics.na();

        let tcc_kernels = compute_tcc_socs(source, optics, &grid, max_kernels, wavelength_nm)?;

        if tcc_kernels.is_empty() {
            return Err(LithographyError::NoDiffractionOrders);
        }

        Ok(Self {
            tcc_kernels,
            grid,
            wavelength_nm,
            na,
            flare_fraction: optics.flare_fraction(),
            fft,
        })
    }

    /// Compute the aerial image for a given mask and defocus.
    ///
    /// Uses the pre-computed SOCS kernels for efficient evaluation.
    /// The aerial image is the sum of weighted coherent images:
    ///   I(x,y) = sum_k eigenvalue_k * |IFFT(kernel_k * mask_spectrum)|^2
    pub fn compute(&self, mask: &Mask, defocus_nm: f64) -> Grid2D<f64> {
        let n = self.grid.size;
        let mask_spectrum = mask.spectrum(&self.grid, &self.fft);

        // Apply defocus to kernels and compute SOCS sum
        let defocus_phase_grid = self.compute_defocus_phase(defocus_nm);

        // Parallel computation over SOCS kernels
        #[cfg(feature = "parallel")]
        let partial_images: Vec<Array2<f64>> = self
            .tcc_kernels
            .par_iter()
            .map(|(eigenvalue, kernel)| {
                let fft = Fft2D::new(); // thread-local FFT
                let mut product = Array2::zeros((n, n));

                // Multiply mask spectrum by kernel (with defocus)
                for i in 0..n {
                    for j in 0..n {
                        let defocus_mod = defocus_phase_grid[[i, j]];
                        product[[i, j]] = mask_spectrum[[i, j]] * kernel[[i, j]] * defocus_mod;
                    }
                }

                // IFFT to get coherent field
                fft.inverse(&mut product);

                // Square magnitude, weighted by eigenvalue
                let mut intensity = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        intensity[[i, j]] = eigenvalue * product[[i, j]].norm_sqr();
                    }
                }
                intensity
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let partial_images: Vec<Array2<f64>> = self
            .tcc_kernels
            .iter()
            .map(|(eigenvalue, kernel)| {
                let fft = Fft2D::new(); // thread-local FFT
                let mut product = Array2::zeros((n, n));

                // Multiply mask spectrum by kernel (with defocus)
                for i in 0..n {
                    for j in 0..n {
                        let defocus_mod = defocus_phase_grid[[i, j]];
                        product[[i, j]] = mask_spectrum[[i, j]] * kernel[[i, j]] * defocus_mod;
                    }
                }

                // IFFT to get coherent field
                fft.inverse(&mut product);

                // Square magnitude, weighted by eigenvalue
                let mut intensity = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        intensity[[i, j]] = eigenvalue * product[[i, j]].norm_sqr();
                    }
                }
                intensity
            })
            .collect();

        // Sum all partial images
        let mut total = Array2::zeros((n, n));
        for partial in &partial_images {
            total += partial;
        }

        // Add flare (uniform background)
        if self.flare_fraction > 0.0 {
            let mean_intensity: f64 = total.iter().sum::<f64>() / (n * n) as f64;
            let flare = self.flare_fraction * mean_intensity;
            total.mapv_inplace(|v| v * (1.0 - self.flare_fraction) + flare);
        }

        let field = self.grid.field_size_nm();
        let half = field / 2.0;
        Grid2D {
            data: total,
            x_min_nm: -half,
            x_max_nm: half,
            y_min_nm: -half,
            y_max_nm: half,
        }
    }

    /// Compute polychromatic aerial image by summing over spectral samples.
    /// Accounts for CaF2 chromatic aberration at VUV wavelengths.
    pub fn compute_polychromatic(
        &self,
        mask: &Mask,
        defocus_nm: f64,
        source: &(impl LithographySource + ?Sized),
        optics: &(impl OpticalSystem + ?Sized),
    ) -> Grid2D<f64> {
        let spectral_weights = source.spectral_weights();

        if spectral_weights.len() <= 1 {
            return self.compute(mask, defocus_nm);
        }

        let n = self.grid.size;
        let field = self.grid.field_size_nm();
        let half = field / 2.0;
        let mut total = Array2::zeros((n, n));

        for (wl, weight) in &spectral_weights {
            let delta_pm = (wl - source.wavelength_nm()) * 1000.0; // nm to pm
            let chromatic_defocus = optics.chromatic_defocus(delta_pm);
            let effective_defocus = defocus_nm + chromatic_defocus;

            let image = self.compute(mask, effective_defocus);
            total.scaled_add(*weight, &image.data);
        }

        Grid2D {
            data: total,
            x_min_nm: -half,
            x_max_nm: half,
            y_min_nm: -half,
            y_max_nm: half,
        }
    }

    /// Number of SOCS kernels used.
    pub fn num_kernels(&self) -> usize {
        self.tcc_kernels.len()
    }

    /// Grid configuration.
    pub fn grid(&self) -> &GridConfig {
        &self.grid
    }

    /// Compute MNSL (Moiré Nanosphere Lithographic Reflection) emission pattern.
    ///
    /// Integrates MNSL simulation with the aerial image engine's existing
    /// capabilities, leveraging the engine's grid configuration and optical setup.
    pub fn compute_mnsl(&self, config: &MnslConfig) -> MnslResult {
        // Create MNSL engine with our grid configuration
        let engine = MnslEngine::new(config.clone(), self.grid.clone());

        // Compute the MNSL emission pattern
        engine.compute_emission()
    }

    fn compute_defocus_phase(&self, defocus_nm: f64) -> Array2<Complex64> {
        let n = self.grid.size;
        let freq_step = self.grid.freq_step();
        let cutoff = self.na / self.wavelength_nm;
        let mut phase_grid = Array2::from_elem((n, n), Complex64::new(1.0, 0.0));

        if defocus_nm.abs() < 1e-12 {
            return phase_grid;
        }

        for i in 0..n {
            for j in 0..n {
                let fy = if i < n / 2 {
                    i as f64 * freq_step
                } else {
                    (i as f64 - n as f64) * freq_step
                };
                let fx = if j < n / 2 {
                    j as f64 * freq_step
                } else {
                    (j as f64 - n as f64) * freq_step
                };

                let rho = (fx * fx + fy * fy).sqrt() / cutoff;
                if rho <= 1.0 {
                    let phase = std::f64::consts::PI * defocus_nm * rho * rho * self.na * self.na
                        / self.wavelength_nm;
                    phase_grid[[i, j]] = Complex::from_polar(1.0, phase);
                }
            }
        }

        phase_grid
    }
}

/// Compute TCC and decompose into SOCS kernels.
///
/// The TCC is a 4D function TCC(f1, f2) describing the cross-correlation
/// of the optical transfer at two frequency points, integrated over the source.
/// We compute it as a 2D matrix indexed by frequency pairs, then eigendecompose
/// to get the SOCS (Sum Of Coherent Systems) kernels.
fn compute_tcc_socs(
    source: &(impl LithographySource + ?Sized),
    optics: &(impl OpticalSystem + ?Sized),
    grid: &GridConfig,
    max_kernels: usize,
    wavelength_nm: f64,
) -> Result<Vec<(f64, Array2<Complex64>)>> {
    let n = grid.size;
    let freq_step = grid.freq_step();
    let cutoff = optics.na() / wavelength_nm;

    // Collect in-pupil frequency indices
    let mut freq_indices: Vec<(usize, usize, f64, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let fy = if i < n / 2 {
                i as f64 * freq_step
            } else {
                (i as f64 - n as f64) * freq_step
            };
            let fx = if j < n / 2 {
                j as f64 * freq_step
            } else {
                (j as f64 - n as f64) * freq_step
            };
            let rho = (fx * fx + fy * fy).sqrt() / cutoff;
            if rho <= 1.0 {
                freq_indices.push((i, j, fx, fy));
            }
        }
    }

    let n_freq = freq_indices.len();
    if n_freq == 0 {
        return Err(LithographyError::NoDiffractionOrders);
    }

    // Build TCC matrix: TCC(m, n) = integral_over_source J(fs) * H(fs+fm) * H*(fs+fn) d_fs
    // For discrete source: sum over source sampling points
    let source_samples = generate_source_samples(source, cutoff);
    if source_samples.is_empty() {
        return Err(LithographyError::NoDiffractionOrders);
    }

    // Build TCC matrix (Hermitian)
    let mut tcc = nalgebra::DMatrix::<num::Complex<f64>>::zeros(n_freq, n_freq);

    for &(sx, sy, sw) in &source_samples {
        // For each source point, compute pupil values at shifted frequencies
        let mut pupil_vals: Vec<Complex64> = Vec::with_capacity(n_freq);
        for &(_i, _j, fx, fy) in &freq_indices {
            let shifted_fx = (fx + sx) / cutoff;
            let shifted_fy = (fy + sy) / cutoff;
            let h = optics.pupil_function(shifted_fx, shifted_fy, 0.0, wavelength_nm);
            pupil_vals.push(h);
        }

        // TCC(m, n) += weight * H(fs+fm) * H*(fs+fn)
        for m in 0..n_freq {
            for nn in m..n_freq {
                let val = nalgebra::Complex::new(
                    sw * (pupil_vals[m] * pupil_vals[nn].conj()).re,
                    sw * (pupil_vals[m] * pupil_vals[nn].conj()).im,
                );
                tcc[(m, nn)] += val;
                if nn > m {
                    tcc[(nn, m)] += val.conj();
                }
            }
        }
    }

    // Eigendecompose (TCC is Hermitian positive semi-definite)
    // Use power iteration / partial eigendecomposition for efficiency
    let eigenvalues_and_vectors = eigendecompose_hermitian(&tcc, max_kernels);

    // Convert eigenvectors back to 2D frequency-domain kernels
    let mut kernels: Vec<(f64, Array2<Complex64>)> = Vec::new();
    let total_energy: f64 = eigenvalues_and_vectors.iter().map(|(ev, _)| ev.abs()).sum();

    for (eigenvalue, eigenvector) in eigenvalues_and_vectors {
        if eigenvalue.abs() < 1e-12 * total_energy {
            break;
        }

        let mut kernel = Array2::from_elem((n, n), Complex64::new(0.0, 0.0));
        for (k, &(i, j, _fx, _fy)) in freq_indices.iter().enumerate() {
            kernel[[i, j]] = Complex64::new(eigenvector[k].re, eigenvector[k].im);
        }

        kernels.push((eigenvalue, kernel));
    }

    Ok(kernels)
}

/// Generate source sampling points: (fx, fy, weight).
/// Uses a grid of points within the source shape.
fn generate_source_samples(
    source: &(impl LithographySource + ?Sized),
    cutoff: f64,
) -> Vec<(f64, f64, f64)> {
    let n_samples = 31; // per dimension
    let step = 2.0 * cutoff / n_samples as f64;
    let mut samples = Vec::new();
    let mut total_weight = 0.0;

    for iy in 0..n_samples {
        for ix in 0..n_samples {
            let fx = -cutoff + (ix as f64 + 0.5) * step;
            let fy = -cutoff + (iy as f64 + 0.5) * step;

            // Normalize to sigma units
            let fx_norm = fx / cutoff;
            let fy_norm = fy / cutoff;

            let w = source.intensity_at(fx_norm, fy_norm);
            if w > 0.0 {
                samples.push((fx, fy, w));
                total_weight += w;
            }
        }
    }

    // Normalize weights
    if total_weight > 0.0 {
        for s in &mut samples {
            s.2 /= total_weight;
        }
    }

    samples
}

/// Simple eigendecomposition of a Hermitian matrix.
/// Returns (eigenvalue, eigenvector) pairs sorted by decreasing |eigenvalue|.
fn eigendecompose_hermitian(
    matrix: &nalgebra::DMatrix<nalgebra::Complex<f64>>,
    max_eigenvectors: usize,
) -> Vec<(f64, Vec<nalgebra::Complex<f64>>)> {
    let n = matrix.nrows();
    if n == 0 {
        return Vec::new();
    }

    // Convert to real symmetric form for eigendecomposition.
    // For a Hermitian matrix H, we can use the real/imaginary parts.
    // However, for simplicity and correctness, use power iteration.
    let k = max_eigenvectors.min(n);
    let mut results: Vec<(f64, Vec<nalgebra::Complex<f64>>)> = Vec::with_capacity(k);

    // Work with a deflated copy
    let mut deflated = matrix.clone();

    for _ in 0..k {
        let (eigenvalue, eigenvector) = power_iteration(&deflated, 200, 1e-10);

        if eigenvalue.abs() < 1e-15 {
            break;
        }

        // Deflate: A = A - eigenvalue * v * v^H
        for i in 0..n {
            for j in 0..n {
                deflated[(i, j)] -= nalgebra::Complex::new(eigenvalue, 0.0)
                    * eigenvector[i]
                    * eigenvector[j].conj();
            }
        }

        results.push((eigenvalue, eigenvector));
    }

    // Sort by decreasing |eigenvalue|
    results.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    results
}

/// Power iteration for finding the dominant eigenvalue/eigenvector of a Hermitian matrix.
fn power_iteration(
    matrix: &nalgebra::DMatrix<nalgebra::Complex<f64>>,
    max_iter: usize,
    tol: f64,
) -> (f64, Vec<nalgebra::Complex<f64>>) {
    let n = matrix.nrows();

    // Random-ish initial vector
    let mut v = nalgebra::DVector::<nalgebra::Complex<f64>>::from_fn(n, |i, _| {
        nalgebra::Complex::new(((i * 7 + 3) % 13) as f64 / 13.0, 0.0)
    });

    // Normalize
    let norm = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 0.0 {
        v /= nalgebra::Complex::new(norm, 0.0);
    }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        let w = matrix * &v;
        let new_eigenvalue = v
            .iter()
            .zip(w.iter())
            .map(|(vi, wi)| (vi.conj() * wi).re)
            .sum::<f64>();

        let w_norm = w.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if w_norm < 1e-30 {
            break;
        }
        v = &w / nalgebra::Complex::new(w_norm, 0.0);

        if (new_eigenvalue - eigenvalue).abs() < tol * new_eigenvalue.abs().max(1e-15) {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }

    (eigenvalue, v.iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::ProjectionOptics;
    use crate::source::VuvSource;
    use approx::assert_relative_eq;

    fn make_test_engine(sigma: f64, na: f64) -> AerialImageEngine {
        let source = VuvSource {
            illumination: crate::source::IlluminationShape::Conventional { sigma },
            ..VuvSource::f2_laser(sigma).unwrap()
        };
        let optics = ProjectionOptics::new(na).unwrap();
        let grid = GridConfig {
            size: 128,
            pixel_nm: 2.0,
        };
        AerialImageEngine::new(&source, &optics, grid, 20).unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let engine = make_test_engine(0.5, 0.75);
        assert!(engine.num_kernels() > 0);
        assert!(engine.num_kernels() <= 20);
    }

    #[test]
    fn test_aerial_image_non_negative() {
        let engine = make_test_engine(0.5, 0.75);
        let mask = Mask::line_space(65.0, 180.0).unwrap();
        let image = engine.compute(&mask, 0.0);
        for &v in image.data.iter() {
            assert!(v >= -1e-10, "Intensity must be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_symmetric_mask_symmetric_image() {
        let engine = make_test_engine(0.5, 0.75);
        // Centered rectangular feature should give symmetric image
        let mask = Mask {
            mask_type: crate::mask::MaskType::Binary,
            features: vec![crate::mask::MaskFeature::Rect {
                x: 0.0,
                y: 0.0,
                w: 80.0,
                h: 256.0,
            }],
            dark_field: true,
        };
        let image = engine.compute(&mask, 0.0);
        let n = image.data.ncols();
        let center_row = image.data.nrows() / 2;

        // Check left-right symmetry
        for j in 0..n / 2 {
            let left = image.data[[center_row, j]];
            let right = image.data[[center_row, n - 1 - j]];
            assert_relative_eq!(left, right, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_defocus_reduces_contrast() {
        let engine = make_test_engine(0.5, 0.75);
        let mask = Mask::line_space(65.0, 180.0).unwrap();

        let image_focus = engine.compute(&mask, 0.0);
        let image_defocus = engine.compute(&mask, 200.0);

        let contrast = |img: &Array2<f64>| -> f64 {
            let max = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = img.iter().cloned().fold(f64::INFINITY, f64::min);
            (max - min) / (max + min)
        };

        let c0 = contrast(&image_focus.data);
        let c1 = contrast(&image_defocus.data);
        assert!(
            c0 > c1,
            "In-focus contrast {} should exceed defocused contrast {}",
            c0,
            c1
        );
    }
}
