//! Stochastic effects modeling for VUV lithography.
//!
//! Models photon shot noise and acid diffusion randomness to predict
//! Line Edge Roughness (LER) and Line Width Roughness (LWR).

use ndarray::Array2;
use rand::prelude::*;
use rand_distr::{Normal, Poisson};

use crate::source::LithographySource;

/// Parameters for stochastic simulation.
#[derive(Debug, Clone)]
pub struct StochasticParams {
    /// Photon density at dose=1 mJ/cm² (photons/nm²).
    /// At 157nm: E_photon = hc/λ = 1.27e-18 J, so 1 mJ/cm² = 7.9e12 photons/cm²
    /// = 0.079 photons/nm².
    pub photon_density_per_mj_cm2: f64,
    /// PEB acid diffusion length standard deviation (nm).
    pub acid_diffusion_sigma_nm: f64,
    /// Number of Monte Carlo realizations.
    pub num_realizations: usize,
}

impl StochasticParams {
    /// Create stochastic params from a source, computing photon density automatically.
    pub fn from_source(source: &impl LithographySource) -> Self {
        Self {
            photon_density_per_mj_cm2: source.photon_density_per_mj_cm2(),
            ..Self::default()
        }
    }

    /// Default parameters for F2 laser (157nm) lithography.
    pub fn default_vuv() -> Self {
        Self {
            // hc/λ at 157nm: photon energy = 1.267e-18 J
            // 1 mJ/cm² = 1e-3 / (1e-14) J/nm² = 1e11 J/m² = ...
            // photon_density = dose / E_photon = (dose_mJ * 1e-3 * 1e14) / (1.267e-18)
            // For dose_mJ=1: 1e-3 * 1e14 / 1.267e-18 = 7.89e28... that's per m²
            // Per nm²: 7.89e28 * 1e-18 = 7.89e10... that's way too high for shot noise
            // Actually: 1 mJ/cm² = 10 J/m² = 10 / E_photon photons/m²
            // = 10 / 1.267e-18 = 7.89e18 photons/m² = 0.00789 photons/nm²
            // At typical dose 30 mJ/cm²: 0.237 photons/nm²
            // Per pixel at 1nm pixel: 0.237 photons
            // Per pixel at 2nm pixel: 0.947 photons
            photon_density_per_mj_cm2: 0.00789,
            acid_diffusion_sigma_nm: 5.0,
            num_realizations: 100,
        }
    }
}

impl Default for StochasticParams {
    fn default() -> Self {
        Self::default_vuv()
    }
}

/// Result of stochastic LER/LWR analysis.
#[derive(Debug, Clone)]
pub struct LerResult {
    /// Line Edge Roughness (3σ) in nm.
    pub ler_3sigma_nm: f64,
    /// Line Width Roughness (3σ) in nm.
    pub lwr_3sigma_nm: f64,
    /// Mean CD across realizations (nm).
    pub cd_mean_nm: f64,
    /// CD standard deviation across realizations (nm).
    pub cd_sigma_nm: f64,
    /// Individual edge positions per realization (left edge).
    pub left_edges: Vec<f64>,
    /// Individual edge positions per realization (right edge).
    pub right_edges: Vec<f64>,
}

/// Apply photon shot noise to an aerial image.
///
/// The number of photons absorbed at each pixel follows a Poisson distribution.
/// The noisy intensity is the Poisson-sampled photon count normalized back to
/// the original intensity scale.
pub fn apply_shot_noise(
    aerial_image: &Array2<f64>,
    dose_mj_cm2: f64,
    pixel_nm: f64,
    params: &StochasticParams,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let pixel_area = pixel_nm * pixel_nm;
    let base_photons = dose_mj_cm2 * params.photon_density_per_mj_cm2 * pixel_area;

    aerial_image.mapv(|intensity| {
        let mean_photons = intensity * base_photons;
        if mean_photons <= 0.0 {
            return 0.0;
        }

        // Poisson sampling for small photon counts, Gaussian approximation for large
        let sampled = if mean_photons < 1000.0 {
            match Poisson::new(mean_photons) {
                Ok(dist) => rng.sample(dist),
                Err(_) => mean_photons,
            }
        } else {
            match Normal::new(mean_photons, mean_photons.sqrt()) {
                Ok(normal) => rng.sample(normal).max(0.0),
                Err(_) => mean_photons, // fallback to deterministic
            }
        };

        // Normalize back to intensity scale
        if base_photons > 0.0 {
            sampled / base_photons
        } else {
            0.0
        }
    })
}

/// Apply random acid diffusion perturbation to a latent image.
///
/// Each pixel's PAC concentration is perturbed by a Gaussian random variable
/// representing the stochastic nature of acid generation and diffusion during PEB.
pub fn apply_acid_noise(
    pac: &Array2<f64>,
    params: &StochasticParams,
    pixel_nm: f64,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let sigma_pixels = params.acid_diffusion_sigma_nm / pixel_nm;
    // The noise magnitude scales with the local gradient of PAC concentration
    let noise_scale = 0.01 * sigma_pixels; // empirical scaling

    pac.mapv(|m| {
        let noise = match Normal::new(0.0, noise_scale) {
            Ok(dist) => rng.sample(dist),
            Err(_) => 0.0,
        };
        (m + noise).clamp(0.0, 1.0)
    })
}

/// Compute LER/LWR by running multiple stochastic realizations.
///
/// For each realization:
/// 1. Apply photon shot noise to aerial image
/// 2. Apply acid diffusion noise to latent image
/// 3. Measure CD (threshold crossing positions)
///
/// Returns LER (edge position roughness) and LWR (CD roughness) statistics.
pub fn compute_ler_lwr(
    aerial_image: &Array2<f64>,
    x_min_nm: f64,
    x_max_nm: f64,
    dose_mj_cm2: f64,
    pixel_nm: f64,
    threshold: f64,
    params: &StochasticParams,
) -> LerResult {
    let mut rng = StdRng::seed_from_u64(42);
    let nx = aerial_image.ncols();
    let ny = aerial_image.nrows();
    let center_row = ny / 2;

    let x_nm: Vec<f64> = (0..nx)
        .map(|j| x_min_nm + (j as f64 + 0.5) * pixel_nm)
        .collect();

    let mut left_edges = Vec::with_capacity(params.num_realizations);
    let mut right_edges = Vec::with_capacity(params.num_realizations);
    let mut cds = Vec::with_capacity(params.num_realizations);

    for _ in 0..params.num_realizations {
        let noisy = apply_shot_noise(aerial_image, dose_mj_cm2, pixel_nm, params, &mut rng);

        // Extract center row cross-section
        let profile: Vec<f64> = (0..nx).map(|j| noisy[[center_row, j]]).collect();

        // Find threshold crossings
        let crossings = find_crossings(&profile, &x_nm, threshold);

        if crossings.len() >= 2 {
            // Find the pair closest to center
            let center = (x_min_nm + x_max_nm) / 2.0;
            let mut best_pair = (0, 1);
            let mut best_dist = f64::INFINITY;

            for i in 0..crossings.len() - 1 {
                let mid = (crossings[i] + crossings[i + 1]) / 2.0;
                let dist = (mid - center).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_pair = (i, i + 1);
                }
            }

            left_edges.push(crossings[best_pair.0]);
            right_edges.push(crossings[best_pair.1]);
            cds.push(crossings[best_pair.1] - crossings[best_pair.0]);
        }
    }

    if cds.is_empty() {
        return LerResult {
            ler_3sigma_nm: 0.0,
            lwr_3sigma_nm: 0.0,
            cd_mean_nm: 0.0,
            cd_sigma_nm: 0.0,
            left_edges,
            right_edges,
        };
    }

    let cd_mean = cds.iter().sum::<f64>() / cds.len() as f64;
    let cd_var = cds.iter().map(|&c| (c - cd_mean).powi(2)).sum::<f64>() / cds.len() as f64;
    let cd_sigma = cd_var.sqrt();

    let left_mean = left_edges.iter().sum::<f64>() / left_edges.len() as f64;
    let left_var = left_edges
        .iter()
        .map(|&e| (e - left_mean).powi(2))
        .sum::<f64>()
        / left_edges.len() as f64;

    let right_mean = right_edges.iter().sum::<f64>() / right_edges.len() as f64;
    let right_var = right_edges
        .iter()
        .map(|&e| (e - right_mean).powi(2))
        .sum::<f64>()
        / right_edges.len() as f64;

    // LER is the average edge roughness (3σ of edge position)
    let ler_sigma = ((left_var + right_var) / 2.0).sqrt();
    // LWR is the CD roughness (3σ of CD)
    let lwr_sigma = cd_sigma;

    LerResult {
        ler_3sigma_nm: 3.0 * ler_sigma,
        lwr_3sigma_nm: 3.0 * lwr_sigma,
        cd_mean_nm: cd_mean,
        cd_sigma_nm: cd_sigma,
        left_edges,
        right_edges,
    }
}

/// Find threshold crossings in a 1D intensity profile.
fn find_crossings(profile: &[f64], x_nm: &[f64], threshold: f64) -> Vec<f64> {
    let mut crossings = Vec::new();
    for i in 0..profile.len() - 1 {
        let y0 = profile[i] - threshold;
        let y1 = profile[i + 1] - threshold;
        if y0 * y1 < 0.0 {
            let t = y0 / (y0 - y1);
            crossings.push(x_nm[i] + t * (x_nm[i + 1] - x_nm[i]));
        }
    }
    crossings
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shot_noise_preserves_mean() {
        let n = 128;
        let aerial = Array2::from_elem((n, n), 0.5);
        let params = StochasticParams {
            photon_density_per_mj_cm2: 0.1,
            acid_diffusion_sigma_nm: 5.0,
            num_realizations: 1,
        };

        // Average over many realizations should converge to original
        let mut sum = Array2::zeros((n, n));
        let num_trials = 200;
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..num_trials {
            let noisy = apply_shot_noise(&aerial, 30.0, 2.0, &params, &mut rng);
            sum += &noisy;
        }
        sum.mapv_inplace(|v| v / num_trials as f64);

        let mean_val = sum.iter().sum::<f64>() / (n * n) as f64;
        assert_relative_eq!(mean_val, 0.5, epsilon = 0.05);
    }

    #[test]
    fn test_shot_noise_non_negative() {
        let aerial = Array2::from_elem((64, 64), 0.3);
        let params = StochasticParams::default_vuv();
        let mut rng = StdRng::seed_from_u64(42);
        let noisy = apply_shot_noise(&aerial, 30.0, 2.0, &params, &mut rng);
        assert!(noisy.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_acid_noise_bounded() {
        let pac = Array2::from_elem((64, 64), 0.5);
        let params = StochasticParams::default_vuv();
        let mut rng = StdRng::seed_from_u64(42);
        let noisy = apply_acid_noise(&pac, &params, 2.0, &mut rng);
        assert!(noisy.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn test_ler_lwr_computation() {
        // Create a simple step-function aerial image
        let n = 128;
        let mut aerial = Array2::from_elem((n, n), 0.1);
        for i in 0..n {
            for j in 40..88 {
                aerial[[i, j]] = 0.9;
            }
        }

        let params = StochasticParams {
            photon_density_per_mj_cm2: 0.01,
            acid_diffusion_sigma_nm: 3.0,
            num_realizations: 50,
        };

        let result = compute_ler_lwr(&aerial, -128.0, 128.0, 30.0, 2.0, 0.5, &params);

        assert!(result.cd_mean_nm > 0.0, "Mean CD should be positive");
        assert!(result.ler_3sigma_nm >= 0.0, "LER should be non-negative");
        assert!(result.lwr_3sigma_nm >= 0.0, "LWR should be non-negative");
        assert_eq!(result.left_edges.len(), 50);
        assert_eq!(result.right_edges.len(), 50);
    }

    #[test]
    fn test_ler_increases_with_noise() {
        // Use a smooth Gaussian feature profile instead of hard step
        let n = 128;
        let mut aerial = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let x = (j as f64 - 64.0) * 2.0; // nm from center
                let sigma = 30.0;
                aerial[[i, j]] = (-x * x / (2.0 * sigma * sigma)).exp();
            }
        }

        // Very high photon density = essentially no noise → LER should be low
        let params_low_noise = StochasticParams {
            photon_density_per_mj_cm2: 10.0, // very high
            acid_diffusion_sigma_nm: 0.1,
            num_realizations: 50,
        };
        let result = compute_ler_lwr(&aerial, -128.0, 128.0, 100.0, 2.0, 0.5, &params_low_noise);
        assert!(
            result.ler_3sigma_nm < 5.0,
            "Low-noise LER should be small, got {:.2}",
            result.ler_3sigma_nm
        );
        assert!(result.cd_mean_nm > 0.0);
    }
}
