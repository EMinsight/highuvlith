//! Quantum lithography simulation (theoretical).
//!
//! Models the use of entangled N-photon states (NOON states) to achieve
//! sub-Rayleigh resolution at λ/(2N) instead of the classical λ/2 limit.
//! This is a theoretical/research module — practical implementation faces
//! extreme flux limitations.

use ndarray::Array2;

/// Quantum lithography parameters.
#[derive(Debug, Clone)]
pub struct QuantumLithographyParams {
    /// Number of entangled photons (N). N=2 is biphoton.
    pub num_entangled_photons: usize,
    /// Source wavelength (nm).
    pub wavelength_nm: f64,
    /// Numerical aperture.
    pub na: f64,
    /// Entanglement fidelity (0-1). Accounts for decoherence.
    pub fidelity: f64,
}

impl Default for QuantumLithographyParams {
    fn default() -> Self {
        Self {
            num_entangled_photons: 2,
            wavelength_nm: 157.63,
            na: 0.75,
            fidelity: 1.0,
        }
    }
}

impl QuantumLithographyParams {
    /// Effective wavelength: λ_eff = λ / N.
    pub fn effective_wavelength_nm(&self) -> f64 {
        self.wavelength_nm / self.num_entangled_photons as f64
    }

    /// Quantum-enhanced Rayleigh resolution: 0.61 × λ / (N × NA).
    pub fn quantum_resolution_nm(&self) -> f64 {
        0.61 * self.wavelength_nm / (self.num_entangled_photons as f64 * self.na)
    }

    /// Classical Rayleigh resolution for comparison: 0.61 × λ / NA.
    pub fn classical_resolution_nm(&self) -> f64 {
        0.61 * self.wavelength_nm / self.na
    }

    /// Resolution improvement factor.
    pub fn resolution_factor(&self) -> f64 {
        self.num_entangled_photons as f64
    }

    /// Estimate relative flux (photons/s) compared to classical.
    /// N-photon entangled states have dramatically lower generation rates.
    pub fn relative_flux(&self) -> f64 {
        // Approximate: flux scales as η^N where η ~ 0.01 (pair generation efficiency)
        let eta: f64 = 0.01;
        eta.powi(self.num_entangled_photons as i32 - 1)
    }

    /// Estimate exposure time ratio compared to classical.
    /// Combines lower flux with N-photon absorption requirement.
    pub fn exposure_time_ratio(&self) -> f64 {
        1.0 / self.relative_flux()
    }
}

/// Compute N-photon absorption pattern (quantum lithography aerial image).
///
/// For N-photon entangled state, the exposure pattern is:
///   E_N(x) = |E(x)|^(2N) (coherent N-photon absorption)
///
/// This produces sharper features than classical |E(x)|² imaging.
pub fn compute_quantum_aerial_image(
    classical_aerial: &Array2<f64>,
    params: &QuantumLithographyParams,
) -> Array2<f64> {
    let n_photons = params.num_entangled_photons;
    let fidelity = params.fidelity;

    classical_aerial.mapv(|intensity| {
        // N-photon absorption: I^N for entangled state
        let quantum_signal = intensity.powi(n_photons as i32);
        // Mix with classical signal based on fidelity
        // fidelity=1: pure quantum; fidelity=0: classical
        fidelity * quantum_signal + (1.0 - fidelity) * intensity
    })
}

/// Compare classical vs quantum imaging for a given pattern.
#[derive(Debug)]
pub struct QuantumComparison {
    /// Classical aerial image.
    pub classical: Array2<f64>,
    /// Quantum aerial image.
    pub quantum: Array2<f64>,
    /// Classical image contrast.
    pub classical_contrast: f64,
    /// Quantum image contrast.
    pub quantum_contrast: f64,
    /// Classical Rayleigh resolution (nm).
    pub classical_resolution_nm: f64,
    /// Quantum resolution (nm).
    pub quantum_resolution_nm: f64,
    /// Exposure time penalty factor.
    pub exposure_time_ratio: f64,
}

/// Generate a comparison between classical and quantum lithography.
pub fn compare_classical_quantum(
    classical_aerial: &Array2<f64>,
    params: &QuantumLithographyParams,
) -> QuantumComparison {
    let quantum = compute_quantum_aerial_image(classical_aerial, params);

    let classical_contrast = image_contrast(classical_aerial);
    let quantum_contrast = image_contrast(&quantum);

    QuantumComparison {
        classical: classical_aerial.clone(),
        quantum,
        classical_contrast,
        quantum_contrast,
        classical_resolution_nm: params.classical_resolution_nm(),
        quantum_resolution_nm: params.quantum_resolution_nm(),
        exposure_time_ratio: params.exposure_time_ratio(),
    }
}

fn image_contrast(img: &Array2<f64>) -> f64 {
    let max = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = img.iter().cloned().fold(f64::INFINITY, f64::min);
    if max + min > 0.0 { (max - min) / (max + min) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_effective_wavelength() {
        let params = QuantumLithographyParams::default();
        assert_relative_eq!(params.effective_wavelength_nm(), 157.63 / 2.0);
    }

    #[test]
    fn test_quantum_resolution_better() {
        let params = QuantumLithographyParams::default();
        assert!(params.quantum_resolution_nm() < params.classical_resolution_nm());
        assert_relative_eq!(
            params.quantum_resolution_nm(),
            params.classical_resolution_nm() / 2.0,
            epsilon = 0.01
        );
    }

    #[test]
    fn test_exposure_time_very_long() {
        let params = QuantumLithographyParams {
            num_entangled_photons: 2,
            ..Default::default()
        };
        assert!(params.exposure_time_ratio() > 10.0,
            "Biphoton exposure should be much longer: {:.0}×", params.exposure_time_ratio());
    }

    #[test]
    fn test_quantum_sharpening() {
        // Quantum N-photon absorption should increase contrast
        let n = 64;
        let aerial = Array2::from_shape_fn((n, n), |(_, j)| {
            let x = (j as f64 - 32.0) / 10.0;
            0.5 + 0.3 * (-x * x / 2.0).exp()
        });

        let params = QuantumLithographyParams {
            num_entangled_photons: 2,
            fidelity: 1.0,
            ..Default::default()
        };

        let quantum = compute_quantum_aerial_image(&aerial, &params);
        let c_class = image_contrast(&aerial);
        let c_quant = image_contrast(&quantum);

        assert!(c_quant > c_class,
            "Quantum contrast ({:.4}) should exceed classical ({:.4})", c_quant, c_class);
    }

    #[test]
    fn test_fidelity_interpolation() {
        let aerial = Array2::from_elem((16, 16), 0.5);
        let params_full = QuantumLithographyParams { fidelity: 1.0, ..Default::default() };
        let params_none = QuantumLithographyParams { fidelity: 0.0, ..Default::default() };

        let q_full = compute_quantum_aerial_image(&aerial, &params_full);
        let q_none = compute_quantum_aerial_image(&aerial, &params_none);

        // fidelity=0 should give back classical
        assert_relative_eq!(q_none[[0, 0]], 0.5, epsilon = 1e-10);
        // fidelity=1 should give I^N = 0.5^2 = 0.25
        assert_relative_eq!(q_full[[0, 0]], 0.25, epsilon = 1e-10);
    }
}
