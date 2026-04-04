//! Schwarzschild reflective objective for EUV and soft X-ray lithography.
//!
//! A Schwarzschild objective uses two concentric spherical mirrors
//! (convex primary + concave secondary) to form an image. It provides
//! NA up to ~0.3 for EUV/soft X-ray wavelengths where refractive optics
//! are impossible.

use num::Complex;
use serde::{Deserialize, Serialize};

use crate::types::Complex64;

/// Schwarzschild two-mirror reflective objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchwarzschildObjective {
    /// Numerical aperture (typical: 0.08-0.33 for EUV).
    pub numerical_aperture: f64,
    /// Central obscuration ratio (fraction of pupil blocked by secondary mirror).
    /// Typically 0.2-0.4.
    pub obscuration_ratio: f64,
    /// Reduction ratio (e.g., 4.0 for 4x demagnification).
    pub reduction_ratio: f64,
    /// Per-mirror reflectivity (0-1). Squared for two-mirror system.
    pub mirror_reflectivity: f64,
    /// Flare fraction from mirror scatter.
    pub flare: f64,
}

impl SchwarzschildObjective {
    /// Create a Schwarzschild objective for EUV at 13.5nm.
    pub fn euv_standard() -> Self {
        Self {
            numerical_aperture: 0.33,
            obscuration_ratio: 0.25,
            reduction_ratio: 4.0,
            mirror_reflectivity: 0.67, // Mo/Si multilayer at 13.5nm
            flare: 0.03,
        }
    }

    /// Create for BEUV at 6.7nm.
    pub fn beuv() -> Self {
        Self {
            numerical_aperture: 0.25,
            obscuration_ratio: 0.3,
            reduction_ratio: 4.0,
            mirror_reflectivity: 0.50, // La/B4C multilayer at 6.7nm
            flare: 0.05,
        }
    }

    /// Create for soft X-ray at ~1nm.
    pub fn soft_xray(na: f64) -> crate::error::Result<Self> {
        if na.is_nan() || na <= 0.0 || na >= 1.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "numerical_aperture",
                value: if na.is_nan() { f64::NAN } else { na },
                reason: "must be in range (0, 1) and not NaN",
            });
        }
        Ok(Self {
            numerical_aperture: na,
            obscuration_ratio: 0.3,
            reduction_ratio: 1.0,     // typically 1:1 for X-ray microscopy
            mirror_reflectivity: 0.3, // grazing-incidence or multilayer
            flare: 0.05,
        })
    }

    /// Two-mirror system transmission: R².
    pub fn system_transmission(&self) -> f64 {
        self.mirror_reflectivity * self.mirror_reflectivity
    }
}

impl super::OpticalSystem for SchwarzschildObjective {
    fn pupil_function(
        &self,
        fx_norm: f64,
        fy_norm: f64,
        defocus_nm: f64,
        wavelength_nm: f64,
    ) -> Complex64 {
        let rho = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();

        // Outside aperture
        if rho > 1.0 {
            return Complex64::new(0.0, 0.0);
        }

        // Central obscuration (secondary mirror shadow)
        if rho < self.obscuration_ratio {
            return Complex64::new(0.0, 0.0);
        }

        // Transmission: two-mirror reflectivity
        let transmission = self.system_transmission().sqrt();

        // Defocus phase
        let defocus_phase = std::f64::consts::PI
            * defocus_nm
            * rho
            * rho
            * self.numerical_aperture
            * self.numerical_aperture
            / wavelength_nm;

        Complex::from_polar(transmission, defocus_phase)
    }

    fn na(&self) -> f64 {
        self.numerical_aperture
    }

    fn reduction(&self) -> f64 {
        self.reduction_ratio
    }

    fn flare_fraction(&self) -> f64 {
        self.flare
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::OpticalSystem;
    use approx::assert_relative_eq;

    #[test]
    fn test_euv_na() {
        let obj = SchwarzschildObjective::euv_standard();
        assert_relative_eq!(obj.na(), 0.33, epsilon = 1e-10);
    }

    #[test]
    fn test_annular_pupil() {
        let obj = SchwarzschildObjective::euv_standard();
        // Center is blocked
        let p_center = obj.pupil_function(0.0, 0.0, 0.0, 13.5);
        assert_relative_eq!(p_center.norm(), 0.0, epsilon = 1e-10);
        // Mid-ring passes
        let p_mid = obj.pupil_function(0.5, 0.0, 0.0, 13.5);
        assert!(p_mid.norm() > 0.0);
        // Outside blocked
        let p_out = obj.pupil_function(1.5, 0.0, 0.0, 13.5);
        assert_relative_eq!(p_out.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_system_transmission() {
        let obj = SchwarzschildObjective::euv_standard();
        // 0.67² ≈ 0.449
        assert_relative_eq!(obj.system_transmission(), 0.67 * 0.67, epsilon = 1e-10);
    }

    #[test]
    fn test_beuv_lower_reflectivity() {
        let euv = SchwarzschildObjective::euv_standard();
        let beuv = SchwarzschildObjective::beuv();
        assert!(
            beuv.system_transmission() < euv.system_transmission(),
            "BEUV should have lower transmission than EUV"
        );
    }

    #[test]
    fn test_euv_resolution() {
        let obj = SchwarzschildObjective::euv_standard();
        let res = obj.rayleigh_resolution(13.5);
        // 0.61 × 13.5 / 0.33 ≈ 24.9nm
        assert_relative_eq!(res, 24.9, epsilon = 0.5);
    }

    #[test]
    fn test_soft_xray_invalid_na() {
        assert!(SchwarzschildObjective::soft_xray(0.0).is_err());
        assert!(SchwarzschildObjective::soft_xray(-0.1).is_err());
        assert!(SchwarzschildObjective::soft_xray(1.0).is_err());
        assert!(SchwarzschildObjective::soft_xray(f64::NAN).is_err());
        // Valid NA should succeed
        assert!(SchwarzschildObjective::soft_xray(0.15).is_ok());
    }
}
