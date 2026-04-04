pub mod schwarzschild;
pub mod zone_plate;

use num::Complex;
use serde::{Deserialize, Serialize};

use crate::math::zernike;
use crate::types::Complex64;

/// Trait for any optical system that produces a pupil function.
/// The aerial image engine uses this interface, enabling refractive lenses,
/// zone plates, grazing-incidence mirrors, and Schwarzschild objectives.
pub trait OpticalSystem: Send + Sync {
    /// Evaluate pupil function at normalized frequency (fx, fy).
    fn pupil_function(
        &self,
        fx_norm: f64,
        fy_norm: f64,
        defocus_nm: f64,
        wavelength_nm: f64,
    ) -> Complex64;

    /// Image-side numerical aperture.
    fn na(&self) -> f64;

    /// Reduction ratio (e.g., 4.0 for 4x demagnification).
    fn reduction(&self) -> f64;

    /// Stray light / flare fraction.
    fn flare_fraction(&self) -> f64;

    /// Chromatic defocus for a wavelength offset (pm). Default: 0 (achromatic).
    fn chromatic_defocus(&self, _delta_wavelength_pm: f64) -> f64 {
        0.0
    }

    /// Cutoff frequency in 1/nm.
    fn cutoff_frequency(&self, wavelength_nm: f64) -> f64 {
        self.na() / wavelength_nm
    }

    /// Rayleigh resolution limit in nm.
    fn rayleigh_resolution(&self, wavelength_nm: f64) -> f64 {
        0.61 * wavelength_nm / self.na()
    }
}

/// Pupil apodization model (transmission variation across the pupil).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum Apodization {
    /// Uniform transmission.
    #[default]
    Uniform,
    /// Radial transmission profile: T(rho) = 1 - alpha * rho^2.
    Quadratic { alpha: f64 },
    /// Gaussian apodization: T(rho) = exp(-alpha * rho^2).
    Gaussian { alpha: f64 },
}

/// Projection optics specification for VUV lithography.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionOptics {
    /// Numerical aperture (image side).
    pub na: f64,
    /// Reduction ratio (e.g., 4.0 for 4x reduction).
    pub reduction: f64,
    /// Zernike aberration coefficients: (fringe_index, coefficient_in_waves).
    pub zernike_coefficients: Vec<(usize, f64)>,
    /// Flare fraction (stray light added as uniform background).
    pub flare_fraction: f64,
    /// Axial chromatic aberration coefficient (nm defocus per pm bandwidth).
    /// For CaF2-only lens at 157nm, typical value ~10-30 nm/pm.
    pub axial_chromatic_nm_per_pm: f64,
    /// Pupil apodization from coating effects.
    pub apodization: Apodization,
}

impl ProjectionOptics {
    /// Create optics with default VUV parameters.
    pub fn new(na: f64) -> crate::error::Result<Self> {
        if na.is_nan() || na <= 0.0 || na >= 1.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "numerical_aperture",
                value: if na.is_nan() { f64::NAN } else { na },
                reason: "must be in range (0, 1) and not NaN",
            });
        }
        Ok(Self {
            na,
            reduction: 4.0,
            zernike_coefficients: Vec::new(),
            flare_fraction: 0.02, // 2% flare typical for VUV
            axial_chromatic_nm_per_pm: 15.0,
            apodization: Apodization::default(),
        })
    }

    /// Maximum spatial frequency that passes through the pupil, in 1/nm.
    pub fn cutoff_frequency(&self, wavelength_nm: f64) -> f64 {
        self.na / wavelength_nm
    }

    /// Rayleigh resolution limit: 0.61 * lambda / NA.
    pub fn rayleigh_resolution(&self, wavelength_nm: f64) -> f64 {
        0.61 * wavelength_nm / self.na
    }

    /// Depth of focus (Rayleigh criterion): +/- lambda / (2 * NA^2).
    pub fn rayleigh_dof(&self, wavelength_nm: f64) -> f64 {
        wavelength_nm / (2.0 * self.na * self.na)
    }

    /// Evaluate the pupil function at normalized frequency (fx, fy),
    /// where frequencies are in units of NA/lambda.
    ///
    /// Returns Complex64: magnitude is transmission, phase includes
    /// aberrations and defocus.
    pub fn pupil_function(
        &self,
        fx_norm: f64,
        fy_norm: f64,
        defocus_nm: f64,
        wavelength_nm: f64,
    ) -> Complex64 {
        let rho = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();

        // Outside pupil
        if rho > 1.0 {
            return Complex64::new(0.0, 0.0);
        }

        let theta = fy_norm.atan2(fx_norm);

        // Aberration phase from Zernike coefficients
        let aberration_phase = zernike::pupil_phase(&self.zernike_coefficients, rho, theta);

        // Defocus phase: W_defocus = defocus * rho^2 * NA^2 / (2 * lambda)
        // Expressed in radians: 2*pi/lambda * defocus * rho^2 * NA^2 / 2
        let defocus_phase =
            std::f64::consts::PI * defocus_nm * rho * rho * self.na * self.na / wavelength_nm;

        let total_phase = aberration_phase + defocus_phase;

        // Pupil transmission (apodization)
        let transmission = match &self.apodization {
            Apodization::Uniform => 1.0,
            Apodization::Quadratic { alpha } => (1.0 - alpha * rho * rho).max(0.0),
            Apodization::Gaussian { alpha } => (-alpha * rho * rho).exp(),
        };

        Complex::from_polar(transmission, total_phase)
    }

    /// Chromatic defocus for a wavelength offset from center (in pm).
    pub fn chromatic_defocus(&self, delta_wavelength_pm: f64) -> f64 {
        self.axial_chromatic_nm_per_pm * delta_wavelength_pm
    }
}

impl OpticalSystem for ProjectionOptics {
    fn pupil_function(
        &self,
        fx_norm: f64,
        fy_norm: f64,
        defocus_nm: f64,
        wavelength_nm: f64,
    ) -> Complex64 {
        // Delegate to inherent method via UFCS
        ProjectionOptics::pupil_function(self, fx_norm, fy_norm, defocus_nm, wavelength_nm)
    }

    fn na(&self) -> f64 {
        self.na
    }

    fn reduction(&self) -> f64 {
        self.reduction
    }

    fn flare_fraction(&self) -> f64 {
        self.flare_fraction
    }

    fn chromatic_defocus(&self, delta_wavelength_pm: f64) -> f64 {
        ProjectionOptics::chromatic_defocus(self, delta_wavelength_pm)
    }
}

impl Default for ProjectionOptics {
    fn default() -> Self {
        Self::new(0.75).expect("default NA 0.75 is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cutoff_frequency() {
        let optics = ProjectionOptics::new(0.75).unwrap();
        let fc = optics.cutoff_frequency(157.0);
        assert_relative_eq!(fc, 0.75 / 157.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rayleigh_resolution() {
        let optics = ProjectionOptics::new(0.75).unwrap();
        let res = optics.rayleigh_resolution(157.0);
        // 0.61 * 157 / 0.75 = 127.7 nm
        assert_relative_eq!(res, 0.61 * 157.0 / 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_pupil_inside() {
        let optics = ProjectionOptics::new(0.75).unwrap();
        let p = optics.pupil_function(0.0, 0.0, 0.0, 157.0);
        assert_relative_eq!(p.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pupil_outside() {
        let optics = ProjectionOptics::new(0.75).unwrap();
        let p = optics.pupil_function(1.5, 0.0, 0.0, 157.0);
        assert_relative_eq!(p.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pupil_at_edge() {
        let optics = ProjectionOptics::new(0.75).unwrap();
        let p = optics.pupil_function(1.0, 0.0, 0.0, 157.0);
        assert_relative_eq!(p.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_defocus_adds_phase() {
        let optics = ProjectionOptics::new(0.75).unwrap();
        let p0 = optics.pupil_function(0.5, 0.0, 0.0, 157.0);
        let p1 = optics.pupil_function(0.5, 0.0, 100.0, 157.0);
        // Same magnitude, different phase
        assert_relative_eq!(p0.norm(), p1.norm(), epsilon = 1e-10);
        assert!((p0.arg() - p1.arg()).abs() > 1e-6);
    }

    #[test]
    fn test_aberrated_pupil() {
        let mut optics = ProjectionOptics::new(0.75).unwrap();
        // Add spherical aberration (Z9)
        optics.zernike_coefficients.push((9, 0.05));
        let p = optics.pupil_function(0.5, 0.0, 0.0, 157.0);
        // Should still have unit magnitude (no apodization)
        assert_relative_eq!(p.norm(), 1.0, epsilon = 1e-10);
        // But non-zero phase
        assert!(p.arg().abs() > 1e-6);
    }

    #[test]
    fn test_invalid_na_zero() {
        assert!(ProjectionOptics::new(0.0).is_err());
    }

    #[test]
    fn test_invalid_na_one() {
        assert!(ProjectionOptics::new(1.0).is_err());
        assert!(ProjectionOptics::new(-0.5).is_err());
        assert!(ProjectionOptics::new(f64::NAN).is_err());
    }
}
