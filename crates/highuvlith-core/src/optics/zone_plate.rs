//! Fresnel zone plate optics for X-ray lithography.
//!
//! Zone plates are diffractive focusing elements — the primary optic for
//! X-ray microscopy and lithography. Resolution is determined by the
//! outermost zone width Δr_N.

use num::Complex;
use serde::{Deserialize, Serialize};

use crate::types::Complex64;

/// Zone plate diffraction efficiency model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZonePlateEfficiency {
    /// Binary amplitude zone plate (~10% into first order).
    Binary,
    /// Phase zone plate with π phase shift (~40% first-order efficiency).
    Phase,
    /// Blazed (graded profile) zone plate (up to ~100% theoretical).
    Blazed { efficiency: f64 },
}

impl ZonePlateEfficiency {
    fn first_order_efficiency(&self) -> f64 {
        match self {
            Self::Binary => 1.0 / (std::f64::consts::PI * std::f64::consts::PI), // 1/π² ≈ 10.1%
            Self::Phase => 4.0 / (std::f64::consts::PI * std::f64::consts::PI), // 4/π² ≈ 40.5%
            Self::Blazed { efficiency } => *efficiency,
        }
    }
}

/// Fresnel zone plate optical system.
///
/// Zone radii follow: r_n = √(n × λ × f)
/// Resolution limit: ~Δr_N (outermost zone width)
/// NA = λ / (2 × Δr_N)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FresnelZonePlate {
    /// Outermost zone width (nm) — determines resolution.
    pub outermost_zone_width_nm: f64,
    /// Number of zones.
    pub num_zones: usize,
    /// Design wavelength (nm).
    pub design_wavelength_nm: f64,
    /// Efficiency model.
    pub efficiency: ZonePlateEfficiency,
    /// Central stop fraction (0-1). Blocks zero-order undiffracted light.
    pub central_stop_fraction: f64,
    /// Reduction ratio (e.g., 1.0 for 1:1, typically 1.0 for zone plate lithography).
    pub reduction_ratio: f64,
}

impl FresnelZonePlate {
    /// Create a zone plate for a given resolution and wavelength.
    pub fn new(outermost_zone_width_nm: f64, design_wavelength_nm: f64) -> Self {
        // Number of zones: N = r_N / Δr_N, where r_N² = N × λ × f
        // For given Δr_N and λ: f = (2 × Δr_N)² / λ (from NA = λ/(2×Δr_N))
        // r_N = N × Δr_N, so N = r_N / Δr_N
        // Typical: ~100-1000 zones
        let na = design_wavelength_nm / (2.0 * outermost_zone_width_nm);
        let focal_length_nm = outermost_zone_width_nm / na; // simplified
        let r_n_sq = focal_length_nm * design_wavelength_nm;
        let num_zones = (r_n_sq / (outermost_zone_width_nm * outermost_zone_width_nm))
            .sqrt()
            .ceil() as usize;

        Self {
            outermost_zone_width_nm,
            num_zones: num_zones.max(10),
            design_wavelength_nm,
            efficiency: ZonePlateEfficiency::Phase,
            central_stop_fraction: 0.0,
            reduction_ratio: 1.0,
        }
    }

    /// Focal length in nm: f = r_N² / (N × λ).
    pub fn focal_length_nm(&self) -> f64 {
        let r_n = self.num_zones as f64 * self.outermost_zone_width_nm;
        r_n * r_n / (self.num_zones as f64 * self.design_wavelength_nm)
    }

    /// Diameter of the zone plate in nm.
    pub fn diameter_nm(&self) -> f64 {
        2.0 * self.num_zones as f64 * self.outermost_zone_width_nm
    }

    /// Chromatic aberration: Δf/f = Δλ/λ (zone plates are strongly chromatic).
    pub fn chromatic_defocus_per_pm(&self) -> f64 {
        let f = self.focal_length_nm();
        // Δf = f × Δλ/λ; for Δλ in pm: Δf = f × (Δλ_pm × 1e-3) / λ
        f * 1e-3 / self.design_wavelength_nm
    }
}

impl super::OpticalSystem for FresnelZonePlate {
    fn pupil_function(
        &self,
        fx_norm: f64,
        fy_norm: f64,
        defocus_nm: f64,
        wavelength_nm: f64,
    ) -> Complex64 {
        let rho = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();

        // Outside zone plate aperture
        if rho > 1.0 {
            return Complex64::new(0.0, 0.0);
        }

        // Central stop blocks low spatial frequencies
        if rho < self.central_stop_fraction {
            return Complex64::new(0.0, 0.0);
        }

        // Zone plate transmission includes efficiency factor
        let transmission = self.efficiency.first_order_efficiency().sqrt();

        // Defocus phase (same as refractive optics)
        let na = self.na();
        let defocus_phase =
            std::f64::consts::PI * defocus_nm * rho * rho * na * na / wavelength_nm;

        // Zone plate introduces additional chromatic phase error
        // when operating off-design wavelength
        let chromatic_phase = if (wavelength_nm - self.design_wavelength_nm).abs() > 1e-6 {
            let delta_f = self.focal_length_nm()
                * (wavelength_nm - self.design_wavelength_nm)
                / self.design_wavelength_nm;
            std::f64::consts::PI * delta_f * rho * rho * na * na / wavelength_nm
        } else {
            0.0
        };

        Complex::from_polar(transmission, defocus_phase + chromatic_phase)
    }

    fn na(&self) -> f64 {
        self.design_wavelength_nm / (2.0 * self.outermost_zone_width_nm)
    }

    fn reduction(&self) -> f64 {
        self.reduction_ratio
    }

    fn flare_fraction(&self) -> f64 {
        // Zone plates have significant zero-order and higher-order diffraction
        // that contributes to background (if no central stop)
        if self.central_stop_fraction > 0.0 {
            0.05 // with central stop
        } else {
            0.15 // without central stop, significant zero-order leakage
        }
    }

    fn chromatic_defocus(&self, delta_wavelength_pm: f64) -> f64 {
        self.chromatic_defocus_per_pm() * delta_wavelength_pm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::OpticalSystem;
    use approx::assert_relative_eq;

    #[test]
    fn test_zone_plate_na() {
        // 25nm outermost zone at 1nm wavelength: NA = 1/(2×25) = 0.02
        let zp = FresnelZonePlate::new(25.0, 1.0);
        assert_relative_eq!(zp.na(), 0.02, epsilon = 0.001);
    }

    #[test]
    fn test_zone_plate_resolution() {
        // Resolution ≈ outermost zone width
        let zp = FresnelZonePlate::new(15.0, 0.5);
        let res = zp.rayleigh_resolution(0.5);
        // Rayleigh = 0.61 × λ / NA = 0.61 × 0.5 / (0.5/(2×15)) = 0.61 × 0.5 / 0.0167 ≈ 18.3
        // This should be close to the outermost zone width
        assert!(
            (res - 15.0).abs() < 5.0,
            "Resolution {:.1}nm should be close to zone width 15nm",
            res
        );
    }

    #[test]
    fn test_pupil_inside() {
        let zp = FresnelZonePlate::new(25.0, 1.0);
        let p = zp.pupil_function(0.5, 0.0, 0.0, 1.0);
        assert!(p.norm() > 0.0, "Pupil should transmit inside aperture");
    }

    #[test]
    fn test_pupil_outside() {
        let zp = FresnelZonePlate::new(25.0, 1.0);
        let p = zp.pupil_function(1.5, 0.0, 0.0, 1.0);
        assert_relative_eq!(p.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_central_stop() {
        let mut zp = FresnelZonePlate::new(25.0, 1.0);
        zp.central_stop_fraction = 0.3;
        let p_center = zp.pupil_function(0.1, 0.0, 0.0, 1.0);
        let p_edge = zp.pupil_function(0.5, 0.0, 0.0, 1.0);
        assert_relative_eq!(p_center.norm(), 0.0, epsilon = 1e-10);
        assert!(p_edge.norm() > 0.0);
    }

    #[test]
    fn test_strong_chromatic_aberration() {
        let zp = FresnelZonePlate::new(25.0, 1.0);
        // Zone plates: Δf/f = Δλ/λ, so chromatic defocus is very large
        let chrom = zp.chromatic_defocus(1.0); // 1 pm offset
        assert!(
            chrom.abs() > 0.0,
            "Zone plates should have strong chromatic aberration"
        );
    }

    #[test]
    fn test_efficiency_values() {
        assert_relative_eq!(
            ZonePlateEfficiency::Binary.first_order_efficiency(),
            0.101,
            epsilon = 0.01
        );
        assert_relative_eq!(
            ZonePlateEfficiency::Phase.first_order_efficiency(),
            0.405,
            epsilon = 0.01
        );
    }
}
