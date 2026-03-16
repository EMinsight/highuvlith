//! Diamond substrate modeling for lithography.
//!
//! Diamond (C) is emerging as a next-gen substrate for quantum computing,
//! power electronics, and X-ray window applications due to its extreme
//! thermal conductivity and optical properties.

use crate::materials::dispersion::SellmeierCoefficients;
use crate::thinfilm::{FilmLayer, FilmStack};
use crate::types::Complex64;

/// Diamond Sellmeier coefficients (type IIa, UV-visible-IR).
/// Reference: Peter, 1923; valid ~225nm to far-IR.
pub fn diamond_sellmeier() -> SellmeierCoefficients {
    SellmeierCoefficients {
        b: vec![0.3306, 4.3356],
        c: vec![0.00030625, 0.011236],
    }
}

/// Diamond thermal and mechanical properties.
pub struct DiamondProperties {
    /// Thermal conductivity (W/m·K). Natural diamond: ~2200.
    pub thermal_conductivity: f64,
    /// Thermal expansion coefficient (1/K).
    pub thermal_expansion: f64,
    /// Bandgap energy (eV). Type IIa: 5.47 eV → UV cutoff at ~227nm.
    pub bandgap_ev: f64,
    /// Density (g/cm³).
    pub density: f64,
}

impl Default for DiamondProperties {
    fn default() -> Self {
        Self {
            thermal_conductivity: 2200.0,
            thermal_expansion: 1.0e-6,
            bandgap_ev: 5.47,
            density: 3.515,
        }
    }
}

impl DiamondProperties {
    /// UV cutoff wavelength (nm) from bandgap.
    pub fn uv_cutoff_nm(&self) -> f64 {
        super::energy::ev_to_nm(self.bandgap_ev)
    }

    /// Estimate maximum dose (mJ/cm²) before thermal distortion exceeds tolerance.
    /// Simple model: ΔT = dose / (ρ × c_p × thickness), distortion = α × ΔT × area^0.5
    pub fn max_dose_mj_cm2(
        &self,
        thickness_um: f64,
        distortion_tolerance_nm: f64,
    ) -> f64 {
        let specific_heat = 0.509; // J/(g·K) for diamond
        let rho_cgs = self.density; // g/cm³
        let thickness_cm = thickness_um * 1e-4;
        // ΔT for distortion: distortion = α × ΔT × characteristic_length
        // Simplified: max_delta_t = tolerance / (α × thickness_um * 1000)
        let max_delta_t = distortion_tolerance_nm / (self.thermal_expansion * thickness_um * 1e3);
        // dose = ρ × c_p × thickness × ΔT (in J/cm²)
        let dose_j_cm2 = rho_cgs * specific_heat * thickness_cm * max_delta_t;
        dose_j_cm2 * 1e3 // convert to mJ/cm²
    }
}

/// Create a resist-on-diamond film stack for VUV lithography.
pub fn resist_on_diamond(resist_thickness_nm: f64, wavelength_nm: f64) -> FilmStack {
    let sellmeier = diamond_sellmeier();
    let n_diamond = if wavelength_nm > 225.0 {
        sellmeier.refractive_index(wavelength_nm)
    } else {
        2.7 // approximate for deep UV (absorbing regime)
    };

    FilmStack::new_vuv(
        vec![FilmLayer {
            name: "resist".to_string(),
            thickness_nm: resist_thickness_nm,
            n: Complex64::new(1.65, 0.015), // VUV fluoropolymer
        }],
        Complex64::new(n_diamond, 0.0), // diamond substrate (transparent)
    )
}

/// Create a diamond-on-silicon film stack (diamond membrane).
pub fn diamond_on_silicon(diamond_thickness_nm: f64, wavelength_nm: f64) -> FilmStack {
    let sellmeier = diamond_sellmeier();
    let n_diamond = if wavelength_nm > 225.0 {
        sellmeier.refractive_index(wavelength_nm)
    } else {
        2.7
    };

    FilmStack::new_vuv(
        vec![FilmLayer {
            name: "diamond".to_string(),
            thickness_nm: diamond_thickness_nm,
            n: Complex64::new(n_diamond, 0.0),
        }],
        Complex64::new(0.88, 2.10), // Si at 157nm
    )
}

/// X-ray transmission through a diamond window.
/// At hard X-ray energies (>5 keV), diamond is nearly transparent.
pub fn xray_transmission(thickness_um: f64, energy_kev: f64) -> f64 {
    // Carbon linear absorption coefficient (approximate, from NIST)
    // μ/ρ ≈ 4.6 cm²/g at 8 keV, scales as ~E^-3
    let mu_rho_8kev = 4.6; // cm²/g
    let mu_rho = mu_rho_8kev * (8.0 / energy_kev).powi(3);
    let mu = mu_rho * 3.515; // linear coefficient (1/cm)
    let thickness_cm = thickness_um * 1e-4;
    (-mu * thickness_cm).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_diamond_refractive_index() {
        let s = diamond_sellmeier();
        let n = s.refractive_index(589.0); // sodium D line
        assert_relative_eq!(n, 2.417, epsilon = 0.01);
    }

    #[test]
    fn test_diamond_uv_cutoff() {
        let props = DiamondProperties::default();
        let cutoff = props.uv_cutoff_nm();
        assert!(cutoff > 225.0 && cutoff < 230.0, "Diamond UV cutoff should be ~227nm, got {}", cutoff);
    }

    #[test]
    fn test_diamond_thermal_capacity() {
        let props = DiamondProperties::default();
        // Diamond should tolerate very high doses due to thermal conductivity
        let max_dose = props.max_dose_mj_cm2(500.0, 1.0);
        assert!(max_dose > 100.0, "Diamond should tolerate high dose, got {} mJ/cm²", max_dose);
    }

    #[test]
    fn test_xray_transmission_high_energy() {
        // At 10 keV, 100μm diamond should be highly transparent
        let t = xray_transmission(100.0, 10.0);
        assert!(t > 0.9, "Diamond should be >90% transparent at 10 keV, got {:.1}%", t * 100.0);
    }

    #[test]
    fn test_xray_transmission_low_energy() {
        // At 1 keV, diamond absorbs more
        let t_low = xray_transmission(100.0, 1.0);
        let t_high = xray_transmission(100.0, 10.0);
        assert!(t_low < t_high, "Lower energy should have more absorption");
    }

    #[test]
    fn test_resist_on_diamond_stack() {
        let stack = resist_on_diamond(150.0, 589.0);
        assert_eq!(stack.layers.len(), 1);
        // Diamond substrate has high real index, near-zero imaginary
        assert!(stack.substrate.im.abs() < 0.01);
    }
}
