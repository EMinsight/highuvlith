//! Energy-wavelength conversion utilities for X-ray and UV lithography.

/// Convert photon energy (eV) to wavelength (nm).
pub fn ev_to_nm(ev: f64) -> f64 {
    1239.84193 / ev
}

/// Convert wavelength (nm) to photon energy (eV).
pub fn nm_to_ev(nm: f64) -> f64 {
    1239.84193 / nm
}

/// Convert photon energy (keV) to wavelength (nm).
pub fn kev_to_nm(kev: f64) -> f64 {
    1.23984193 / kev
}

/// Convert wavelength (nm) to photon energy (keV).
pub fn nm_to_kev(nm: f64) -> f64 {
    1.23984193 / nm
}

/// Planck constant times speed of light in eV·nm.
pub const HC_EV_NM: f64 = 1239.84193;

/// Classical electron radius in nm.
pub const R_ELECTRON_NM: f64 = 2.8179403e-6;

/// Avogadro's number.
pub const AVOGADRO: f64 = 6.02214076e23;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_roundtrip() {
        assert_relative_eq!(ev_to_nm(nm_to_ev(157.63)), 157.63, epsilon = 1e-6);
    }

    #[test]
    fn test_known_values() {
        // F2 laser: 157.63nm = 7.87 eV
        assert_relative_eq!(nm_to_ev(157.63), 7.87, epsilon = 0.01);
        // Cu K-alpha: 0.154nm = 8.05 keV
        assert_relative_eq!(nm_to_kev(0.154), 8.05, epsilon = 0.05);
        // EUV: 13.5nm = 91.8 eV
        assert_relative_eq!(nm_to_ev(13.5), 91.8, epsilon = 0.5);
    }
}
