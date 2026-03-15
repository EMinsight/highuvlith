use serde::{Deserialize, Serialize};

/// Sellmeier dispersion model: n^2 = 1 + sum_i(B_i * lambda^2 / (lambda^2 - C_i))
/// where lambda is in micrometers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SellmeierCoefficients {
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

impl SellmeierCoefficients {
    /// Evaluate refractive index at a given wavelength (in nm).
    pub fn refractive_index(&self, wavelength_nm: f64) -> f64 {
        let lambda_um = wavelength_nm * 1e-3;
        let lambda_sq = lambda_um * lambda_um;

        let mut n_sq = 1.0;
        for (bi, ci) in self.b.iter().zip(self.c.iter()) {
            n_sq += bi * lambda_sq / (lambda_sq - ci);
        }

        n_sq.sqrt()
    }

    /// Evaluate dn/dlambda at a given wavelength (in nm). Returns dn/dlambda in 1/nm.
    pub fn dispersion(&self, wavelength_nm: f64) -> f64 {
        let lambda_um = wavelength_nm * 1e-3;
        let lambda_sq = lambda_um * lambda_um;

        let mut n_sq = 1.0;
        let mut dn_sq_dlambda = 0.0;
        for (bi, ci) in self.b.iter().zip(self.c.iter()) {
            let denom = lambda_sq - ci;
            n_sq += bi * lambda_sq / denom;
            // d(n^2)/d(lambda) = B_i * (-2 * lambda * C_i) / (lambda^2 - C_i)^2
            dn_sq_dlambda += bi * (-2.0 * lambda_um * ci) / (denom * denom);
        }

        let n = n_sq.sqrt();
        // dn/dlambda_um = (1/(2n)) * dn^2/dlambda_um
        let dn_dlambda_um = dn_sq_dlambda / (2.0 * n);
        // Convert to per nm
        dn_dlambda_um * 1e-3
    }
}

/// CaF2 Sellmeier coefficients valid from ~130nm to ~10um.
/// Reference: Daimon & Masumura, Appl. Opt. 41, 5275 (2002).
pub fn caf2_sellmeier() -> SellmeierCoefficients {
    SellmeierCoefficients {
        b: vec![0.5675888, 0.4710914, 3.8484723],
        c: vec![0.00252643, 0.01007833, 1200.5560],
    }
}

/// MgF2 (ordinary ray) Sellmeier coefficients.
/// Reference: Dodge, Appl. Opt. 23, 1980 (1984).
pub fn mgf2_ordinary_sellmeier() -> SellmeierCoefficients {
    SellmeierCoefficients {
        b: vec![0.48755108, 0.39875031, 2.3120353],
        c: vec![0.00188217, 0.00886863, 566.13559],
    }
}

/// LiF Sellmeier coefficients.
/// Reference: Li, J. Phys. Chem. Ref. Data 5, 329 (1976).
pub fn lif_sellmeier() -> SellmeierCoefficients {
    SellmeierCoefficients {
        b: vec![0.92549, 6.96747],
        c: vec![0.00784, 566.0],
    }
}

/// BaF2 Sellmeier coefficients.
pub fn baf2_sellmeier() -> SellmeierCoefficients {
    SellmeierCoefficients {
        b: vec![0.643356, 0.506762, 3.8261],
        c: vec![0.00301679, 0.01294200, 2045.76],
    }
}

/// Fused silica (SiO2) Sellmeier coefficients.
/// Note: absorbs below ~165nm, only valid for DUV reference.
pub fn sio2_sellmeier() -> SellmeierCoefficients {
    SellmeierCoefficients {
        b: vec![0.6961663, 0.4079426, 0.8974794],
        c: vec![0.00467914826, 0.01351206324, 97.93400254],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_caf2_at_157nm() {
        let sellmeier = caf2_sellmeier();
        let n = sellmeier.refractive_index(157.0);
        // CaF2 at 157nm: n ~ 1.559
        assert_relative_eq!(n, 1.559, epsilon = 0.02);
    }

    #[test]
    fn test_caf2_at_visible() {
        let sellmeier = caf2_sellmeier();
        let n = sellmeier.refractive_index(589.0); // sodium D line
        // CaF2 at 589nm: n ~ 1.434
        assert_relative_eq!(n, 1.434, epsilon = 0.005);
    }

    #[test]
    fn test_dispersion_negative() {
        let sellmeier = caf2_sellmeier();
        // Normal dispersion: dn/dlambda < 0
        let disp = sellmeier.dispersion(157.0);
        assert!(disp < 0.0, "CaF2 should have normal dispersion at 157nm");
    }

    #[test]
    fn test_caf2_dispersion_steeper_at_vuv() {
        let sellmeier = caf2_sellmeier();
        let disp_157 = sellmeier.dispersion(157.0).abs();
        let disp_589 = sellmeier.dispersion(589.0).abs();
        // Dispersion is much steeper at VUV than visible
        assert!(disp_157 > disp_589 * 5.0);
    }
}
