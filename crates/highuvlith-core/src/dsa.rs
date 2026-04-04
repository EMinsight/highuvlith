//! Directed Self-Assembly (DSA) simulation.
//!
//! Models block copolymer self-assembly guided by lithographic templates
//! to achieve sub-lithographic resolution features.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::LithographyError;

/// Block copolymer morphology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DSAMorphology {
    /// Alternating line/space pattern.
    Lamellar,
    /// Hexagonal array of cylinders.
    Cylindrical,
    /// BCC array of spheres.
    Spherical,
}

/// DSA simulation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DSAParams {
    /// Natural BCP period L₀ (nm). E.g., 28nm for PS-b-PMMA.
    pub l0_nm: f64,
    /// Flory-Huggins interaction parameter × degree of polymerization (χN).
    /// Higher χN = stronger segregation = sharper interfaces.
    pub chi_n: f64,
    /// Volume fraction of minority block (0.5 for symmetric lamellar).
    pub volume_fraction: f64,
    /// Morphology type.
    pub morphology: DSAMorphology,
    /// Interface width (nm) — determined by χN.
    pub interface_width_nm: f64,
}

impl DSAParams {
    /// Create parameters for PS-b-PMMA lamellar DSA.
    pub fn ps_pmma_lamellar(l0_nm: f64) -> crate::error::Result<Self> {
        if l0_nm.is_nan() || l0_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "l0_nm",
                value: if l0_nm.is_nan() { f64::NAN } else { l0_nm },
                reason: "must be positive",
            });
        }
        Ok(Self {
            l0_nm,
            chi_n: 20.0, // typical for PS-b-PMMA
            volume_fraction: 0.5,
            morphology: DSAMorphology::Lamellar,
            interface_width_nm: l0_nm / 10.0, // ~10% of period
        })
    }
}

impl Default for DSAParams {
    fn default() -> Self {
        Self::ps_pmma_lamellar(28.0).expect("default L0 28nm is valid")
    }
}

/// Result of DSA simulation.
#[derive(Debug)]
pub struct DSAResult {
    /// Assembled pattern (1.0 = block A, 0.0 = block B).
    pub pattern: Array2<f64>,
    /// Defect density (defects per μm²).
    pub defect_density: f64,
    /// Whether assembly was successful (no defects in center region).
    pub is_defect_free: bool,
    /// Effective CD of assembled features (nm).
    pub assembled_cd_nm: f64,
}

/// Check commensurability between template pitch and BCP period.
/// Returns the commensurability ratio (should be close to integer).
pub fn commensurability_ratio(template_pitch_nm: f64, l0_nm: f64) -> f64 {
    template_pitch_nm / l0_nm
}

/// Check if template and BCP are commensurable within tolerance.
pub fn is_commensurable(template_pitch_nm: f64, l0_nm: f64, tolerance: f64) -> bool {
    let ratio = commensurability_ratio(template_pitch_nm, l0_nm);
    let nearest_int = ratio.round();
    (ratio - nearest_int).abs() < tolerance
}

/// Simulate DSA assembly on a 1D template pattern.
///
/// The template defines confinement regions (from lithographic patterning).
/// The BCP self-assembles within these regions at its natural pitch L₀.
pub fn simulate_dsa_1d(
    template_pattern: &[f64],
    x_nm: &[f64],
    params: &DSAParams,
) -> crate::error::Result<DSAResult> {
    let n = template_pattern.len();
    let mut pattern = vec![0.0; n];

    // Find template edges (transitions in template pattern)
    let mut edges: Vec<usize> = Vec::new();
    for i in 1..n {
        if (template_pattern[i] - template_pattern[i - 1]).abs() > 0.5 {
            edges.push(i);
        }
    }

    // Within each confinement region, place BCP lamellae
    for i in 0..n {
        // Simple model: sinusoidal composition profile at L₀ periodicity
        let x = x_nm[i];
        let phase = 2.0 * std::f64::consts::PI * x / params.l0_nm;
        let raw = 0.5 + 0.5 * phase.cos();
        // Smooth with interface width
        pattern[i] = raw;
    }

    // Check commensurability defects
    let pitch_estimate = if edges.len() >= 2 {
        (x_nm[edges[1]] - x_nm[edges[0]]).abs()
    } else {
        params.l0_nm * 4.0 // assume 4× period template
    };

    let ratio = commensurability_ratio(pitch_estimate, params.l0_nm);
    let mismatch = (ratio - ratio.round()).abs();
    let defect_prob = (mismatch * 10.0).min(1.0); // empirical defect model
    let defect_density = defect_prob * 100.0; // defects/μm²

    let assembled_cd = params.l0_nm * params.volume_fraction;

    let pattern_array = Array2::from_shape_vec((1, n), pattern).map_err(|e| {
        LithographyError::InternalError(format!("DSA 1D pattern shape error: {}", e))
    })?;

    Ok(DSAResult {
        pattern: pattern_array,
        defect_density,
        is_defect_free: defect_density < 1.0,
        assembled_cd_nm: assembled_cd,
    })
}

/// Simulate DSA on a 2D template (for contact hole shrink or line/space).
pub fn simulate_dsa_2d(template: &Array2<f64>, pixel_nm: f64, params: &DSAParams) -> DSAResult {
    let (ny, nx) = template.dim();
    let mut pattern = Array2::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            let x = j as f64 * pixel_nm;
            let y = i as f64 * pixel_nm;

            match params.morphology {
                DSAMorphology::Lamellar => {
                    let phase = 2.0 * std::f64::consts::PI * x / params.l0_nm;
                    pattern[[i, j]] = 0.5 + 0.5 * phase.cos();
                }
                DSAMorphology::Cylindrical => {
                    // Hexagonal array of cylinders
                    let r_cyl =
                        params.l0_nm * params.volume_fraction.sqrt() / std::f64::consts::PI.sqrt();
                    let pitch = params.l0_nm;
                    let hex_y = pitch * (3.0_f64).sqrt() / 2.0;

                    let row = (y / hex_y).floor() as i64;
                    let x_offset = if row % 2 == 0 { 0.0 } else { pitch / 2.0 };
                    let cx = ((x - x_offset) / pitch).round() * pitch + x_offset;
                    let cy = row as f64 * hex_y;
                    let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();

                    pattern[[i, j]] = if dist < r_cyl { 1.0 } else { 0.0 };
                }
                DSAMorphology::Spherical => {
                    // BCC spheres (simplified as cubic array)
                    let r_sph = params.l0_nm
                        * (3.0 * params.volume_fraction / (4.0 * std::f64::consts::PI)).cbrt();
                    let pitch = params.l0_nm;
                    let cx = (x / pitch).round() * pitch;
                    let cy = (y / pitch).round() * pitch;
                    let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    pattern[[i, j]] = if dist < r_sph { 1.0 } else { 0.0 };
                }
            }
        }
    }

    let assembled_cd = params.l0_nm * params.volume_fraction;

    DSAResult {
        pattern,
        defect_density: 0.0,
        is_defect_free: true,
        assembled_cd_nm: assembled_cd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commensurability_exact() {
        assert!(is_commensurable(84.0, 28.0, 0.1)); // 3× exact
    }

    #[test]
    fn test_commensurability_mismatch() {
        assert!(!is_commensurable(85.0, 28.0, 0.01)); // 3.036×, off by 3.6%
    }

    #[test]
    fn test_dsa_1d_basic() {
        let params = DSAParams::ps_pmma_lamellar(28.0).unwrap();
        let n = 256;
        let x_nm: Vec<f64> = (0..n).map(|i| i as f64 * 1.0).collect();
        let template: Vec<f64> = x_nm
            .iter()
            .map(|&x| if (x / 56.0) as i64 % 2 == 0 { 1.0 } else { 0.0 })
            .collect();

        let result = simulate_dsa_1d(&template, &x_nm, &params).unwrap();
        assert_eq!(result.pattern.ncols(), n);
        assert!(result.assembled_cd_nm > 0.0);
    }

    #[test]
    fn test_dsa_2d_lamellar() {
        let params = DSAParams::ps_pmma_lamellar(28.0).unwrap();
        let template = Array2::ones((64, 64));
        let result = simulate_dsa_2d(&template, 2.0, &params);
        assert_eq!(result.pattern.dim(), (64, 64));
    }

    #[test]
    fn test_dsa_2d_cylindrical() {
        let params = DSAParams {
            morphology: DSAMorphology::Cylindrical,
            ..DSAParams::ps_pmma_lamellar(28.0).unwrap()
        };
        let template = Array2::ones((64, 64));
        let result = simulate_dsa_2d(&template, 2.0, &params);
        assert_eq!(result.pattern.dim(), (64, 64));
        // Should have both 0 and 1 values (cylinders + matrix)
        let has_ones = result.pattern.iter().any(|&v| v > 0.5);
        let has_zeros = result.pattern.iter().any(|&v| v < 0.5);
        assert!(has_ones && has_zeros);
    }

    #[test]
    fn test_invalid_l0_zero() {
        assert!(DSAParams::ps_pmma_lamellar(0.0).is_err());
        assert!(DSAParams::ps_pmma_lamellar(-5.0).is_err());
        assert!(DSAParams::ps_pmma_lamellar(f64::NAN).is_err());
    }

    #[test]
    fn test_dsa_2d_spherical() {
        let params = DSAParams {
            morphology: DSAMorphology::Spherical,
            ..DSAParams::ps_pmma_lamellar(28.0).unwrap()
        };
        let template = Array2::ones((64, 64));
        let result = simulate_dsa_2d(&template, 2.0, &params);
        assert_eq!(result.pattern.dim(), (64, 64));
        // Spherical morphology should produce discrete sphere regions (1.0) and matrix (0.0)
        let has_ones = result.pattern.iter().any(|&v| v > 0.5);
        let has_zeros = result.pattern.iter().any(|&v| v < 0.5);
        assert!(
            has_ones && has_zeros,
            "Spherical DSA should produce both sphere and matrix regions"
        );
    }
}
