use ndarray::Array2;
use num::Complex;
use serde::{Deserialize, Serialize};

use crate::math::fft2d::Fft2D;
use crate::types::{Complex64, GridConfig};

/// Mask type (determines phase and transmission of dark regions).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MaskType {
    /// Binary chrome-on-glass mask.
    #[default]
    Binary,
    /// Attenuated phase-shift mask.
    AttenuatedPSM {
        /// Transmission of the absorber (typically 0.06 = 6%).
        transmission: f64,
        /// Phase shift in degrees (typically 180).
        phase_deg: f64,
    },
    /// Alternating aperture phase-shift mask.
    AlternatingPSM,
}

/// A geometric feature on the mask (at wafer scale).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaskFeature {
    /// Rectangle defined by center (x, y) and dimensions (w, h).
    Rect { x: f64, y: f64, w: f64, h: f64 },
    /// Polygon defined by vertices (x, y pairs).
    Polygon { vertices: Vec<(f64, f64)> },
}

/// Mask / reticle specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mask {
    pub mask_type: MaskType,
    /// Features are "bright" regions (clear/transparent). Background is dark.
    pub features: Vec<MaskFeature>,
    /// If true, features are dark (opaque) and background is clear.
    pub dark_field: bool,
}

impl Mask {
    /// Create a 1D line/space pattern (bright-field: lines are opaque).
    /// cd_nm: line width at wafer scale.
    /// pitch_nm: line+space pitch at wafer scale.
    /// The pattern is centered and repeated across the field.
    pub fn line_space(cd_nm: f64, pitch_nm: f64) -> crate::error::Result<Self> {
        if cd_nm.is_nan() || cd_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "cd_nm",
                value: cd_nm,
                reason: "must be positive",
            });
        }
        if pitch_nm.is_nan() || pitch_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "pitch_nm",
                value: pitch_nm,
                reason: "must be positive",
            });
        }
        if cd_nm >= pitch_nm {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "cd_nm",
                value: cd_nm,
                reason: "must be less than pitch_nm",
            });
        }
        // Create spaces (clear regions) between lines
        let num_periods = 10; // enough to fill a typical field
        let mut features = Vec::new();
        let space_width = pitch_nm - cd_nm;
        let total_width = num_periods as f64 * pitch_nm;
        let start = -total_width / 2.0;

        for i in 0..num_periods {
            let line_center_x = start + (i as f64 + 0.5) * pitch_nm;
            // Space is centered between lines
            let space_center_x = line_center_x + pitch_nm / 2.0;
            if i < num_periods - 1 {
                features.push(MaskFeature::Rect {
                    x: space_center_x,
                    y: 0.0,
                    w: space_width,
                    h: total_width, // infinite in y (periodic)
                });
            }
        }

        // For bright-field L/S, the features are the clear spaces,
        // and lines are the dark background
        Ok(Self {
            mask_type: MaskType::Binary,
            features,
            dark_field: false,
        })
    }

    /// Create a contact hole pattern.
    pub fn contact_hole(
        diameter_nm: f64,
        pitch_x_nm: f64,
        pitch_y_nm: f64,
    ) -> crate::error::Result<Self> {
        if diameter_nm.is_nan() || diameter_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "diameter_nm",
                value: diameter_nm,
                reason: "must be positive",
            });
        }
        if pitch_x_nm.is_nan() || pitch_x_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "pitch_x_nm",
                value: pitch_x_nm,
                reason: "must be positive",
            });
        }
        if pitch_y_nm.is_nan() || pitch_y_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "pitch_y_nm",
                value: pitch_y_nm,
                reason: "must be positive",
            });
        }
        let num_x = 5;
        let num_y = 5;
        let mut features = Vec::new();
        let start_x = -(num_x as f64) / 2.0 * pitch_x_nm;
        let start_y = -(num_y as f64) / 2.0 * pitch_y_nm;

        for iy in 0..num_y {
            for ix in 0..num_x {
                features.push(MaskFeature::Rect {
                    x: start_x + (ix as f64 + 0.5) * pitch_x_nm,
                    y: start_y + (iy as f64 + 0.5) * pitch_y_nm,
                    w: diameter_nm,
                    h: diameter_nm,
                });
            }
        }

        Ok(Self {
            mask_type: MaskType::Binary,
            features,
            dark_field: true,
        })
    }

    /// Rasterize the mask to a 2D complex transmittance array.
    pub fn rasterize(&self, grid: &GridConfig) -> Array2<Complex64> {
        let n = grid.size;
        let field = grid.field_size_nm();
        let half = field / 2.0;
        let pixel = grid.pixel_nm;

        // Start with background transmittance
        let bg = if self.dark_field {
            self.absorber_transmittance()
        } else {
            Complex64::new(1.0, 0.0) // clear background
        };
        let fg = if self.dark_field {
            Complex64::new(1.0, 0.0) // clear features
        } else {
            self.absorber_transmittance()
        };

        let mut mask = Array2::from_elem((n, n), bg);

        for feature in &self.features {
            match feature {
                MaskFeature::Rect { x, y, w, h } => {
                    let x_min = x - w / 2.0;
                    let x_max = x + w / 2.0;
                    let y_min = y - h / 2.0;
                    let y_max = y + h / 2.0;

                    for iy in 0..n {
                        let py = -half + (iy as f64 + 0.5) * pixel;
                        if py < y_min || py > y_max {
                            continue;
                        }
                        for ix in 0..n {
                            let px = -half + (ix as f64 + 0.5) * pixel;
                            if px >= x_min && px <= x_max {
                                mask[[iy, ix]] = if self.dark_field {
                                    Complex64::new(1.0, 0.0)
                                } else {
                                    fg
                                };
                            }
                        }
                    }
                }
                MaskFeature::Polygon { vertices } => {
                    for iy in 0..n {
                        let py = -half + (iy as f64 + 0.5) * pixel;
                        for ix in 0..n {
                            let px = -half + (ix as f64 + 0.5) * pixel;
                            if point_in_polygon(px, py, vertices) {
                                mask[[iy, ix]] = if self.dark_field {
                                    Complex64::new(1.0, 0.0)
                                } else {
                                    fg
                                };
                            }
                        }
                    }
                }
            }
        }

        mask
    }

    /// Compute the mask spectrum (2D FFT of transmittance).
    pub fn spectrum(&self, grid: &GridConfig, fft: &Fft2D) -> Array2<Complex64> {
        let mut transmittance = self.rasterize(grid);
        fft.forward(&mut transmittance);
        transmittance
    }

    fn absorber_transmittance(&self) -> Complex64 {
        match &self.mask_type {
            MaskType::Binary => Complex64::new(0.0, 0.0),
            MaskType::AttenuatedPSM {
                transmission,
                phase_deg,
            } => {
                let amp = transmission.sqrt();
                let phase = phase_deg.to_radians();
                Complex::from_polar(amp, phase)
            }
            MaskType::AlternatingPSM => Complex64::new(0.0, 0.0),
        }
    }
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(x: f64, y: f64, vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = vertices[i];
        let (xj, yj) = vertices[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Default for Mask {
    fn default() -> Self {
        Self::line_space(65.0, 180.0).expect("default mask parameters are valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_space_creates_features() {
        let mask = Mask::line_space(65.0, 180.0).unwrap();
        assert!(!mask.features.is_empty());
        assert!(!mask.dark_field);
    }

    #[test]
    fn test_rasterize_has_correct_size() {
        let mask = Mask::line_space(65.0, 180.0).unwrap();
        let grid = GridConfig::default();
        let raster = mask.rasterize(&grid);
        assert_eq!(raster.dim(), (512, 512));
    }

    #[test]
    fn test_binary_mask_values() {
        let mask = Mask::line_space(100.0, 200.0).unwrap();
        let grid = GridConfig {
            size: 256,
            pixel_nm: 2.0,
        };
        let raster = mask.rasterize(&grid);
        // All values should be 0 or 1
        for &v in raster.iter() {
            let norm = v.norm();
            assert!(
                (norm - 0.0).abs() < 1e-10 || (norm - 1.0).abs() < 1e-10,
                "Binary mask should only have 0 or 1 transmittance, got {}",
                norm
            );
        }
    }

    #[test]
    fn test_attenuated_psm_transmittance() {
        let mask = Mask {
            mask_type: MaskType::AttenuatedPSM {
                transmission: 0.06,
                phase_deg: 180.0,
            },
            features: vec![MaskFeature::Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 100.0,
            }],
            dark_field: false,
        };
        let grid = GridConfig {
            size: 64,
            pixel_nm: 4.0,
        };
        let raster = mask.rasterize(&grid);
        // Center pixel should be absorber
        let center = raster[[32, 32]];
        // Absorber has sqrt(0.06) amplitude with 180 degree phase
        let expected_amp = 0.06_f64.sqrt();
        assert!((center.norm() - expected_amp).abs() < 0.01);
    }

    #[test]
    fn test_contact_hole_dark_field() {
        let mask = Mask::contact_hole(80.0, 200.0, 200.0).unwrap();
        assert!(mask.dark_field);
        assert_eq!(mask.features.len(), 25); // 5x5 array
    }

    #[test]
    fn test_point_in_polygon() {
        let triangle = vec![(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)];
        assert!(point_in_polygon(2.0, 1.0, &triangle));
        assert!(!point_in_polygon(5.0, 5.0, &triangle));
    }

    #[test]
    fn test_invalid_cd_exceeds_pitch() {
        // CD >= pitch should return Err
        assert!(Mask::line_space(100.0, 50.0).is_err());
        assert!(Mask::line_space(100.0, 100.0).is_err());
    }

    #[test]
    fn test_invalid_negative_cd() {
        assert!(Mask::line_space(-10.0, 100.0).is_err());
        assert!(Mask::line_space(0.0, 100.0).is_err());
    }

    #[test]
    fn test_contact_hole_valid() {
        let mask = Mask::contact_hole(50.0, 150.0, 150.0);
        assert!(mask.is_ok());
        let mask = mask.unwrap();
        assert!(mask.dark_field);
        assert!(!mask.features.is_empty());
    }

    #[test]
    fn test_contact_hole_invalid_diameter() {
        // Diameter <= 0 should return Err
        assert!(Mask::contact_hole(0.0, 150.0, 150.0).is_err());
        assert!(Mask::contact_hole(-10.0, 150.0, 150.0).is_err());
        // Invalid pitch
        assert!(Mask::contact_hole(50.0, 0.0, 150.0).is_err());
        assert!(Mask::contact_hole(50.0, 150.0, -1.0).is_err());
    }
}
