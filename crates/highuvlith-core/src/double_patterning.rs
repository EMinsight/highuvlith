//! Double patterning simulation.
//!
//! Simulates Litho-Etch-Litho-Etch (LELE) double patterning where two
//! exposure passes with different masks create features below the
//! single-exposure resolution limit.

use ndarray::Array2;

use crate::aerial::AerialImageEngine;
use crate::mask::Mask;
use crate::metrics;
use crate::types::Grid2D;

/// Double patterning configuration.
#[derive(Debug, Clone)]
pub struct DoublePatterningConfig {
    /// Overlay error in x (nm) between first and second exposure.
    pub overlay_x_nm: f64,
    /// Overlay error in y (nm).
    pub overlay_y_nm: f64,
    /// Dose for first exposure (mJ/cm²).
    pub dose1_mj_cm2: f64,
    /// Dose for second exposure (mJ/cm²).
    pub dose2_mj_cm2: f64,
    /// Focus for first exposure (nm).
    pub focus1_nm: f64,
    /// Focus for second exposure (nm).
    pub focus2_nm: f64,
}

impl Default for DoublePatterningConfig {
    fn default() -> Self {
        Self {
            overlay_x_nm: 0.0,
            overlay_y_nm: 0.0,
            dose1_mj_cm2: 30.0,
            dose2_mj_cm2: 30.0,
            focus1_nm: 0.0,
            focus2_nm: 0.0,
        }
    }
}

/// Result of double patterning simulation.
#[derive(Debug)]
pub struct DoublePatterningResult {
    /// Combined aerial image (sum of both exposures).
    pub combined_aerial: Grid2D<f64>,
    /// First exposure aerial image.
    pub aerial1: Grid2D<f64>,
    /// Second exposure aerial image.
    pub aerial2: Grid2D<f64>,
    /// Contrast of combined image.
    pub combined_contrast: f64,
}

/// Simulate LELE double patterning with two masks.
///
/// The combined latent image is the product of the individual exposures:
///   m_combined = m1 * m2
/// where m1 and m2 are the PAC concentrations from each exposure.
///
/// For aerial image analysis (before resist), we combine the dose-weighted
/// intensities: I_combined = dose1 * I1 + dose2 * I2(shifted).
pub fn simulate_double_patterning(
    engine: &AerialImageEngine,
    mask1: &Mask,
    mask2: &Mask,
    config: &DoublePatterningConfig,
) -> DoublePatterningResult {
    let aerial1 = engine.compute(mask1, config.focus1_nm);
    let aerial2 = engine.compute(mask2, config.focus2_nm);

    let grid = engine.grid();
    let pixel = grid.pixel_nm;

    // Apply overlay shift to second exposure
    let shift_x_pixels = (config.overlay_x_nm / pixel).round() as i64;
    let shift_y_pixels = (config.overlay_y_nm / pixel).round() as i64;

    let mut combined = Array2::zeros(aerial1.data.dim());
    let (ny, nx) = combined.dim();

    for i in 0..ny {
        for j in 0..nx {
            let i1 = aerial1.data[[i, j]] * config.dose1_mj_cm2;

            // Shifted second exposure
            let si = ((i as i64 - shift_y_pixels).rem_euclid(ny as i64)) as usize;
            let sj = ((j as i64 - shift_x_pixels).rem_euclid(nx as i64)) as usize;
            let i2 = aerial2.data[[si, sj]] * config.dose2_mj_cm2;

            combined[[i, j]] = i1 + i2;
        }
    }

    // Normalize
    let total_dose = config.dose1_mj_cm2 + config.dose2_mj_cm2;
    if total_dose > 0.0 {
        combined.mapv_inplace(|v| v / total_dose);
    }

    let combined_contrast = metrics::image_contrast(&combined);

    DoublePatterningResult {
        combined_aerial: Grid2D {
            data: combined,
            x_min_nm: aerial1.x_min_nm,
            x_max_nm: aerial1.x_max_nm,
            y_min_nm: aerial1.y_min_nm,
            y_max_nm: aerial1.y_max_nm,
        },
        aerial1,
        aerial2,
        combined_contrast,
    }
}

/// Create complementary mask pair for LELE double patterning.
///
/// Given a line/space pattern at half-pitch, creates two masks each with
/// double the pitch (every other line), that together form the original pattern.
pub fn split_mask_lele(cd_nm: f64, pitch_nm: f64) -> crate::error::Result<(Mask, Mask)> {
    // Each mask has lines at 2x the original pitch
    let double_pitch = 2.0 * pitch_nm;

    let mask1 = Mask::line_space(cd_nm, double_pitch)?;

    // Second mask is shifted by one pitch
    let mut mask2_features = Vec::new();
    let num_periods = 10;
    let space_width = double_pitch - cd_nm;
    let total_width = num_periods as f64 * double_pitch;
    let start = -total_width / 2.0;

    for i in 0..num_periods {
        let line_center_x = start + (i as f64 + 0.5) * double_pitch + pitch_nm; // shifted
        let space_center_x = line_center_x + double_pitch / 2.0;
        if i < num_periods - 1 {
            mask2_features.push(crate::mask::MaskFeature::Rect {
                x: space_center_x,
                y: 0.0,
                w: space_width,
                h: total_width,
            });
        }
    }

    let mask2 = Mask {
        mask_type: crate::mask::MaskType::Binary,
        features: mask2_features,
        dark_field: false,
    };

    Ok((mask1, mask2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optics::ProjectionOptics;
    use crate::source::VuvSource;
    use crate::types::GridConfig;

    #[test]
    fn test_double_patterning_basic() {
        let source = VuvSource::f2_laser(0.7).unwrap();
        let optics = ProjectionOptics::new(0.75).unwrap();
        let grid = GridConfig {
            size: 128,
            pixel_nm: 2.0,
        };
        let engine = AerialImageEngine::new(&source, &optics, grid, 10).unwrap();

        let (mask1, mask2) = split_mask_lele(65.0, 180.0).unwrap();
        let config = DoublePatterningConfig::default();
        let result = simulate_double_patterning(&engine, &mask1, &mask2, &config);

        assert!(result.combined_contrast > 0.0);
        assert!(result.combined_aerial.data.iter().all(|&v| v >= -1e-10));
    }

    #[test]
    fn test_overlay_shifts_result() {
        let source = VuvSource::f2_laser(0.7).unwrap();
        let optics = ProjectionOptics::new(0.75).unwrap();
        let grid = GridConfig {
            size: 128,
            pixel_nm: 2.0,
        };
        let engine = AerialImageEngine::new(&source, &optics, grid, 10).unwrap();

        let (mask1, mask2) = split_mask_lele(65.0, 180.0).unwrap();

        let config_zero = DoublePatterningConfig::default();
        let config_shifted = DoublePatterningConfig {
            overlay_x_nm: 10.0,
            ..Default::default()
        };

        let result_zero = simulate_double_patterning(&engine, &mask1, &mask2, &config_zero);
        let result_shifted = simulate_double_patterning(&engine, &mask1, &mask2, &config_shifted);

        // Overlay error should change the combined image
        let diff: f64 = result_zero
            .combined_aerial
            .data
            .iter()
            .zip(result_shifted.combined_aerial.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > 0.01,
            "Overlay shift should change the combined image"
        );
    }

    #[test]
    fn test_split_mask_creates_pair() {
        let (mask1, mask2) = split_mask_lele(65.0, 180.0).unwrap();
        assert!(!mask1.features.is_empty());
        assert!(!mask2.features.is_empty());
    }
}
