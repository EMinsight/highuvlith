use ndarray::Array2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::aerial::AerialImageEngine;
use crate::error::LithographyError;
use crate::mask::Mask;
use crate::metrics;

/// A single point on a Bossung curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BossungPoint {
    pub dose_mj_cm2: f64,
    pub focus_nm: f64,
    pub cd_nm: f64,
}

/// Process window analysis result.
#[derive(Debug, Clone)]
pub struct ProcessWindow {
    /// Dose values used.
    pub doses: Vec<f64>,
    /// Focus values used.
    pub focuses: Vec<f64>,
    /// CD at each (dose_index, focus_index).
    pub cd_matrix: Array2<f64>,
    /// Target CD in nm.
    pub cd_target_nm: f64,
    /// CD tolerance in percent.
    pub cd_tolerance_pct: f64,
}

impl ProcessWindow {
    /// Compute the process window by sweeping dose and focus.
    pub fn compute(
        engine: &AerialImageEngine,
        mask: &Mask,
        doses: &[f64],
        focuses: &[f64],
        cd_threshold: f64,
        cd_target_nm: f64,
        cd_tolerance_pct: f64,
    ) -> crate::error::Result<Self> {
        let n_doses = doses.len();
        let n_focuses = focuses.len();

        // Parallel computation over all dose-focus combinations
        let grid = engine.grid();
        let field = grid.field_size_nm();
        let half = field / 2.0;

        let cd_values: Vec<f64> = doses
            .iter()
            .flat_map(|&_dose| {
                focuses.iter().map(move |&focus| {
                    let aerial = engine.compute(mask, focus);
                    // Measure CD from the center cross-section
                    metrics::measure_cd_2d(&aerial.data, -half, half, cd_threshold).unwrap_or(0.0)
                })
            })
            .collect();

        let cd_matrix = Array2::from_shape_vec((n_doses, n_focuses), cd_values).map_err(|e| {
            LithographyError::InternalError(format!("CD matrix shape error: {}", e))
        })?;

        Ok(Self {
            doses: doses.to_vec(),
            focuses: focuses.to_vec(),
            cd_matrix,
            cd_target_nm,
            cd_tolerance_pct,
        })
    }

    /// Compute depth of focus (DOF) at the target CD.
    /// DOF is the focus range over which CD stays within tolerance.
    pub fn depth_of_focus(&self) -> f64 {
        let cd_min = self.cd_target_nm * (1.0 - self.cd_tolerance_pct / 100.0);
        let cd_max = self.cd_target_nm * (1.0 + self.cd_tolerance_pct / 100.0);

        // Find the dose row closest to target CD at best focus
        let best_dose_idx = self.best_dose_index();

        let mut min_focus = f64::INFINITY;
        let mut max_focus = f64::NEG_INFINITY;

        for (j, &focus) in self.focuses.iter().enumerate() {
            let cd = self.cd_matrix[[best_dose_idx, j]];
            if cd >= cd_min && cd <= cd_max {
                min_focus = min_focus.min(focus);
                max_focus = max_focus.max(focus);
            }
        }

        if max_focus > min_focus {
            max_focus - min_focus
        } else {
            0.0
        }
    }

    /// Compute exposure latitude (EL) at best focus.
    /// EL is the dose range (as percentage) over which CD stays within tolerance.
    pub fn exposure_latitude(&self) -> f64 {
        let cd_min = self.cd_target_nm * (1.0 - self.cd_tolerance_pct / 100.0);
        let cd_max = self.cd_target_nm * (1.0 + self.cd_tolerance_pct / 100.0);

        let best_focus_idx = self.best_focus_index();

        let mut min_dose = f64::INFINITY;
        let mut max_dose = f64::NEG_INFINITY;

        for (i, &dose) in self.doses.iter().enumerate() {
            let cd = self.cd_matrix[[i, best_focus_idx]];
            if cd >= cd_min && cd <= cd_max {
                min_dose = min_dose.min(dose);
                max_dose = max_dose.max(dose);
            }
        }

        if max_dose > min_dose {
            let center_dose = (min_dose + max_dose) / 2.0;
            (max_dose - min_dose) / center_dose * 100.0
        } else {
            0.0
        }
    }

    /// Extract Bossung curves (CD vs focus at each dose).
    pub fn bossung_curves(&self) -> Vec<Vec<BossungPoint>> {
        self.doses
            .iter()
            .enumerate()
            .map(|(i, &dose)| {
                self.focuses
                    .iter()
                    .enumerate()
                    .map(|(j, &focus)| BossungPoint {
                        dose_mj_cm2: dose,
                        focus_nm: focus,
                        cd_nm: self.cd_matrix[[i, j]],
                    })
                    .collect()
            })
            .collect()
    }

    /// Find the dose index that gives CD closest to target at best focus.
    fn best_dose_index(&self) -> usize {
        let best_focus = self.best_focus_index();
        let mut best_idx = 0;
        let mut best_err = f64::INFINITY;

        for (i, _) in self.doses.iter().enumerate() {
            let err = (self.cd_matrix[[i, best_focus]] - self.cd_target_nm).abs();
            if err < best_err {
                best_err = err;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Find the focus index closest to zero (best focus).
    fn best_focus_index(&self) -> usize {
        self.focuses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Batch simulation: compute aerial images for multiple defocus values.
/// Returns Vec of (focus_nm, aerial_image_data).
pub fn batch_defocus(
    engine: &AerialImageEngine,
    mask: &Mask,
    focuses: &[f64],
) -> Vec<(f64, Array2<f64>)> {
    #[cfg(feature = "parallel")]
    let result = focuses
        .par_iter()
        .map(|&focus| {
            let image = engine.compute(mask, focus);
            (focus, image.data)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let result = focuses
        .iter()
        .map(|&focus| {
            let image = engine.compute(mask, focus);
            (focus, image.data)
        })
        .collect();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bossung_curves_shape() {
        let pw = ProcessWindow {
            doses: vec![25.0, 30.0, 35.0],
            focuses: vec![-100.0, 0.0, 100.0],
            cd_matrix: Array2::from_shape_vec(
                (3, 3),
                vec![60.0, 65.0, 62.0, 58.0, 63.0, 60.0, 55.0, 60.0, 57.0],
            )
            .unwrap(),
            cd_target_nm: 63.0,
            cd_tolerance_pct: 10.0,
        };

        let curves = pw.bossung_curves();
        assert_eq!(curves.len(), 3); // one per dose
        assert_eq!(curves[0].len(), 3); // one per focus
    }

    #[test]
    fn test_dof_positive_for_valid_pw() {
        // target=65, tol=10% -> range [58.5, 71.5]
        let pw = ProcessWindow {
            doses: vec![25.0, 30.0, 35.0],
            focuses: vec![-200.0, -100.0, 0.0, 100.0, 200.0],
            cd_matrix: Array2::from_shape_vec(
                (3, 5),
                vec![
                    50.0, 60.0, 65.0, 60.0, 50.0, // dose 25: 60 is in [58.5,71.5]
                    55.0, 63.0, 67.0, 63.0, 55.0, // dose 30
                    58.0, 66.0, 70.0, 66.0, 58.0, // dose 35
                ],
            )
            .unwrap(),
            cd_target_nm: 65.0,
            cd_tolerance_pct: 10.0,
        };

        let dof = pw.depth_of_focus();
        assert!(
            dof > 0.0,
            "DOF should be positive for this process window, got {}",
            dof
        );
    }

    #[test]
    fn test_el_positive_for_valid_pw() {
        // target=65, tol=10% -> range [58.5, 71.5]
        let pw = ProcessWindow {
            doses: vec![20.0, 25.0, 30.0, 35.0, 40.0],
            focuses: vec![-100.0, 0.0, 100.0],
            cd_matrix: Array2::from_shape_vec(
                (5, 3),
                vec![
                    55.0, 60.0, 55.0, // dose 20: 60 in range
                    58.0, 63.0, 58.0, // dose 25: 63 in range
                    60.0, 65.0, 60.0, // dose 30: 65 in range, 60 in range
                    62.0, 68.0, 62.0, // dose 35: 68 in range, 62 in range
                    64.0, 71.0, 64.0, // dose 40: 71 in range, 64 in range
                ],
            )
            .unwrap(),
            cd_target_nm: 65.0,
            cd_tolerance_pct: 10.0,
        };

        let el = pw.exposure_latitude();
        assert!(
            el > 0.0,
            "EL should be positive for this process window, got {}",
            el
        );
    }

    #[test]
    fn test_batch_defocus_returns_correct_count() {
        use crate::aerial::AerialImageEngine;
        use crate::optics::ProjectionOptics;
        use crate::source::VuvSource;
        use crate::types::GridConfig;

        let source = VuvSource::f2_laser(0.5).unwrap();
        let optics = ProjectionOptics::new(0.75).unwrap();
        let grid = GridConfig {
            size: 64,
            pixel_nm: 4.0,
        };
        let engine = AerialImageEngine::new(&source, &optics, grid, 10).unwrap();
        let mask = Mask::line_space(65.0, 180.0).unwrap();

        let focuses = vec![-200.0, -100.0, 0.0, 100.0, 200.0];
        let results = batch_defocus(&engine, &mask, &focuses);
        assert_eq!(results.len(), 5);
        for (focus, image) in &results {
            assert!(focuses.contains(focus));
            assert_eq!(image.dim(), (64, 64));
        }
    }

    #[test]
    fn test_process_window_compute() {
        use crate::aerial::AerialImageEngine;
        use crate::optics::ProjectionOptics;
        use crate::source::VuvSource;
        use crate::types::GridConfig;

        let source = VuvSource::f2_laser(0.5).unwrap();
        let optics = ProjectionOptics::new(0.75).unwrap();
        let grid = GridConfig {
            size: 64,
            pixel_nm: 4.0,
        };
        let engine = AerialImageEngine::new(&source, &optics, grid, 10).unwrap();
        let mask = Mask::line_space(65.0, 180.0).unwrap();

        let doses = vec![25.0, 30.0, 35.0];
        let focuses = vec![-100.0, 0.0, 100.0];
        let pw = ProcessWindow::compute(&engine, &mask, &doses, &focuses, 0.5, 65.0, 10.0).unwrap();

        assert_eq!(pw.cd_matrix.dim(), (3, 3));
        assert_eq!(pw.doses.len(), 3);
        assert_eq!(pw.focuses.len(), 3);
    }
}
