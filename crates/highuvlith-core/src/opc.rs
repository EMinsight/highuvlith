use crate::aerial::AerialImageEngine;
use crate::mask::{Mask, MaskFeature};
use crate::metrics;
use serde::{Deserialize, Serialize};

/// OPC rule: bias to apply to features of a given width range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcRule {
    /// Minimum feature width (nm) for this rule.
    pub min_width_nm: f64,
    /// Maximum feature width (nm) for this rule.
    pub max_width_nm: f64,
    /// Bias to add to each edge (nm). Positive = grow feature.
    pub bias_nm: f64,
}

/// Table of OPC rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcRuleTable {
    pub rules: Vec<OpcRule>,
}

impl OpcRuleTable {
    /// Apply rule-based OPC to a mask.
    /// Returns a new mask with biased features.
    pub fn apply(&self, mask: &Mask) -> Mask {
        let new_features: Vec<MaskFeature> = mask
            .features
            .iter()
            .map(|feature| self.bias_feature(feature))
            .collect();

        Mask {
            mask_type: mask.mask_type.clone(),
            features: new_features,
            dark_field: mask.dark_field,
        }
    }

    fn bias_feature(&self, feature: &MaskFeature) -> MaskFeature {
        match feature {
            MaskFeature::Rect { x, y, w, h } => {
                let bias = self.find_bias(*w);
                MaskFeature::Rect {
                    x: *x,
                    y: *y,
                    w: w + 2.0 * bias, // bias applied to both edges
                    h: *h,
                }
            }
            MaskFeature::Polygon { vertices } => {
                // For polygons, apply uniform edge bias (simplified)
                MaskFeature::Polygon {
                    vertices: vertices.clone(),
                }
            }
        }
    }

    fn find_bias(&self, width_nm: f64) -> f64 {
        for rule in &self.rules {
            if width_nm >= rule.min_width_nm && width_nm <= rule.max_width_nm {
                return rule.bias_nm;
            }
        }
        0.0 // no matching rule
    }
}

/// Model-based OPC: iteratively adjust mask edges to match target CD.
///
/// Uses the aerial image engine to simulate the mask at each iteration
/// and adjusts feature edges based on the error between measured and target CD.
pub fn model_based_opc(
    mask: &Mask,
    engine: &AerialImageEngine,
    target_cd_nm: f64,
    cd_threshold: f64,
    max_iterations: usize,
    convergence_tol_nm: f64,
) -> crate::error::Result<(Mask, OpcConvergence)> {
    let mut current_mask = mask.clone();
    let mut convergence = OpcConvergence {
        iterations: Vec::new(),
    };

    let grid = engine.grid();
    let field = grid.field_size_nm();
    let half = field / 2.0;

    for iter in 0..max_iterations {
        // Simulate current mask
        let aerial = engine.compute(&current_mask, 0.0);
        let measured_cd =
            metrics::measure_cd_2d(&aerial.data, -half, half, cd_threshold)
                .unwrap_or(0.0);

        let error = measured_cd - target_cd_nm;

        convergence.iterations.push(OpcIteration {
            iteration: iter,
            cd_nm: measured_cd,
            error_nm: error,
        });

        // Check convergence
        if error.abs() < convergence_tol_nm {
            return Ok((current_mask, convergence));
        }

        // Adjust mask features: simple proportional correction
        let correction = -error * 0.5; // damped correction factor
        current_mask = bias_mask_features(&current_mask, correction);
    }

    let last_error = convergence
        .iterations
        .last()
        .map(|it| it.error_nm.abs())
        .unwrap_or(f64::INFINITY);

    if last_error > convergence_tol_nm {
        Err(crate::error::LithographyError::ConvergenceFailure {
            iterations: max_iterations,
            residual: last_error,
        })
    } else {
        Ok((current_mask, convergence))
    }
}

/// Apply a uniform bias to all rectangular features in a mask.
fn bias_mask_features(mask: &Mask, bias_nm: f64) -> Mask {
    let new_features: Vec<MaskFeature> = mask
        .features
        .iter()
        .map(|feature| match feature {
            MaskFeature::Rect { x, y, w, h } => MaskFeature::Rect {
                x: *x,
                y: *y,
                w: (w + 2.0 * bias_nm).max(1.0),
                h: *h,
            },
            other => other.clone(),
        })
        .collect();

    Mask {
        mask_type: mask.mask_type.clone(),
        features: new_features,
        dark_field: mask.dark_field,
    }
}

/// OPC convergence history.
#[derive(Debug, Clone)]
pub struct OpcConvergence {
    pub iterations: Vec<OpcIteration>,
}

/// Single OPC iteration result.
#[derive(Debug, Clone)]
pub struct OpcIteration {
    pub iteration: usize,
    pub cd_nm: f64,
    pub error_nm: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::MaskType;

    #[test]
    fn test_rule_based_opc_applies_bias() {
        let rules = OpcRuleTable {
            rules: vec![OpcRule {
                min_width_nm: 50.0,
                max_width_nm: 100.0,
                bias_nm: 5.0,
            }],
        };

        let mask = Mask {
            mask_type: MaskType::Binary,
            features: vec![MaskFeature::Rect {
                x: 0.0,
                y: 0.0,
                w: 65.0,
                h: 500.0,
            }],
            dark_field: false,
        };

        let biased = rules.apply(&mask);
        match &biased.features[0] {
            MaskFeature::Rect { w, .. } => {
                assert!((w - 75.0).abs() < 1e-10, "Expected 65 + 2*5 = 75, got {}", w);
            }
            _ => panic!("Expected Rect"),
        }
    }

    #[test]
    fn test_rule_no_match_zero_bias() {
        let rules = OpcRuleTable {
            rules: vec![OpcRule {
                min_width_nm: 50.0,
                max_width_nm: 100.0,
                bias_nm: 5.0,
            }],
        };

        let mask = Mask {
            mask_type: MaskType::Binary,
            features: vec![MaskFeature::Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0, // outside rule range
                h: 500.0,
            }],
            dark_field: false,
        };

        let biased = rules.apply(&mask);
        match &biased.features[0] {
            MaskFeature::Rect { w, .. } => {
                assert!((w - 200.0).abs() < 1e-10, "No rule should match");
            }
            _ => panic!("Expected Rect"),
        }
    }

    #[test]
    fn test_bias_mask_positive() {
        let mask = Mask {
            mask_type: MaskType::Binary,
            features: vec![MaskFeature::Rect {
                x: 0.0,
                y: 0.0,
                w: 65.0,
                h: 500.0,
            }],
            dark_field: false,
        };

        let biased = bias_mask_features(&mask, 3.0);
        match &biased.features[0] {
            MaskFeature::Rect { w, .. } => {
                assert!((w - 71.0).abs() < 1e-10);
            }
            _ => panic!("Expected Rect"),
        }
    }
}
