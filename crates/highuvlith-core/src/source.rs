use serde::{Deserialize, Serialize};

/// Spatial coherence / illumination shape of the VUV source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IlluminationShape {
    /// Conventional circular partial coherence.
    Conventional { sigma: f64 },
    /// Annular illumination.
    Annular {
        sigma_inner: f64,
        sigma_outer: f64,
    },
    /// Quadrupole illumination.
    Quadrupole {
        sigma_center: f64,
        sigma_radius: f64,
        opening_angle_deg: f64,
    },
    /// Dipole illumination.
    Dipole {
        sigma_center: f64,
        sigma_radius: f64,
        orientation_deg: f64,
    },
}

/// Spectral line shape of the VUV laser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpectralShape {
    /// Lorentzian (typical for excimer lasers).
    Lorentzian,
    /// Gaussian.
    Gaussian,
    /// Tabulated measured spectrum.
    Tabulated {
        wavelengths_nm: Vec<f64>,
        intensities: Vec<f64>,
    },
}

impl Default for SpectralShape {
    fn default() -> Self {
        Self::Lorentzian
    }
}

/// VUV laser source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VuvSource {
    /// Center wavelength in nm (e.g., 157.63 for F2, 126.0 for Ar2).
    pub wavelength_nm: f64,
    /// Spectral bandwidth FWHM in pm (~1.1 pm for F2 laser).
    pub bandwidth_pm: f64,
    /// Number of spectral sampling points for polychromatic simulation.
    pub spectral_samples: usize,
    /// Spectral line shape.
    pub spectral_shape: SpectralShape,
    /// Pulse energy in mJ.
    pub pulse_energy_mj: f64,
    /// Repetition rate in Hz.
    pub rep_rate_hz: f64,
    /// Illumination pupil shape.
    pub illumination: IlluminationShape,
}

impl VuvSource {
    /// Create an F2 excimer laser source with default parameters.
    pub fn f2_laser(sigma: f64) -> Self {
        Self {
            wavelength_nm: 157.63,
            bandwidth_pm: 1.1,
            spectral_samples: 5,
            spectral_shape: SpectralShape::Lorentzian,
            pulse_energy_mj: 10.0,
            rep_rate_hz: 4000.0,
            illumination: IlluminationShape::Conventional { sigma },
        }
    }

    /// Create an Ar2 excimer laser source.
    pub fn ar2_laser(sigma: f64) -> Self {
        Self {
            wavelength_nm: 126.0,
            bandwidth_pm: 5.0,
            spectral_samples: 7,
            spectral_shape: SpectralShape::Lorentzian,
            pulse_energy_mj: 5.0,
            rep_rate_hz: 1000.0,
            illumination: IlluminationShape::Conventional { sigma },
        }
    }

    /// Evaluate the source intensity at a given pupil coordinate (fx, fy),
    /// normalized to the cutoff frequency NA/lambda.
    ///
    /// Returns the source intensity weight (0.0 if outside the source shape).
    pub fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64 {
        let rho = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();
        match &self.illumination {
            IlluminationShape::Conventional { sigma } => {
                if rho <= *sigma {
                    1.0
                } else {
                    0.0
                }
            }
            IlluminationShape::Annular {
                sigma_inner,
                sigma_outer,
            } => {
                if rho >= *sigma_inner && rho <= *sigma_outer {
                    1.0
                } else {
                    0.0
                }
            }
            IlluminationShape::Quadrupole {
                sigma_center,
                sigma_radius,
                opening_angle_deg,
            } => {
                let angle = fy_norm.atan2(fx_norm).to_degrees();
                let half_open = opening_angle_deg / 2.0;
                let in_pole = |center_angle: f64| -> bool {
                    let da = ((angle - center_angle + 180.0).rem_euclid(360.0)) - 180.0;
                    da.abs() <= half_open
                };
                let dist_to_center = |center_angle: f64| -> f64 {
                    let cx = sigma_center * center_angle.to_radians().cos();
                    let cy = sigma_center * center_angle.to_radians().sin();
                    ((fx_norm - cx).powi(2) + (fy_norm - cy).powi(2)).sqrt()
                };
                for &pole_angle in &[0.0, 90.0, 180.0, 270.0] {
                    if in_pole(pole_angle) && dist_to_center(pole_angle) <= *sigma_radius {
                        return 1.0;
                    }
                }
                0.0
            }
            IlluminationShape::Dipole {
                sigma_center,
                sigma_radius,
                orientation_deg,
            } => {
                let orient = orientation_deg.to_radians();
                for &sign in &[1.0_f64, -1.0] {
                    let cx = sigma_center * (orient + sign * std::f64::consts::PI).cos();
                    let cy = sigma_center * (orient + sign * std::f64::consts::PI).sin();
                    let dist = ((fx_norm - cx).powi(2) + (fy_norm - cy).powi(2)).sqrt();
                    if dist <= *sigma_radius {
                        return 1.0;
                    }
                }
                0.0
            }
        }
    }

    /// Generate spectral sampling points (wavelength_nm, weight) for polychromatic simulation.
    pub fn spectral_weights(&self) -> Vec<(f64, f64)> {
        if self.spectral_samples <= 1 {
            return vec![(self.wavelength_nm, 1.0)];
        }

        let bw_nm = self.bandwidth_pm * 1e-3;
        let half_range = 2.5 * bw_nm; // sample out to 2.5x FWHM
        let step = 2.0 * half_range / (self.spectral_samples - 1) as f64;

        let mut weights: Vec<(f64, f64)> = Vec::with_capacity(self.spectral_samples);
        let mut total = 0.0;

        for i in 0..self.spectral_samples {
            let wl = self.wavelength_nm - half_range + i as f64 * step;
            let dw = wl - self.wavelength_nm;
            let w = match &self.spectral_shape {
                SpectralShape::Lorentzian => {
                    let gamma = bw_nm / 2.0;
                    gamma * gamma / (dw * dw + gamma * gamma)
                }
                SpectralShape::Gaussian => {
                    let sigma = bw_nm / (2.0 * (2.0_f64.ln()).sqrt());
                    (-dw * dw / (2.0 * sigma * sigma)).exp()
                }
                SpectralShape::Tabulated {
                    wavelengths_nm,
                    intensities,
                } => {
                    // Linear interpolation in tabulated spectrum
                    interpolate_linear(wavelengths_nm, intensities, wl)
                }
            };
            total += w;
            weights.push((wl, w));
        }

        // Normalize weights
        if total > 0.0 {
            for (_, w) in &mut weights {
                *w /= total;
            }
        }

        weights
    }
}

fn interpolate_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let pos = xs.partition_point(|&v| v < x);
    if pos == 0 {
        return ys[0];
    }
    let t = (x - xs[pos - 1]) / (xs[pos] - xs[pos - 1]);
    ys[pos - 1] + t * (ys[pos] - ys[pos - 1])
}

impl Default for VuvSource {
    fn default() -> Self {
        Self::f2_laser(0.7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conventional_source_inside() {
        let src = VuvSource::f2_laser(0.5);
        assert_relative_eq!(src.intensity_at(0.0, 0.0), 1.0);
        assert_relative_eq!(src.intensity_at(0.3, 0.3), 1.0);
    }

    #[test]
    fn test_conventional_source_outside() {
        let src = VuvSource::f2_laser(0.5);
        assert_relative_eq!(src.intensity_at(0.6, 0.0), 0.0);
    }

    #[test]
    fn test_spectral_weights_sum_to_one() {
        let src = VuvSource::f2_laser(0.7);
        let weights = src.spectral_weights();
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_spectral_weights_centered() {
        let src = VuvSource::f2_laser(0.7);
        let weights = src.spectral_weights();
        // Center weight should be the largest
        let center_idx = weights.len() / 2;
        let center_w = weights[center_idx].1;
        for (i, (_, w)) in weights.iter().enumerate() {
            if i != center_idx {
                assert!(center_w >= *w);
            }
        }
    }
}
