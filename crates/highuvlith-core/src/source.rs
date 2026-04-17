use serde::{Deserialize, Serialize};

/// Trait for any illumination source.
pub trait LithographySource: Send + Sync {
    /// Center wavelength in nm.
    fn wavelength_nm(&self) -> f64;

    /// Photon energy in eV.
    fn photon_energy_ev(&self) -> f64 {
        1239.84193 / self.wavelength_nm()
    }

    /// Spectral bandwidth FWHM in pm.
    fn bandwidth_pm(&self) -> f64;

    /// Source intensity at normalized pupil coordinate.
    fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64;

    /// Spectral sampling weights for polychromatic simulation.
    fn spectral_weights(&self) -> Vec<(f64, f64)>;

    /// Photon density at dose=1 mJ/cm^2 (photons/nm^2).
    /// Computed from wavelength: higher energy photons = fewer photons per unit dose.
    fn photon_density_per_mj_cm2(&self) -> f64 {
        let e_photon_j = 6.62607015e-34 * 2.99792458e8 / (self.wavelength_nm() * 1e-9);
        // 1 mJ/cm^2 = 10 J/m^2; convert to photons/nm^2
        10.0 / e_photon_j * 1e-18
    }
}

/// Spatial coherence / illumination shape of the VUV source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IlluminationShape {
    /// Conventional circular partial coherence.
    Conventional { sigma: f64 },
    /// Annular illumination.
    Annular { sigma_inner: f64, sigma_outer: f64 },
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum SpectralShape {
    /// Lorentzian (typical for excimer lasers).
    #[default]
    Lorentzian,
    /// Gaussian.
    Gaussian,
    /// Tabulated measured spectrum.
    Tabulated {
        wavelengths_nm: Vec<f64>,
        intensities: Vec<f64>,
    },
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

/// Validate sigma for conventional illumination. Shared across source types.
fn validate_sigma(sigma: f64) -> crate::error::Result<()> {
    if sigma.is_nan() || sigma <= 0.0 || sigma > 1.0 {
        return Err(crate::error::LithographyError::InvalidParameter {
            name: "sigma",
            value: if sigma.is_nan() { f64::NAN } else { sigma },
            reason: "must be in range (0, 1]",
        });
    }
    Ok(())
}

impl VuvSource {
    /// Create an F2 excimer laser source with default parameters.
    pub fn f2_laser(sigma: f64) -> crate::error::Result<Self> {
        validate_sigma(sigma)?;
        Ok(Self {
            wavelength_nm: 157.63,
            bandwidth_pm: 1.1,
            spectral_samples: 5,
            spectral_shape: SpectralShape::Lorentzian,
            pulse_energy_mj: 10.0,
            rep_rate_hz: 4000.0,
            illumination: IlluminationShape::Conventional { sigma },
        })
    }

    /// Create an Ar2 excimer laser source.
    pub fn ar2_laser(sigma: f64) -> crate::error::Result<Self> {
        validate_sigma(sigma)?;
        Ok(Self {
            wavelength_nm: 126.0,
            bandwidth_pm: 5.0,
            spectral_samples: 7,
            spectral_shape: SpectralShape::Lorentzian,
            pulse_energy_mj: 5.0,
            rep_rate_hz: 1000.0,
            illumination: IlluminationShape::Conventional { sigma },
        })
    }

    /// Evaluate the source intensity at a given pupil coordinate (fx, fy),
    /// normalized to the cutoff frequency NA/lambda.
    ///
    /// Returns the source intensity weight (0.0 if outside the source shape).
    pub fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64 {
        evaluate_illumination(&self.illumination, fx_norm, fy_norm)
    }

    /// Generate spectral sampling points (wavelength_nm, weight) for polychromatic simulation.
    pub fn spectral_weights(&self) -> Vec<(f64, f64)> {
        evaluate_spectral_weights(
            self.wavelength_nm,
            self.bandwidth_pm,
            self.spectral_samples,
            &self.spectral_shape,
        )
    }
}

impl LithographySource for VuvSource {
    fn wavelength_nm(&self) -> f64 {
        self.wavelength_nm
    }

    fn bandwidth_pm(&self) -> f64 {
        self.bandwidth_pm
    }

    fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64 {
        // Delegate to inherent method via UFCS
        VuvSource::intensity_at(self, fx_norm, fy_norm)
    }

    fn spectral_weights(&self) -> Vec<(f64, f64)> {
        VuvSource::spectral_weights(self)
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
        Self::f2_laser(0.7).expect("default sigma 0.7 is valid")
    }
}

/// Laser-plasma driven free electron laser (LPA-FEL) source.
///
/// Models the compact LPA-FEL architecture demonstrated at LBNL BELLA
/// (Kohrell et al., Phys. Rev. Accel. Beams, 2026): laser wakefield
/// accelerator driving an undulator to produce coherent, narrow-band
/// radiation. The initial demo at 100 MeV reached 420 nm with 1 kHz
/// bunch rate and 8+ hours of feedback-stabilized operation. The
/// planned 500 MeV upgrade targets 20-30 nm — the regime this struct
/// is primarily intended to model for EUV lithography studies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpaFelSource {
    /// Center wavelength in nm (target 20-30 nm range at 500 MeV).
    pub wavelength_nm: f64,
    /// Spectral bandwidth FWHM in pm. FEL output is much narrower
    /// than excimer; SASE ~0.1 % relative, seeded/self-seeded tighter.
    pub bandwidth_pm: f64,
    /// Electron beam kinetic energy in MeV (100 MeV baseline, 500 MeV target).
    pub electron_energy_mev: f64,
    /// Bunch repetition rate in Hz (1000 Hz for BELLA architecture).
    pub rep_rate_hz: f64,
    /// Pulse energy in µJ (FEL regime, distinct from mJ-class excimers).
    pub pulse_energy_uj: f64,
    /// Pulse duration in femtoseconds (FEL output is ultrashort).
    pub pulse_duration_fs: f64,
    /// Shot-to-shot intensity jitter as a fraction (e.g. 0.03 = 3 %).
    /// Reflects feedback-stabilized stability demonstrated at BELLA.
    pub shot_to_shot_stability: f64,
    /// Transverse coherence fraction in [0, 1]. FEL is high-coherence
    /// relative to excimer; values >0.8 are typical for seeded modes.
    pub transverse_coherence_fraction: f64,
    /// Number of spectral sampling points for polychromatic simulation.
    pub spectral_samples: usize,
    /// Spectral line shape (Gaussian is a good default for seeded FEL).
    pub spectral_shape: SpectralShape,
    /// Illumination pupil shape.
    pub illumination: IlluminationShape,
}

impl LpaFelSource {
    /// BELLA baseline: ~420 nm at 100 MeV (current demonstrated config).
    /// Retained as a reference fixture — this wavelength is NOT useful
    /// for EUV lithography; use `bella_target_25nm` for that regime.
    pub fn bella_baseline_100mev() -> crate::error::Result<Self> {
        validate_sigma(0.7)?;
        Ok(Self {
            wavelength_nm: 420.0,
            bandwidth_pm: 420.0,
            electron_energy_mev: 100.0,
            rep_rate_hz: 1000.0,
            pulse_energy_uj: 1.0,
            pulse_duration_fs: 10.0,
            shot_to_shot_stability: 0.05,
            transverse_coherence_fraction: 0.85,
            spectral_samples: 5,
            spectral_shape: SpectralShape::Gaussian,
            illumination: IlluminationShape::Conventional { sigma: 0.7 },
        })
    }

    /// BELLA 500 MeV target config: ~25 nm EUV for lithography.
    /// Represents the projected performance after the funded electron
    /// beam upgrade to 500 MeV.
    pub fn bella_target_25nm(sigma: f64) -> crate::error::Result<Self> {
        validate_sigma(sigma)?;
        Ok(Self {
            wavelength_nm: 25.0,
            bandwidth_pm: 25.0,
            electron_energy_mev: 500.0,
            rep_rate_hz: 1000.0,
            pulse_energy_uj: 5.0,
            pulse_duration_fs: 10.0,
            shot_to_shot_stability: 0.03,
            transverse_coherence_fraction: 0.9,
            spectral_samples: 5,
            spectral_shape: SpectralShape::Gaussian,
            illumination: IlluminationShape::Conventional { sigma },
        })
    }

    /// Generic LPA-FEL constructor. Wavelength must be positive and
    /// typically in the 20-30 nm range for EUV lithography studies;
    /// values outside this are accepted but flagged via validation only.
    pub fn new(wavelength_nm: f64, sigma: f64) -> crate::error::Result<Self> {
        if wavelength_nm <= 0.0 || wavelength_nm.is_nan() {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "wavelength_nm",
                value: if wavelength_nm.is_nan() {
                    f64::NAN
                } else {
                    wavelength_nm
                },
                reason: "must be > 0",
            });
        }
        validate_sigma(sigma)?;
        Ok(Self {
            wavelength_nm,
            bandwidth_pm: wavelength_nm * 1e-3,
            electron_energy_mev: 500.0,
            rep_rate_hz: 1000.0,
            pulse_energy_uj: 5.0,
            pulse_duration_fs: 10.0,
            shot_to_shot_stability: 0.03,
            transverse_coherence_fraction: 0.9,
            spectral_samples: 5,
            spectral_shape: SpectralShape::Gaussian,
            illumination: IlluminationShape::Conventional { sigma },
        })
    }

    /// Pupil illumination intensity (shared logic with VuvSource).
    pub fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64 {
        evaluate_illumination(&self.illumination, fx_norm, fy_norm)
    }

    /// Polychromatic spectral sampling weights.
    pub fn spectral_weights(&self) -> Vec<(f64, f64)> {
        evaluate_spectral_weights(
            self.wavelength_nm,
            self.bandwidth_pm,
            self.spectral_samples,
            &self.spectral_shape,
        )
    }
}

impl LithographySource for LpaFelSource {
    fn wavelength_nm(&self) -> f64 {
        self.wavelength_nm
    }

    fn bandwidth_pm(&self) -> f64 {
        self.bandwidth_pm
    }

    fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64 {
        LpaFelSource::intensity_at(self, fx_norm, fy_norm)
    }

    fn spectral_weights(&self) -> Vec<(f64, f64)> {
        LpaFelSource::spectral_weights(self)
    }
}

impl Default for LpaFelSource {
    fn default() -> Self {
        Self::bella_target_25nm(0.7).expect("default sigma 0.7 is valid")
    }
}

/// Evaluate illumination pupil intensity. Shared by VuvSource and LpaFelSource.
fn evaluate_illumination(shape: &IlluminationShape, fx_norm: f64, fy_norm: f64) -> f64 {
    let rho = (fx_norm * fx_norm + fy_norm * fy_norm).sqrt();
    match shape {
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

/// Evaluate spectral sampling weights. Shared by VuvSource and LpaFelSource.
fn evaluate_spectral_weights(
    wavelength_nm: f64,
    bandwidth_pm: f64,
    spectral_samples: usize,
    spectral_shape: &SpectralShape,
) -> Vec<(f64, f64)> {
    if spectral_samples <= 1 {
        return vec![(wavelength_nm, 1.0)];
    }

    let bw_nm = bandwidth_pm * 1e-3;
    let half_range = 2.5 * bw_nm;
    let step = 2.0 * half_range / (spectral_samples - 1) as f64;

    let mut weights: Vec<(f64, f64)> = Vec::with_capacity(spectral_samples);
    let mut total = 0.0;

    for i in 0..spectral_samples {
        let wl = wavelength_nm - half_range + i as f64 * step;
        let dw = wl - wavelength_nm;
        let w = match spectral_shape {
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
            } => interpolate_linear(wavelengths_nm, intensities, wl),
        };
        total += w;
        weights.push((wl, w));
    }

    if total > 0.0 {
        for (_, w) in &mut weights {
            *w /= total;
        }
    }

    weights
}

/// Type-erased source wrapper. Dispatches trait methods to the held
/// concrete variant. Used by the PyO3 bindings and CLI config to
/// carry either a VUV excimer or an LPA-FEL source through APIs
/// that only need the `LithographySource` interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceKind {
    Vuv(VuvSource),
    LpaFel(LpaFelSource),
}

impl SourceKind {
    /// Outer sigma of the illumination pupil, if it is a conventional
    /// circular shape. Returns `None` for annular/quadrupole/dipole.
    pub fn sigma_outer(&self) -> Option<f64> {
        let illum = match self {
            SourceKind::Vuv(s) => &s.illumination,
            SourceKind::LpaFel(s) => &s.illumination,
        };
        match illum {
            IlluminationShape::Conventional { sigma } => Some(*sigma),
            IlluminationShape::Annular { sigma_outer, .. } => Some(*sigma_outer),
            _ => None,
        }
    }

    /// Short label identifying the source family (for display/logging).
    pub fn kind_label(&self) -> &'static str {
        match self {
            SourceKind::Vuv(_) => "vuv",
            SourceKind::LpaFel(_) => "lpa_fel",
        }
    }
}

impl LithographySource for SourceKind {
    fn wavelength_nm(&self) -> f64 {
        match self {
            SourceKind::Vuv(s) => s.wavelength_nm(),
            SourceKind::LpaFel(s) => s.wavelength_nm(),
        }
    }

    fn bandwidth_pm(&self) -> f64 {
        match self {
            SourceKind::Vuv(s) => s.bandwidth_pm(),
            SourceKind::LpaFel(s) => s.bandwidth_pm(),
        }
    }

    fn intensity_at(&self, fx_norm: f64, fy_norm: f64) -> f64 {
        match self {
            SourceKind::Vuv(s) => s.intensity_at(fx_norm, fy_norm),
            SourceKind::LpaFel(s) => s.intensity_at(fx_norm, fy_norm),
        }
    }

    fn spectral_weights(&self) -> Vec<(f64, f64)> {
        match self {
            SourceKind::Vuv(s) => s.spectral_weights(),
            SourceKind::LpaFel(s) => s.spectral_weights(),
        }
    }
}

impl Default for SourceKind {
    fn default() -> Self {
        SourceKind::Vuv(VuvSource::default())
    }
}

impl From<VuvSource> for SourceKind {
    fn from(s: VuvSource) -> Self {
        SourceKind::Vuv(s)
    }
}

impl From<LpaFelSource> for SourceKind {
    fn from(s: LpaFelSource) -> Self {
        SourceKind::LpaFel(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conventional_source_inside() {
        let src = VuvSource::f2_laser(0.5).unwrap();
        assert_relative_eq!(src.intensity_at(0.0, 0.0), 1.0);
        assert_relative_eq!(src.intensity_at(0.3, 0.3), 1.0);
    }

    #[test]
    fn test_conventional_source_outside() {
        let src = VuvSource::f2_laser(0.5).unwrap();
        assert_relative_eq!(src.intensity_at(0.6, 0.0), 0.0);
    }

    #[test]
    fn test_spectral_weights_sum_to_one() {
        let src = VuvSource::f2_laser(0.7).unwrap();
        let weights = src.spectral_weights();
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_spectral_weights_centered() {
        let src = VuvSource::f2_laser(0.7).unwrap();
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

    #[test]
    fn test_f2_laser_wavelength() {
        let src = VuvSource::f2_laser(0.5).unwrap();
        assert_relative_eq!(src.wavelength_nm, 157.63, epsilon = 0.01);
    }

    #[test]
    fn test_ar2_laser_wavelength() {
        let src = VuvSource::ar2_laser(0.5).unwrap();
        assert_relative_eq!(src.wavelength_nm, 126.0, epsilon = 0.1);
    }

    #[test]
    fn test_photon_energy_ev() {
        let src = VuvSource::f2_laser(0.5).unwrap();
        let energy = src.photon_energy_ev();
        // hc/lambda = 1239.84193 / 157.63 ~ 7.866 eV
        assert_relative_eq!(energy, 1239.84193 / 157.63, epsilon = 0.01);
        assert!(
            energy > 7.8 && energy < 8.0,
            "F2 photon energy should be ~7.9 eV, got {}",
            energy
        );
    }

    #[test]
    fn test_invalid_sigma_rejected() {
        assert!(VuvSource::f2_laser(0.0).is_err());
        assert!(VuvSource::f2_laser(-1.0).is_err());
        assert!(VuvSource::f2_laser(f64::NAN).is_err());
    }

    #[test]
    fn test_lpa_fel_wavelength_in_target_range() {
        let src = LpaFelSource::bella_target_25nm(0.7).unwrap();
        assert!(
            (20.0..=30.0).contains(&src.wavelength_nm),
            "bella_target_25nm should be in 20-30 nm range, got {}",
            src.wavelength_nm
        );
    }

    #[test]
    fn test_lpa_fel_photon_energy_at_25nm() {
        let src = LpaFelSource::bella_target_25nm(0.7).unwrap();
        let energy = src.photon_energy_ev();
        assert_relative_eq!(energy, 1239.84193 / 25.0, epsilon = 0.01);
        assert!(
            (49.0..=50.0).contains(&energy),
            "25 nm photon energy should be ~49.6 eV, got {}",
            energy
        );
    }

    #[test]
    fn test_lpa_fel_spectral_weights_sum_to_one() {
        let src = LpaFelSource::bella_target_25nm(0.7).unwrap();
        let weights = src.spectral_weights();
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lpa_fel_narrow_bandwidth_finite() {
        // Seeded FEL regime: very tight bandwidth. Guards against the
        // normalization blowing up or producing NaN weights.
        let src = LpaFelSource {
            wavelength_nm: 25.0,
            bandwidth_pm: 0.01,
            electron_energy_mev: 500.0,
            rep_rate_hz: 1000.0,
            pulse_energy_uj: 5.0,
            pulse_duration_fs: 10.0,
            shot_to_shot_stability: 0.02,
            transverse_coherence_fraction: 0.95,
            spectral_samples: 7,
            spectral_shape: SpectralShape::Gaussian,
            illumination: IlluminationShape::Conventional { sigma: 0.7 },
        };
        let weights = src.spectral_weights();
        assert_eq!(weights.len(), 7);
        for (wl, w) in &weights {
            assert!(wl.is_finite(), "wavelength not finite: {}", wl);
            assert!(w.is_finite(), "weight not finite: {}", w);
            assert!(*w >= 0.0, "weight negative: {}", w);
        }
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lpa_fel_invalid_sigma_rejected() {
        assert!(LpaFelSource::bella_target_25nm(0.0).is_err());
        assert!(LpaFelSource::bella_target_25nm(-0.5).is_err());
        assert!(LpaFelSource::bella_target_25nm(f64::NAN).is_err());
        assert!(LpaFelSource::new(25.0, 1.5).is_err());
        assert!(LpaFelSource::new(-10.0, 0.7).is_err());
        assert!(LpaFelSource::new(f64::NAN, 0.7).is_err());
    }

    #[test]
    fn test_source_kind_trait_dispatch() {
        let vuv: SourceKind = VuvSource::f2_laser(0.7).unwrap().into();
        let fel: SourceKind = LpaFelSource::bella_target_25nm(0.7).unwrap().into();

        assert_relative_eq!(vuv.wavelength_nm(), 157.63, epsilon = 0.01);
        assert_relative_eq!(fel.wavelength_nm(), 25.0, epsilon = 0.01);

        // Both dispatch pupil evaluation through the trait
        assert_relative_eq!(vuv.intensity_at(0.0, 0.0), 1.0);
        assert_relative_eq!(fel.intensity_at(0.0, 0.0), 1.0);
        assert_relative_eq!(fel.intensity_at(0.9, 0.0), 0.0);

        // Spectral weights normalize to 1 through the trait
        let fel_sum: f64 = fel.spectral_weights().iter().map(|(_, w)| w).sum();
        assert_relative_eq!(fel_sum, 1.0, epsilon = 1e-12);

        // Labels and sigma_outer helpers
        assert_eq!(vuv.kind_label(), "vuv");
        assert_eq!(fel.kind_label(), "lpa_fel");
        assert_eq!(vuv.sigma_outer(), Some(0.7));
        assert_eq!(fel.sigma_outer(), Some(0.7));
    }

    #[test]
    fn test_source_kind_toml_roundtrip() {
        let fel: SourceKind = LpaFelSource::bella_target_25nm(0.6).unwrap().into();
        let toml_str = toml::to_string(&fel).unwrap();
        assert!(
            toml_str.contains("type = \"lpa_fel\""),
            "serialized TOML missing tag: {}",
            toml_str
        );
        let parsed: SourceKind = toml::from_str(&toml_str).unwrap();
        match parsed {
            SourceKind::LpaFel(s) => {
                assert_relative_eq!(s.wavelength_nm, 25.0, epsilon = 0.01);
                assert_relative_eq!(s.electron_energy_mev, 500.0, epsilon = 0.01);
            }
            _ => panic!("expected LpaFel variant"),
        }

        let vuv: SourceKind = VuvSource::f2_laser(0.7).unwrap().into();
        let vuv_toml = toml::to_string(&vuv).unwrap();
        assert!(
            vuv_toml.contains("type = \"vuv\""),
            "VUV TOML missing tag: {}",
            vuv_toml
        );
        let reparsed: SourceKind = toml::from_str(&vuv_toml).unwrap();
        assert!(matches!(reparsed, SourceKind::Vuv(_)));
    }
}
