use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub source: SourceConfig,
    pub optics: OpticsConfig,
    pub mask: MaskConfig,
    #[serde(default)]
    pub grid: GridConfig,
    #[serde(default)]
    pub process: ProcessConfig,
}

/// Source configuration section of the TOML.
///
/// Back-compatible: if `type` is absent, it defaults to `"vuv"`, so
/// existing configs that specify only `wavelength_nm`, `sigma`, and
/// `bandwidth_pm` continue to work unchanged. Set `type = "lpa_fel"`
/// to select the laser-plasma driven FEL source; the LPA-FEL-specific
/// fields (`electron_energy_mev`, `pulse_duration_fs`, `rep_rate_hz`)
/// are optional and default to BELLA-target values.
#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    #[serde(rename = "type", default = "default_source_type")]
    pub source_type: String,
    #[serde(default = "default_wavelength")]
    pub wavelength_nm: f64,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    #[serde(default = "default_bandwidth")]
    pub bandwidth_pm: f64,
    // LPA-FEL-specific fields. All optional; consulted only when
    // `source_type == "lpa_fel"`.
    pub electron_energy_mev: Option<f64>,
    pub pulse_duration_fs: Option<f64>,
    pub rep_rate_hz: Option<f64>,
    pub pulse_energy_uj: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct OpticsConfig {
    #[serde(default = "default_na")]
    pub na: f64,
    #[serde(default = "default_flare")]
    pub flare_fraction: f64,
}

#[derive(Debug, Deserialize)]
pub struct MaskConfig {
    #[serde(default = "default_cd")]
    pub cd_nm: f64,
    #[serde(default = "default_pitch")]
    pub pitch_nm: f64,
}

#[derive(Debug, Deserialize)]
pub struct GridConfig {
    #[serde(default = "default_grid_size")]
    pub size: usize,
    #[serde(default = "default_pixel")]
    pub pixel_nm: f64,
}

#[derive(Debug, Deserialize)]
pub struct ProcessConfig {
    #[serde(default = "default_dose")]
    pub dose_mj_cm2: f64,
    #[serde(default)]
    pub focus_nm: f64,
}

fn default_source_type() -> String {
    "vuv".to_string()
}
fn default_wavelength() -> f64 {
    157.63
}
fn default_sigma() -> f64 {
    0.7
}
fn default_bandwidth() -> f64 {
    1.1
}
fn default_na() -> f64 {
    0.75
}
fn default_flare() -> f64 {
    0.02
}
fn default_cd() -> f64 {
    65.0
}
fn default_pitch() -> f64 {
    180.0
}
fn default_grid_size() -> usize {
    256
}
fn default_pixel() -> f64 {
    1.0
}
fn default_dose() -> f64 {
    30.0
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            size: default_grid_size(),
            pixel_nm: default_pixel(),
        }
    }
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            dose_mj_cm2: default_dose(),
            focus_nm: 0.0,
        }
    }
}

impl SimConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Validate all configuration values for physical and computational correctness.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.source.wavelength_nm <= 0.0 {
            anyhow::bail!(
                "wavelength_nm must be > 0, got {}",
                self.source.wavelength_nm
            );
        }
        if self.optics.na <= 0.0 || self.optics.na > 1.0 {
            anyhow::bail!("NA must be in (0, 1.0], got {}", self.optics.na);
        }
        if self.source.sigma <= 0.0 || self.source.sigma > 1.0 {
            anyhow::bail!("sigma must be in (0, 1.0], got {}", self.source.sigma);
        }
        match self.source.source_type.as_str() {
            "vuv" | "lpa_fel" => {}
            other => anyhow::bail!(
                "unknown source type '{}' (expected 'vuv' or 'lpa_fel')",
                other
            ),
        }
        if self.mask.cd_nm <= 0.0 {
            anyhow::bail!("cd_nm must be > 0, got {}", self.mask.cd_nm);
        }
        if self.mask.pitch_nm <= 0.0 {
            anyhow::bail!("pitch_nm must be > 0, got {}", self.mask.pitch_nm);
        }
        if self.mask.cd_nm >= self.mask.pitch_nm {
            anyhow::bail!(
                "cd_nm ({}) must be < pitch_nm ({})",
                self.mask.cd_nm,
                self.mask.pitch_nm
            );
        }
        if self.grid.size == 0 || (self.grid.size & (self.grid.size - 1)) != 0 {
            anyhow::bail!("grid size must be a power of 2, got {}", self.grid.size);
        }
        if self.grid.pixel_nm <= 0.0 {
            anyhow::bail!("pixel_nm must be > 0, got {}", self.grid.pixel_nm);
        }
        Ok(())
    }

    pub fn to_source(&self) -> highuvlith_core::source::SourceKind {
        use highuvlith_core::source::{
            IlluminationShape, LpaFelSource, SourceKind, SpectralShape, VuvSource,
        };
        match self.source.source_type.as_str() {
            "lpa_fel" => {
                let mut fel = LpaFelSource::new(self.source.wavelength_nm, self.source.sigma)
                    .unwrap_or_else(|_| {
                        LpaFelSource::bella_target_25nm(self.source.sigma.clamp(0.05, 1.0))
                            .expect("clamped sigma is valid")
                    });
                fel.bandwidth_pm = self.source.bandwidth_pm.max(0.0);
                if let Some(e) = self.source.electron_energy_mev {
                    fel.electron_energy_mev = e;
                }
                if let Some(tau) = self.source.pulse_duration_fs {
                    fel.pulse_duration_fs = tau;
                }
                if let Some(rr) = self.source.rep_rate_hz {
                    fel.rep_rate_hz = rr;
                }
                if let Some(pe) = self.source.pulse_energy_uj {
                    fel.pulse_energy_uj = pe;
                }
                SourceKind::LpaFel(fel)
            }
            _ => SourceKind::Vuv(VuvSource {
                wavelength_nm: self.source.wavelength_nm,
                bandwidth_pm: self.source.bandwidth_pm,
                spectral_samples: 5,
                spectral_shape: SpectralShape::Lorentzian,
                pulse_energy_mj: 10.0,
                rep_rate_hz: self.source.rep_rate_hz.unwrap_or(4000.0),
                illumination: IlluminationShape::Conventional {
                    sigma: self.source.sigma,
                },
            }),
        }
    }

    pub fn to_optics(&self) -> anyhow::Result<highuvlith_core::optics::ProjectionOptics> {
        let mut optics = highuvlith_core::optics::ProjectionOptics::new(self.optics.na)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        optics.flare_fraction = self.optics.flare_fraction;
        Ok(optics)
    }

    pub fn to_mask(&self) -> anyhow::Result<highuvlith_core::mask::Mask> {
        highuvlith_core::mask::Mask::line_space(self.mask.cd_nm, self.mask.pitch_nm)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn to_grid(&self) -> anyhow::Result<highuvlith_core::types::GridConfig> {
        highuvlith_core::types::GridConfig::new(self.grid.size, self.grid.pixel_nm)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> SimConfig {
        SimConfig {
            source: SourceConfig {
                source_type: "vuv".to_string(),
                wavelength_nm: 157.63,
                sigma: 0.7,
                bandwidth_pm: 1.1,
                electron_energy_mev: None,
                pulse_duration_fs: None,
                rep_rate_hz: None,
                pulse_energy_uj: None,
            },
            optics: OpticsConfig {
                na: 0.75,
                flare_fraction: 0.02,
            },
            mask: MaskConfig {
                cd_nm: 65.0,
                pitch_nm: 180.0,
            },
            grid: GridConfig {
                size: 256,
                pixel_nm: 1.0,
            },
            process: ProcessConfig {
                dose_mj_cm2: 30.0,
                focus_nm: 0.0,
            },
        }
    }

    #[test]
    fn test_valid_config_passes() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn test_zero_wavelength() {
        let mut cfg = valid_config();
        cfg.source.wavelength_nm = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_na_out_of_range() {
        let mut cfg = valid_config();
        cfg.optics.na = 1.5;
        assert!(cfg.validate().is_err());

        cfg.optics.na = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sigma_out_of_range() {
        let mut cfg = valid_config();
        cfg.source.sigma = 0.0;
        assert!(cfg.validate().is_err());

        cfg.source.sigma = 1.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cd_gte_pitch() {
        let mut cfg = valid_config();
        cfg.mask.cd_nm = 200.0;
        cfg.mask.pitch_nm = 180.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_grid_not_power_of_two() {
        let mut cfg = valid_config();
        cfg.grid.size = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_zero_pixel() {
        let mut cfg = valid_config();
        cfg.grid.pixel_nm = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_unknown_source_type_rejected() {
        let mut cfg = valid_config();
        cfg.source.source_type = "synchrotron".to_string();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lpa_fel_source_type_accepted() {
        let mut cfg = valid_config();
        cfg.source.source_type = "lpa_fel".to_string();
        cfg.source.wavelength_nm = 25.0;
        cfg.source.bandwidth_pm = 25.0;
        cfg.source.electron_energy_mev = Some(500.0);
        cfg.validate().unwrap();
        let src = cfg.to_source();
        assert!(matches!(
            src,
            highuvlith_core::source::SourceKind::LpaFel(_)
        ));
    }

    #[test]
    fn test_toml_back_compat_without_type_tag() {
        let toml_str = r#"
            [source]
            wavelength_nm = 157.63
            sigma = 0.7
            bandwidth_pm = 1.1

            [optics]
            na = 0.75

            [mask]
            cd_nm = 65.0
            pitch_nm = 180.0
        "#;
        let cfg: SimConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.source.source_type, "vuv");
        cfg.validate().unwrap();
        assert!(matches!(
            cfg.to_source(),
            highuvlith_core::source::SourceKind::Vuv(_)
        ));
    }

    #[test]
    fn test_toml_lpa_fel_parsing() {
        let toml_str = r#"
            [source]
            type = "lpa_fel"
            wavelength_nm = 25.0
            sigma = 0.7
            bandwidth_pm = 25.0
            electron_energy_mev = 500.0
            pulse_duration_fs = 10.0
            rep_rate_hz = 1000.0

            [optics]
            na = 0.55

            [mask]
            cd_nm = 30.0
            pitch_nm = 100.0
        "#;
        let cfg: SimConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.source.source_type, "lpa_fel");
        assert_eq!(cfg.source.electron_energy_mev, Some(500.0));
        cfg.validate().unwrap();
        match cfg.to_source() {
            highuvlith_core::source::SourceKind::LpaFel(s) => {
                assert!((s.wavelength_nm - 25.0).abs() < 1e-9);
                assert!((s.electron_energy_mev - 500.0).abs() < 1e-9);
                assert!((s.pulse_duration_fs - 10.0).abs() < 1e-9);
            }
            _ => panic!("expected LpaFel variant"),
        }
    }
}
