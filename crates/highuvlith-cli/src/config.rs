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

#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    #[serde(default = "default_wavelength")]
    pub wavelength_nm: f64,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    #[serde(default = "default_bandwidth")]
    pub bandwidth_pm: f64,
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

fn default_wavelength() -> f64 { 157.63 }
fn default_sigma() -> f64 { 0.7 }
fn default_bandwidth() -> f64 { 1.1 }
fn default_na() -> f64 { 0.75 }
fn default_flare() -> f64 { 0.02 }
fn default_cd() -> f64 { 65.0 }
fn default_pitch() -> f64 { 180.0 }
fn default_grid_size() -> usize { 256 }
fn default_pixel() -> f64 { 1.0 }
fn default_dose() -> f64 { 30.0 }

impl Default for GridConfig {
    fn default() -> Self {
        Self { size: default_grid_size(), pixel_nm: default_pixel() }
    }
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self { dose_mj_cm2: default_dose(), focus_nm: 0.0 }
    }
}

impl SimConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn to_source(&self) -> highuvlith_core::source::VuvSource {
        highuvlith_core::source::VuvSource {
            wavelength_nm: self.source.wavelength_nm,
            bandwidth_pm: self.source.bandwidth_pm,
            spectral_samples: 5,
            spectral_shape: highuvlith_core::source::SpectralShape::Lorentzian,
            pulse_energy_mj: 10.0,
            rep_rate_hz: 4000.0,
            illumination: highuvlith_core::source::IlluminationShape::Conventional {
                sigma: self.source.sigma,
            },
        }
    }

    pub fn to_optics(&self) -> highuvlith_core::optics::ProjectionOptics {
        let mut optics = highuvlith_core::optics::ProjectionOptics::new(self.optics.na);
        optics.flare_fraction = self.optics.flare_fraction;
        optics
    }

    pub fn to_mask(&self) -> highuvlith_core::mask::Mask {
        highuvlith_core::mask::Mask::line_space(self.mask.cd_nm, self.mask.pitch_nm)
    }

    pub fn to_grid(&self) -> anyhow::Result<highuvlith_core::types::GridConfig> {
        highuvlith_core::types::GridConfig::new(self.grid.size, self.grid.pixel_nm)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}
