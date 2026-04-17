use pyo3::prelude::*;

use highuvlith_core::mask::Mask;
use highuvlith_core::optics::ProjectionOptics;
use highuvlith_core::resist::{DevelopmentModel, ResistParams};
use highuvlith_core::source::{
    IlluminationShape, LithographySource, LpaFelSource, SourceKind, SpectralShape, VuvSource,
};
use highuvlith_core::thinfilm::{FilmLayer, FilmStack};
use highuvlith_core::types::{Complex64, GridConfig};

/// Illumination source configuration.
///
/// Wraps any concrete `LithographySource` variant (VUV excimer, LPA-FEL).
/// Use the static factory methods (`f2_laser`, `ar2_laser`,
/// `lpa_fel_bella_25nm`, `lpa_fel`) for preset configurations, or the
/// default constructor for a custom VUV source.
#[pyclass(name = "SourceConfig")]
#[derive(Debug, Clone)]
pub struct PySourceConfig {
    pub inner: SourceKind,
}

#[pymethods]
impl PySourceConfig {
    #[new]
    #[pyo3(signature = (wavelength_nm=157.63, sigma_outer=0.7, bandwidth_pm=1.1, spectral_samples=5))]
    fn new(
        wavelength_nm: f64,
        sigma_outer: f64,
        bandwidth_pm: f64,
        spectral_samples: usize,
    ) -> PyResult<Self> {
        if wavelength_nm <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "wavelength_nm must be positive, got {}",
                wavelength_nm
            )));
        }
        if sigma_outer <= 0.0 || sigma_outer > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sigma_outer must be in (0, 1.0], got {}",
                sigma_outer
            )));
        }
        if bandwidth_pm < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bandwidth_pm must be >= 0, got {}",
                bandwidth_pm
            )));
        }
        if spectral_samples < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "spectral_samples must be >= 1, got {}",
                spectral_samples
            )));
        }
        Ok(Self {
            inner: SourceKind::Vuv(VuvSource {
                wavelength_nm,
                bandwidth_pm,
                spectral_samples,
                spectral_shape: SpectralShape::Lorentzian,
                pulse_energy_mj: 10.0,
                rep_rate_hz: 4000.0,
                illumination: IlluminationShape::Conventional { sigma: sigma_outer },
            }),
        })
    }

    /// Create F2 laser (157nm) source.
    #[staticmethod]
    #[pyo3(signature = (sigma=0.7))]
    fn f2_laser(sigma: f64) -> PyResult<Self> {
        let inner = VuvSource::f2_laser(sigma)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: SourceKind::Vuv(inner),
        })
    }

    /// Create Ar2 excimer (126nm) source.
    #[staticmethod]
    #[pyo3(signature = (sigma=0.7))]
    fn ar2_laser(sigma: f64) -> PyResult<Self> {
        let inner = VuvSource::ar2_laser(sigma)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: SourceKind::Vuv(inner),
        })
    }

    /// Create an LPA-FEL source with the BELLA 500 MeV target config (~25 nm).
    ///
    /// Models the projected performance of the laser-plasma driven FEL
    /// at LBNL BELLA (Kohrell et al., Phys. Rev. Accel. Beams, 2026)
    /// after the funded electron-beam upgrade to 500 MeV.
    #[staticmethod]
    #[pyo3(signature = (sigma=0.7))]
    fn lpa_fel_bella_25nm(sigma: f64) -> PyResult<Self> {
        let inner = LpaFelSource::bella_target_25nm(sigma)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: SourceKind::LpaFel(inner),
        })
    }

    /// Create a custom LPA-FEL source at the given wavelength (nm).
    /// Typically used for 20-30 nm EUV lithography studies; the
    /// remaining FEL-specific parameters default to values consistent
    /// with the BELLA architecture and can be overridden.
    #[staticmethod]
    #[pyo3(signature = (
        wavelength_nm,
        sigma=0.7,
        electron_energy_mev=500.0,
        bandwidth_pm=25.0,
        pulse_duration_fs=10.0,
        rep_rate_hz=1000.0,
    ))]
    fn lpa_fel(
        wavelength_nm: f64,
        sigma: f64,
        electron_energy_mev: f64,
        bandwidth_pm: f64,
        pulse_duration_fs: f64,
        rep_rate_hz: f64,
    ) -> PyResult<Self> {
        if bandwidth_pm < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "bandwidth_pm must be >= 0, got {}",
                bandwidth_pm
            )));
        }
        if electron_energy_mev <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "electron_energy_mev must be positive, got {}",
                electron_energy_mev
            )));
        }
        if pulse_duration_fs <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pulse_duration_fs must be positive, got {}",
                pulse_duration_fs
            )));
        }
        if rep_rate_hz <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "rep_rate_hz must be positive, got {}",
                rep_rate_hz
            )));
        }
        let mut inner = LpaFelSource::new(wavelength_nm, sigma)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        inner.electron_energy_mev = electron_energy_mev;
        inner.bandwidth_pm = bandwidth_pm;
        inner.pulse_duration_fs = pulse_duration_fs;
        inner.rep_rate_hz = rep_rate_hz;
        Ok(Self {
            inner: SourceKind::LpaFel(inner),
        })
    }

    #[getter]
    fn wavelength_nm(&self) -> f64 {
        self.inner.wavelength_nm()
    }

    #[getter]
    fn bandwidth_pm(&self) -> f64 {
        self.inner.bandwidth_pm()
    }

    #[getter]
    fn sigma_outer(&self) -> f64 {
        self.inner.sigma_outer().unwrap_or(0.0)
    }

    #[getter]
    fn spectral_samples(&self) -> usize {
        match &self.inner {
            SourceKind::Vuv(s) => s.spectral_samples,
            SourceKind::LpaFel(s) => s.spectral_samples,
        }
    }

    /// Short label identifying the source family ("vuv" or "lpa_fel").
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind_label()
    }

    /// Electron beam energy in MeV. Returns `None` for non-FEL sources.
    #[getter]
    fn electron_energy_mev(&self) -> Option<f64> {
        match &self.inner {
            SourceKind::LpaFel(s) => Some(s.electron_energy_mev),
            SourceKind::Vuv(_) => None,
        }
    }

    /// Pulse duration in femtoseconds. Returns `None` for non-FEL sources
    /// (excimer pulse durations are nanosecond-scale and not modeled here).
    #[getter]
    fn pulse_duration_fs(&self) -> Option<f64> {
        match &self.inner {
            SourceKind::LpaFel(s) => Some(s.pulse_duration_fs),
            SourceKind::Vuv(_) => None,
        }
    }

    /// Bunch/pulse repetition rate in Hz. Defined for both source types.
    #[getter]
    fn rep_rate_hz(&self) -> f64 {
        match &self.inner {
            SourceKind::Vuv(s) => s.rep_rate_hz,
            SourceKind::LpaFel(s) => s.rep_rate_hz,
        }
    }

    /// Transverse coherence fraction in [0, 1]. Returns `None` for non-FEL sources.
    #[getter]
    fn transverse_coherence_fraction(&self) -> Option<f64> {
        match &self.inner {
            SourceKind::LpaFel(s) => Some(s.transverse_coherence_fraction),
            SourceKind::Vuv(_) => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            SourceKind::Vuv(s) => format!(
                "SourceConfig(kind=vuv, wavelength_nm={}, illumination={:?})",
                s.wavelength_nm, s.illumination
            ),
            SourceKind::LpaFel(s) => format!(
                "SourceConfig(kind=lpa_fel, wavelength_nm={}, E_e={} MeV, tau={} fs)",
                s.wavelength_nm, s.electron_energy_mev, s.pulse_duration_fs
            ),
        }
    }
}

/// Projection optics configuration.
#[pyclass(name = "OpticsConfig")]
#[derive(Debug, Clone)]
pub struct PyOpticsConfig {
    pub inner: ProjectionOptics,
}

#[pymethods]
impl PyOpticsConfig {
    #[new]
    #[pyo3(signature = (numerical_aperture=0.75, reduction=4.0, flare_fraction=0.02))]
    fn new(numerical_aperture: f64, reduction: f64, flare_fraction: f64) -> PyResult<Self> {
        if numerical_aperture <= 0.0 || numerical_aperture > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "numerical_aperture must be in (0, 1.0], got {}",
                numerical_aperture
            )));
        }
        if reduction <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reduction must be positive, got {}",
                reduction
            )));
        }
        if !(0.0..=1.0).contains(&flare_fraction) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "flare_fraction must be in [0, 1.0], got {}",
                flare_fraction
            )));
        }
        let base = ProjectionOptics::new(numerical_aperture)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: ProjectionOptics {
                na: numerical_aperture,
                reduction,
                flare_fraction,
                ..base
            },
        })
    }

    /// Add a Zernike aberration coefficient.
    fn add_aberration(&mut self, fringe_index: usize, coefficient_waves: f64) {
        self.inner
            .zernike_coefficients
            .push((fringe_index, coefficient_waves));
    }

    #[getter]
    fn numerical_aperture(&self) -> f64 {
        self.inner.na
    }

    #[getter]
    fn reduction(&self) -> f64 {
        self.inner.reduction
    }

    #[getter]
    fn flare_fraction(&self) -> f64 {
        self.inner.flare_fraction
    }

    fn rayleigh_resolution(&self, wavelength_nm: f64) -> f64 {
        self.inner.rayleigh_resolution(wavelength_nm)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpticsConfig(NA={}, reduction={}x)",
            self.inner.na, self.inner.reduction
        )
    }
}

/// Mask configuration.
#[pyclass(name = "MaskConfig")]
#[derive(Debug, Clone)]
pub struct PyMaskConfig {
    pub inner: Mask,
}

#[pymethods]
impl PyMaskConfig {
    /// Create a line/space pattern.
    #[staticmethod]
    fn line_space(cd_nm: f64, pitch_nm: f64) -> PyResult<Self> {
        let inner = Mask::line_space(cd_nm, pitch_nm)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a contact hole array.
    #[staticmethod]
    fn contact_hole(diameter_nm: f64, pitch_x_nm: f64, pitch_y_nm: f64) -> PyResult<Self> {
        let inner = Mask::contact_hole(diameter_nm, pitch_x_nm, pitch_y_nm)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("MaskConfig({} features)", self.inner.features.len())
    }
}

/// Photoresist configuration.
#[pyclass(name = "ResistConfig")]
#[derive(Debug, Clone)]
pub struct PyResistConfig {
    pub inner: ResistParams,
}

#[pymethods]
impl PyResistConfig {
    #[new]
    #[pyo3(signature = (
        thickness_nm=150.0,
        dill_a=0.2,
        dill_b=0.45,
        dill_c=0.02,
        peb_diffusion_nm=30.0,
        model="mack"
    ))]
    fn new(
        thickness_nm: f64,
        dill_a: f64,
        dill_b: f64,
        dill_c: f64,
        peb_diffusion_nm: f64,
        model: &str,
    ) -> PyResult<Self> {
        if thickness_nm <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "thickness_nm must be positive, got {}",
                thickness_nm
            )));
        }
        if peb_diffusion_nm < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "peb_diffusion_nm must be >= 0, got {}",
                peb_diffusion_nm
            )));
        }
        let development = match model {
            "threshold" => DevelopmentModel::Threshold { threshold: 0.5 },
            "mack" => DevelopmentModel::default(),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown development model: {}. Use 'threshold' or 'mack'",
                    model
                )))
            }
        };

        Ok(Self {
            inner: ResistParams {
                thickness_nm,
                dill_a,
                dill_b,
                dill_c,
                peb_diffusion_nm,
                development,
            },
        })
    }

    /// Create VUV fluoropolymer resist with default parameters.
    #[staticmethod]
    fn vuv_fluoropolymer() -> Self {
        Self {
            inner: ResistParams::vuv_fluoropolymer(),
        }
    }

    #[getter]
    fn thickness_nm(&self) -> f64 {
        self.inner.thickness_nm
    }

    #[getter]
    fn dill_a(&self) -> f64 {
        self.inner.dill_a
    }

    #[getter]
    fn dill_b(&self) -> f64 {
        self.inner.dill_b
    }

    #[getter]
    fn dill_c(&self) -> f64 {
        self.inner.dill_c
    }

    #[getter]
    fn peb_diffusion_nm(&self) -> f64 {
        self.inner.peb_diffusion_nm
    }

    fn __repr__(&self) -> String {
        format!(
            "ResistConfig(thickness={}nm, A={}, B={}, C={})",
            self.inner.thickness_nm, self.inner.dill_a, self.inner.dill_b, self.inner.dill_c
        )
    }
}

/// Film stack configuration.
#[pyclass(name = "FilmStackConfig")]
#[derive(Debug, Clone)]
pub struct PyFilmStackConfig {
    pub inner: FilmStack,
}

#[pymethods]
impl PyFilmStackConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: FilmStack::default(),
        }
    }

    /// Add a layer to the stack.
    fn add_layer(&mut self, name: &str, thickness_nm: f64, n_real: f64, n_imag: f64) {
        self.inner.layers.push(FilmLayer {
            name: name.to_string(),
            thickness_nm,
            n: Complex64::new(n_real, n_imag),
        });
    }

    /// Set the substrate refractive index.
    fn set_substrate(&mut self, n_real: f64, n_imag: f64) {
        self.inner.substrate = Complex64::new(n_real, n_imag);
    }

    fn __repr__(&self) -> String {
        format!("FilmStackConfig({} layers)", self.inner.layers.len())
    }
}

/// Process parameters.
#[pyclass(name = "ProcessConfig")]
#[derive(Debug, Clone)]
pub struct PyProcessConfig {
    #[pyo3(get, set)]
    pub dose_mj_cm2: f64,
    #[pyo3(get, set)]
    pub focus_nm: f64,
    #[pyo3(get, set)]
    pub development_time_s: f64,
}

#[pymethods]
impl PyProcessConfig {
    #[new]
    #[pyo3(signature = (dose_mj_cm2=30.0, focus_nm=0.0, development_time_s=60.0))]
    fn new(dose_mj_cm2: f64, focus_nm: f64, development_time_s: f64) -> Self {
        Self {
            dose_mj_cm2,
            focus_nm,
            development_time_s,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessConfig(dose={}mJ/cm², focus={}nm)",
            self.dose_mj_cm2, self.focus_nm
        )
    }
}

/// Grid configuration.
#[pyclass(name = "GridConfig")]
#[derive(Debug, Clone)]
pub struct PyGridConfig {
    pub inner: GridConfig,
}

#[pymethods]
impl PyGridConfig {
    #[new]
    #[pyo3(signature = (size=512, pixel_nm=1.0))]
    fn new(size: usize, pixel_nm: f64) -> PyResult<Self> {
        let inner = GridConfig::new(size, pixel_nm)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[getter]
    fn pixel_nm(&self) -> f64 {
        self.inner.pixel_nm
    }

    fn field_size_nm(&self) -> f64 {
        self.inner.field_size_nm()
    }

    fn __repr__(&self) -> String {
        format!(
            "GridConfig({}x{}, pixel={}nm, field={}nm)",
            self.inner.size,
            self.inner.size,
            self.inner.pixel_nm,
            self.inner.field_size_nm()
        )
    }
}
