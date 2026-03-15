use pyo3::prelude::*;

use highuvlith_core::mask::Mask;
use highuvlith_core::optics::ProjectionOptics;
use highuvlith_core::resist::{DevelopmentModel, ResistParams};
use highuvlith_core::source::{IlluminationShape, SpectralShape, VuvSource};
use highuvlith_core::thinfilm::{FilmLayer, FilmStack};
use highuvlith_core::types::{Complex64, GridConfig};

/// VUV laser source configuration.
#[pyclass(name = "SourceConfig")]
#[derive(Debug, Clone)]
pub struct PySourceConfig {
    pub inner: VuvSource,
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
    ) -> Self {
        Self {
            inner: VuvSource {
                wavelength_nm,
                bandwidth_pm,
                spectral_samples,
                spectral_shape: SpectralShape::Lorentzian,
                pulse_energy_mj: 10.0,
                rep_rate_hz: 4000.0,
                illumination: IlluminationShape::Conventional { sigma: sigma_outer },
            },
        }
    }

    /// Create F2 laser (157nm) source.
    #[staticmethod]
    #[pyo3(signature = (sigma=0.7))]
    fn f2_laser(sigma: f64) -> Self {
        Self {
            inner: VuvSource::f2_laser(sigma),
        }
    }

    /// Create Ar2 excimer (126nm) source.
    #[staticmethod]
    #[pyo3(signature = (sigma=0.7))]
    fn ar2_laser(sigma: f64) -> Self {
        Self {
            inner: VuvSource::ar2_laser(sigma),
        }
    }

    #[getter]
    fn wavelength_nm(&self) -> f64 {
        self.inner.wavelength_nm
    }

    #[getter]
    fn bandwidth_pm(&self) -> f64 {
        self.inner.bandwidth_pm
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceConfig(wavelength_nm={}, sigma={:?})",
            self.inner.wavelength_nm, self.inner.illumination
        )
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
    fn new(numerical_aperture: f64, reduction: f64, flare_fraction: f64) -> Self {
        Self {
            inner: ProjectionOptics {
                na: numerical_aperture,
                reduction,
                flare_fraction,
                ..ProjectionOptics::new(numerical_aperture)
            },
        }
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
    fn line_space(cd_nm: f64, pitch_nm: f64) -> Self {
        Self {
            inner: Mask::line_space(cd_nm, pitch_nm),
        }
    }

    /// Create a contact hole array.
    #[staticmethod]
    fn contact_hole(diameter_nm: f64, pitch_x_nm: f64, pitch_y_nm: f64) -> Self {
        Self {
            inner: Mask::contact_hole(diameter_nm, pitch_x_nm, pitch_y_nm),
        }
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
        let inner = GridConfig::new(size, pixel_nm).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })?;
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
