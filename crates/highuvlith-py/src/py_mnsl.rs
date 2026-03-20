use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyList;

use highuvlith_core::mnsl::{
    MnslConfig, MnslEngine, MnslResult, NanosphereArray, SpherePacking, SubstrateCoupling,
    simulate_moire_emission,
};
use highuvlith_core::thinfilm::FilmStack;
use highuvlith_core::types::GridConfig;

/// Python wrapper for SpherePacking enum.
#[pyclass]
#[derive(Clone)]
pub struct PySpherePacking {
    inner: SpherePacking,
}

#[pymethods]
impl PySpherePacking {
    #[classattr]
    const HCP: Self = Self { inner: SpherePacking::HCP };

    #[classattr]
    const FCC: Self = Self { inner: SpherePacking::FCC };

    #[classattr]
    const SIMPLE_CUBIC: Self = Self { inner: SpherePacking::SimpleCubic };

    fn __repr__(&self) -> String {
        match self.inner {
            SpherePacking::HCP => "SpherePacking.HCP".to_string(),
            SpherePacking::FCC => "SpherePacking.FCC".to_string(),
            SpherePacking::SimpleCubic => "SpherePacking.SIMPLE_CUBIC".to_string(),
        }
    }
}

/// Python wrapper for NanosphereArray configuration.
#[pyclass]
#[derive(Clone)]
pub struct PyNanosphereArrayConfig {
    inner: NanosphereArray,
}

#[pymethods]
impl PyNanosphereArrayConfig {
    #[new]
    #[pyo3(signature = (diameter_nm, pitch_nm, orientation_deg=0.0, n_real=1.56, n_imag=0.001, packing=None))]
    fn new(
        diameter_nm: f64,
        pitch_nm: f64,
        orientation_deg: f64,
        n_real: f64,
        n_imag: f64,
        packing: Option<PySpherePacking>,
    ) -> Self {
        let packing_inner = packing.map(|p| p.inner).unwrap_or(SpherePacking::HCP);
        Self {
            inner: NanosphereArray {
                diameter_nm,
                pitch_nm,
                orientation_deg,
                n_real,
                n_imag,
                packing: packing_inner,
            },
        }
    }

    /// Create silica nanosphere array with typical VUV optical constants.
    #[staticmethod]
    fn silica_spheres(diameter_nm: f64, pitch_nm: f64) -> Self {
        Self {
            inner: NanosphereArray::silica_spheres(diameter_nm, pitch_nm),
        }
    }

    /// Create polystyrene nanosphere array.
    #[staticmethod]
    fn polystyrene_spheres(diameter_nm: f64, pitch_nm: f64) -> Self {
        Self {
            inner: NanosphereArray::polystyrene_spheres(diameter_nm, pitch_nm),
        }
    }

    #[getter]
    fn diameter_nm(&self) -> f64 {
        self.inner.diameter_nm
    }

    #[setter]
    fn set_diameter_nm(&mut self, value: f64) {
        self.inner.diameter_nm = value;
    }

    #[getter]
    fn pitch_nm(&self) -> f64 {
        self.inner.pitch_nm
    }

    #[setter]
    fn set_pitch_nm(&mut self, value: f64) {
        self.inner.pitch_nm = value;
    }

    #[getter]
    fn orientation_deg(&self) -> f64 {
        self.inner.orientation_deg
    }

    #[setter]
    fn set_orientation_deg(&mut self, value: f64) {
        self.inner.orientation_deg = value;
    }

    #[getter]
    fn n_real(&self) -> f64 {
        self.inner.n_real
    }

    #[setter]
    fn set_n_real(&mut self, value: f64) {
        self.inner.n_real = value;
    }

    #[getter]
    fn n_imag(&self) -> f64 {
        self.inner.n_imag
    }

    #[setter]
    fn set_n_imag(&mut self, value: f64) {
        self.inner.n_imag = value;
    }

    /// Volume fraction of spheres in the array.
    fn volume_fraction(&self) -> f64 {
        self.inner.volume_fraction()
    }

    fn __repr__(&self) -> String {
        format!(
            "NanosphereArrayConfig(diameter={:.1}nm, pitch={:.1}nm, orientation={:.1}°)",
            self.inner.diameter_nm, self.inner.pitch_nm, self.inner.orientation_deg
        )
    }
}

/// Python wrapper for substrate coupling configuration.
#[pyclass]
#[derive(Clone)]
pub struct PySubstrateCoupling {
    inner: SubstrateCoupling,
}

#[pymethods]
impl PySubstrateCoupling {
    #[new]
    #[pyo3(signature = (coupling_strength=0.5, enable_nearfield=true))]
    fn new(coupling_strength: f64, enable_nearfield: bool) -> Self {
        Self {
            inner: SubstrateCoupling {
                substrate_stack: FilmStack::default(),
                coupling_strength,
                enable_nearfield,
            },
        }
    }

    #[getter]
    fn coupling_strength(&self) -> f64 {
        self.inner.coupling_strength
    }

    #[setter]
    fn set_coupling_strength(&mut self, value: f64) {
        self.inner.coupling_strength = value;
    }

    #[getter]
    fn enable_nearfield(&self) -> bool {
        self.inner.enable_nearfield
    }

    #[setter]
    fn set_enable_nearfield(&mut self, value: bool) {
        self.inner.enable_nearfield = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "SubstrateCoupling(strength={:.2}, nearfield={})",
            self.inner.coupling_strength, self.inner.enable_nearfield
        )
    }
}

/// Python wrapper for MNSL configuration.
#[pyclass]
#[derive(Clone)]
pub struct PyMnslConfig {
    inner: MnslConfig,
}

#[pymethods]
impl PyMnslConfig {
    #[new]
    #[pyo3(signature = (bottom_array, top_array, separation_nm=100.0, substrate=None, wavelength_nm=157.0))]
    fn new(
        bottom_array: PyNanosphereArrayConfig,
        top_array: PyNanosphereArrayConfig,
        separation_nm: f64,
        substrate: Option<PySubstrateCoupling>,
        wavelength_nm: f64,
    ) -> Self {
        let substrate_inner = substrate.map(|s| s.inner).unwrap_or_default();
        Self {
            inner: MnslConfig {
                bottom_array: bottom_array.inner,
                top_array: top_array.inner,
                separation_nm,
                substrate: substrate_inner,
                wavelength_nm,
            },
        }
    }

    #[getter]
    fn bottom_array(&self) -> PyNanosphereArrayConfig {
        PyNanosphereArrayConfig {
            inner: self.inner.bottom_array.clone(),
        }
    }

    #[setter]
    fn set_bottom_array(&mut self, array: PyNanosphereArrayConfig) {
        self.inner.bottom_array = array.inner;
    }

    #[getter]
    fn top_array(&self) -> PyNanosphereArrayConfig {
        PyNanosphereArrayConfig {
            inner: self.inner.top_array.clone(),
        }
    }

    #[setter]
    fn set_top_array(&mut self, array: PyNanosphereArrayConfig) {
        self.inner.top_array = array.inner;
    }

    #[getter]
    fn separation_nm(&self) -> f64 {
        self.inner.separation_nm
    }

    #[setter]
    fn set_separation_nm(&mut self, value: f64) {
        self.inner.separation_nm = value;
    }

    #[getter]
    fn wavelength_nm(&self) -> f64 {
        self.inner.wavelength_nm
    }

    #[setter]
    fn set_wavelength_nm(&mut self, value: f64) {
        self.inner.wavelength_nm = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "MnslConfig(separation={:.1}nm, wavelength={:.1}nm)",
            self.inner.separation_nm, self.inner.wavelength_nm
        )
    }
}

/// Python wrapper for MNSL simulation results.
#[pyclass]
pub struct PyMnslResult {
    inner: MnslResult,
}

#[pymethods]
impl PyMnslResult {
    /// Enhanced emission intensity pattern as 2D numpy array.
    #[getter]
    fn emission_pattern<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.inner.emission_pattern.data)
    }

    /// Underlying Moiré interference pattern as 2D numpy array.
    #[getter]
    fn moire_pattern<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.inner.moire_pattern.data)
    }

    /// Local field enhancement factors as 2D numpy array.
    #[getter]
    fn enhancement_factors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.inner.enhancement_factors.data)
    }

    /// Calculated Moiré period in nm.
    #[getter]
    fn moire_period_nm(&self) -> f64 {
        self.inner.moire_period_nm
    }

    /// Total integrated emission power (relative units).
    #[getter]
    fn total_emission_power(&self) -> f64 {
        self.inner.total_emission_power
    }

    /// Peak enhancement factor.
    #[getter]
    fn peak_enhancement(&self) -> f64 {
        self.inner.peak_enhancement
    }

    /// Peak positions as list of (x, y) coordinates in nm.
    #[getter]
    fn peak_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let peaks = PyList::empty(py);
        for &(x, y) in &self.inner.peak_positions {
            let coord = PyList::new(py, [x, y])?;
            peaks.append(coord)?;
        }
        Ok(peaks)
    }

    /// Get physical coordinates for the grids.
    fn coordinates(&self) -> (Vec<f64>, Vec<f64>) {
        let nx = self.inner.emission_pattern.nx();
        let ny = self.inner.emission_pattern.ny();

        let x_coords: Vec<f64> = (0..nx)
            .map(|i| self.inner.emission_pattern.x_at(i))
            .collect();
        let y_coords: Vec<f64> = (0..ny)
            .map(|i| self.inner.emission_pattern.y_at(i))
            .collect();

        (x_coords, y_coords)
    }

    /// Get 1D cross-section of emission pattern at specified y-coordinate.
    fn cross_section_x(&self, y_nm: f64) -> Vec<f64> {
        let ny = self.inner.emission_pattern.ny();
        let nx = self.inner.emission_pattern.nx();

        // Find closest y-index
        let row = (0..ny)
            .min_by_key(|&i| {
                let grid_y = self.inner.emission_pattern.y_at(i);
                ((grid_y - y_nm).abs() * 1000.0) as i64
            })
            .unwrap_or(ny / 2);

        (0..nx)
            .map(|j| self.inner.emission_pattern.data[[row, j]])
            .collect()
    }

    /// Get 1D cross-section of emission pattern at specified x-coordinate.
    fn cross_section_y(&self, x_nm: f64) -> Vec<f64> {
        let ny = self.inner.emission_pattern.ny();
        let nx = self.inner.emission_pattern.nx();

        // Find closest x-index
        let col = (0..nx)
            .min_by_key(|&j| {
                let grid_x = self.inner.emission_pattern.x_at(j);
                ((grid_x - x_nm).abs() * 1000.0) as i64
            })
            .unwrap_or(nx / 2);

        (0..ny)
            .map(|i| self.inner.emission_pattern.data[[i, col]])
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "MnslResult(period={:.1}nm, peak_enhancement={:.2}x, {} peaks)",
            self.inner.moire_period_nm, self.inner.peak_enhancement, self.inner.peak_positions.len()
        )
    }
}

/// Python wrapper for MNSL simulation engine.
#[pyclass]
pub struct PyMnslEngine {
    inner: MnslEngine,
}

#[pymethods]
impl PyMnslEngine {
    #[new]
    fn new(config: PyMnslConfig, grid_size: usize, pixel_nm: f64) -> PyResult<Self> {
        let grid = GridConfig::new(grid_size, pixel_nm)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

        Ok(Self {
            inner: MnslEngine::new(config.inner, grid),
        })
    }

    /// Compute the complete MNSL emission pattern.
    fn compute_emission(&self) -> PyMnslResult {
        let result = self.inner.compute_emission();
        PyMnslResult { inner: result }
    }
}

/// Quick MNSL simulation function.
#[pyfunction]
#[pyo3(signature = (sphere_diameter_nm, array_pitch_nm, rotation_angle_deg, separation_nm=100.0, grid_size=256, pixel_nm=2.0))]
fn py_simulate_moire_emission(
    sphere_diameter_nm: f64,
    array_pitch_nm: f64,
    rotation_angle_deg: f64,
    separation_nm: f64,
    grid_size: usize,
    pixel_nm: f64,
) -> PyResult<PyMnslResult> {
    let grid = GridConfig::new(grid_size, pixel_nm)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;

    let result = simulate_moire_emission(
        sphere_diameter_nm,
        array_pitch_nm,
        rotation_angle_deg,
        separation_nm,
        grid,
    );

    Ok(PyMnslResult { inner: result })
}

pub fn register_mnsl_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configuration classes
    m.add_class::<PySpherePacking>()?;
    m.add_class::<PyNanosphereArrayConfig>()?;
    m.add_class::<PySubstrateCoupling>()?;
    m.add_class::<PyMnslConfig>()?;

    // Simulation engine and results
    m.add_class::<PyMnslEngine>()?;
    m.add_class::<PyMnslResult>()?;

    // Convenience function
    m.add_function(wrap_pyfunction!(py_simulate_moire_emission, m)?)?;

    Ok(())
}