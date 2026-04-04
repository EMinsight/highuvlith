use numpy::ToPyArray;
use pyo3::prelude::*;

use highuvlith_core::aerial::AerialImageEngine;
use highuvlith_core::process::ProcessWindow;
use highuvlith_core::types::GridConfig;

use crate::py_config::*;

/// Batch simulation with GIL release for parameter sweeps.
#[pyclass(name = "BatchSimulator")]
pub struct PyBatchSimulator {
    engine: AerialImageEngine,
    mask: highuvlith_core::mask::Mask,
    _grid: GridConfig,
}

#[pymethods]
impl PyBatchSimulator {
    #[new]
    #[pyo3(signature = (source, optics, mask, grid=None, max_kernels=30))]
    fn new(
        source: PySourceConfig,
        optics: PyOpticsConfig,
        mask: PyMaskConfig,
        grid: Option<PyGridConfig>,
        max_kernels: usize,
    ) -> PyResult<Self> {
        let grid_config = grid.map(|g| g.inner).unwrap_or_default();

        let engine = AerialImageEngine::new(
            &source.inner,
            &optics.inner,
            grid_config.clone(),
            max_kernels,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            engine,
            mask: mask.inner,
            _grid: grid_config,
        })
    }

    /// Compute process window: CD vs dose and focus.
    /// Returns (doses, focuses, cd_matrix) as numpy arrays.
    #[pyo3(signature = (
        doses,
        focuses,
        cd_threshold=0.3,
        cd_target_nm=65.0,
        cd_tolerance_pct=10.0
    ))]
    fn process_window<'py>(
        &self,
        py: Python<'py>,
        doses: Vec<f64>,
        focuses: Vec<f64>,
        cd_threshold: f64,
        cd_target_nm: f64,
        cd_tolerance_pct: f64,
    ) -> PyResult<PyProcessWindowResult> {
        let pw = py
            .allow_threads(|| {
                ProcessWindow::compute(
                    &self.engine,
                    &self.mask,
                    &doses,
                    &focuses,
                    cd_threshold,
                    cd_target_nm,
                    cd_tolerance_pct,
                )
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyProcessWindowResult { inner: pw })
    }

    /// Batch compute aerial images at multiple focus values.
    /// Returns list of (focus_nm, intensity_array) tuples.
    fn batch_defocus<'py>(
        &self,
        py: Python<'py>,
        focuses: Vec<f64>,
    ) -> PyResult<Vec<(f64, Bound<'py, numpy::PyArray2<f64>>)>> {
        let results = py.allow_threads(|| {
            highuvlith_core::process::batch_defocus(&self.engine, &self.mask, &focuses)
        });

        Ok(results
            .into_iter()
            .map(|(f, arr)| (f, arr.to_pyarray(py)))
            .collect())
    }
}

/// Process window analysis result.
#[pyclass(name = "ProcessWindowResult")]
pub struct PyProcessWindowResult {
    inner: ProcessWindow,
}

#[pymethods]
impl PyProcessWindowResult {
    /// Depth of focus in nm.
    fn depth_of_focus(&self) -> f64 {
        self.inner.depth_of_focus()
    }

    /// Exposure latitude in percent.
    fn exposure_latitude(&self) -> f64 {
        self.inner.exposure_latitude()
    }

    /// Get CD matrix as numpy array (doses x focuses).
    #[getter]
    fn cd_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f64>> {
        self.inner.cd_matrix.to_pyarray(py)
    }

    /// Get dose values.
    #[getter]
    fn doses<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.doses.to_pyarray(py)
    }

    /// Get focus values.
    #[getter]
    fn focuses<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.focuses.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessWindowResult(DOF={:.1}nm, EL={:.1}%)",
            self.inner.depth_of_focus(),
            self.inner.exposure_latitude()
        )
    }
}
