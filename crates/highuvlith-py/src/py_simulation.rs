use pyo3::prelude::*;

use highuvlith_core::aerial::AerialImageEngine;
use highuvlith_core::metrics;
use highuvlith_core::resist;

use crate::py_config::*;
use crate::py_results::*;

/// Core simulation engine. Precomputes TCC for efficient multi-evaluation.
#[pyclass(name = "SimulationEngine")]
pub struct PySimulationEngine {
    engine: AerialImageEngine,
    source: PySourceConfig,
    optics: PyOpticsConfig,
    mask: PyMaskConfig,
    resist: PyResistConfig,
    grid: PyGridConfig,
}

#[pymethods]
impl PySimulationEngine {
    #[new]
    #[pyo3(signature = (source, optics, mask, resist=None, grid=None, max_kernels=30))]
    fn new(
        source: PySourceConfig,
        optics: PyOpticsConfig,
        mask: PyMaskConfig,
        resist: Option<PyResistConfig>,
        grid: Option<PyGridConfig>,
        max_kernels: usize,
    ) -> PyResult<Self> {
        let grid = grid.unwrap_or_else(|| PyGridConfig {
            inner: highuvlith_core::types::GridConfig::default(),
        });
        let resist = resist.unwrap_or_else(|| PyResistConfig {
            inner: highuvlith_core::resist::ResistParams::vuv_fluoropolymer(),
        });

        let engine = AerialImageEngine::new(
            &source.inner,
            &optics.inner,
            grid.inner.clone(),
            max_kernels,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            engine,
            source,
            optics,
            mask,
            resist,
            grid,
        })
    }

    /// Compute aerial image at the given defocus.
    #[pyo3(signature = (focus_nm=0.0))]
    fn compute_aerial_image(
        &self,
        _py: Python<'_>,
        focus_nm: f64,
    ) -> PyResult<PyAerialImageResult> {
        // Release GIL during computation
        let grid = self.engine.compute(&self.mask.inner, focus_nm);

        Ok(PyAerialImageResult::from_grid2d(grid))
    }

    /// Compute polychromatic aerial image (accounts for VUV chromatic aberration).
    #[pyo3(signature = (focus_nm=0.0))]
    fn compute_polychromatic(
        &self,
        _py: Python<'_>,
        focus_nm: f64,
    ) -> PyResult<PyAerialImageResult> {
        let grid = self.engine.compute_polychromatic(
            &self.mask.inner,
            focus_nm,
            &self.source.inner,
            &self.optics.inner,
        );

        Ok(PyAerialImageResult::from_grid2d(grid))
    }

    /// Measure CD at given dose and focus.
    #[pyo3(signature = (dose_mj_cm2=30.0, focus_nm=0.0, threshold=0.3))]
    fn measure_cd(
        &self,
        dose_mj_cm2: f64,
        focus_nm: f64,
        threshold: f64,
    ) -> PyResult<f64> {
        let aerial = self.engine.compute(&self.mask.inner, focus_nm);
        let field = self.grid.inner.field_size_nm();
        let half = field / 2.0;

        metrics::measure_cd_2d(&aerial.data, -half, half, threshold)
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Could not measure CD: no threshold crossings found")
            })
    }

    /// Compute resist profile at given dose and focus.
    #[pyo3(signature = (dose_mj_cm2=30.0, focus_nm=0.0, dev_time_s=60.0))]
    fn compute_resist_profile(
        &self,
        dose_mj_cm2: f64,
        focus_nm: f64,
        dev_time_s: f64,
    ) -> PyResult<PyResistProfileResult> {
        let aerial = self.engine.compute(&self.mask.inner, focus_nm);

        let mut latent = resist::expose(&aerial.data, dose_mj_cm2, &self.resist.inner);

        resist::peb_diffuse(
            &mut latent,
            self.resist.inner.peb_diffusion_nm,
            self.grid.inner.pixel_nm,
        );

        let profile = resist::develop(
            &latent,
            &self.resist.inner,
            dev_time_s,
            self.grid.inner.pixel_nm,
        );

        Ok(PyResistProfileResult::from_profile(profile))
    }

    /// Get image contrast at given focus.
    #[pyo3(signature = (focus_nm=0.0))]
    fn image_contrast(&self, focus_nm: f64) -> f64 {
        let aerial = self.engine.compute(&self.mask.inner, focus_nm);
        metrics::image_contrast(&aerial.data)
    }

    /// Number of SOCS kernels in the decomposition.
    fn num_kernels(&self) -> usize {
        self.engine.num_kernels()
    }

    fn __repr__(&self) -> String {
        format!(
            "SimulationEngine(λ={}nm, NA={}, kernels={})",
            self.source.inner.wavelength_nm,
            self.optics.inner.na,
            self.engine.num_kernels()
        )
    }
}
