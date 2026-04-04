use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

use highuvlith_core::metrics;
use highuvlith_core::resist::ResistProfile;
use highuvlith_core::types::Grid2D;

/// Aerial image simulation result.
#[pyclass(name = "AerialImageResult")]
#[derive(Debug, Clone)]
pub struct PyAerialImageResult {
    pub intensity: ndarray::Array2<f64>,
    pub x_min_nm: f64,
    pub x_max_nm: f64,
    pub y_min_nm: f64,
    pub y_max_nm: f64,
}

impl PyAerialImageResult {
    pub fn from_grid2d(grid: Grid2D<f64>) -> Self {
        Self {
            intensity: grid.data,
            x_min_nm: grid.x_min_nm,
            x_max_nm: grid.x_max_nm,
            y_min_nm: grid.y_min_nm,
            y_max_nm: grid.y_max_nm,
        }
    }
}

#[pymethods]
impl PyAerialImageResult {
    /// Get intensity as a numpy array.
    #[getter]
    fn intensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.intensity.to_pyarray(py)
    }

    /// Get x coordinate array in nm.
    #[getter]
    fn x_nm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let nx = self.intensity.ncols();
        let pixel = (self.x_max_nm - self.x_min_nm) / nx as f64;
        let x: Vec<f64> = (0..nx)
            .map(|j| self.x_min_nm + (j as f64 + 0.5) * pixel)
            .collect();
        x.to_pyarray(py)
    }

    /// Get y coordinate array in nm.
    #[getter]
    fn y_nm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ny = self.intensity.nrows();
        let pixel = (self.y_max_nm - self.y_min_nm) / ny as f64;
        let y: Vec<f64> = (0..ny)
            .map(|i| self.y_min_nm + (i as f64 + 0.5) * pixel)
            .collect();
        y.to_pyarray(py)
    }

    /// Extract a 1D cross-section along x at y=y_nm.
    #[pyo3(signature = (y_nm=0.0))]
    fn cross_section<'py>(
        &self,
        py: Python<'py>,
        y_nm: f64,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let ny = self.intensity.nrows();
        let nx = self.intensity.ncols();
        let pixel_y = (self.y_max_nm - self.y_min_nm) / ny as f64;
        let pixel_x = (self.x_max_nm - self.x_min_nm) / nx as f64;

        // Find nearest row
        let row_idx = ((y_nm - self.y_min_nm) / pixel_y - 0.5)
            .round()
            .clamp(0.0, (ny - 1) as f64) as usize;

        let x: Vec<f64> = (0..nx)
            .map(|j| self.x_min_nm + (j as f64 + 0.5) * pixel_x)
            .collect();
        let vals: Vec<f64> = (0..nx).map(|j| self.intensity[[row_idx, j]]).collect();

        (x.to_pyarray(py), vals.to_pyarray(py))
    }

    /// Compute image contrast: (Imax - Imin) / (Imax + Imin).
    fn image_contrast(&self) -> f64 {
        metrics::image_contrast(&self.intensity)
    }

    /// Compute NILS at y=0 cross-section.
    #[pyo3(signature = (threshold=0.3))]
    fn nils(&self, threshold: f64) -> Option<f64> {
        let nx = self.intensity.ncols();
        let ny = self.intensity.nrows();
        let pixel = (self.x_max_nm - self.x_min_nm) / nx as f64;
        let center_row = ny / 2;

        let x_nm: Vec<f64> = (0..nx)
            .map(|j| self.x_min_nm + (j as f64 + 0.5) * pixel)
            .collect();
        let profile: Vec<f64> = (0..nx).map(|j| self.intensity[[center_row, j]]).collect();

        metrics::nils(&profile, &x_nm, threshold)
    }

    fn __eq__(&self, other: &Self) -> bool {
        let bounds_eq = (self.x_min_nm - other.x_min_nm).abs() < 1e-9
            && (self.x_max_nm - other.x_max_nm).abs() < 1e-9
            && (self.y_min_nm - other.y_min_nm).abs() < 1e-9
            && (self.y_max_nm - other.y_max_nm).abs() < 1e-9;
        if !bounds_eq {
            return false;
        }
        if self.intensity.shape() != other.intensity.shape() {
            return false;
        }
        self.intensity
            .iter()
            .zip(other.intensity.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12)
    }

    fn __repr__(&self) -> String {
        format!(
            "AerialImageResult({}x{}, x=[{:.1}, {:.1}]nm, y=[{:.1}, {:.1}]nm)",
            self.intensity.nrows(),
            self.intensity.ncols(),
            self.x_min_nm,
            self.x_max_nm,
            self.y_min_nm,
            self.y_max_nm,
        )
    }
}

/// Resist profile simulation result.
#[pyclass(name = "ResistProfileResult")]
#[derive(Debug, Clone)]
pub struct PyResistProfileResult {
    x_nm: Vec<f64>,
    height_nm: Vec<f64>,
    thickness_nm: f64,
}

impl PyResistProfileResult {
    pub fn from_profile(profile: ResistProfile) -> Self {
        Self {
            x_nm: profile.x_nm,
            height_nm: profile.height_nm,
            thickness_nm: profile.thickness_nm,
        }
    }
}

#[pymethods]
impl PyResistProfileResult {
    /// X positions in nm.
    #[getter]
    fn x_nm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.x_nm.to_pyarray(py)
    }

    /// Remaining resist height at each position in nm.
    #[getter]
    fn height_nm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.height_nm.to_pyarray(py)
    }

    /// Original resist thickness in nm.
    #[getter]
    fn thickness_nm(&self) -> f64 {
        self.thickness_nm
    }

    fn __eq__(&self, other: &Self) -> bool {
        if self.height_nm.len() != other.height_nm.len() {
            return false;
        }
        if (self.thickness_nm - other.thickness_nm).abs() > 1e-9 {
            return false;
        }
        self.height_nm
            .iter()
            .zip(other.height_nm.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12)
    }

    fn __repr__(&self) -> String {
        format!(
            "ResistProfileResult({} points, thickness={}nm)",
            self.x_nm.len(),
            self.thickness_nm
        )
    }
}
