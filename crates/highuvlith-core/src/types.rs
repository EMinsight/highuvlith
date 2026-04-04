use ndarray::Array2;
use num::Complex;
use serde::{Deserialize, Serialize};

pub type Complex64 = Complex<f64>;

/// 2D grid with physical coordinate mapping.
#[derive(Debug, Clone)]
pub struct Grid2D<T> {
    pub data: Array2<T>,
    pub x_min_nm: f64,
    pub x_max_nm: f64,
    pub y_min_nm: f64,
    pub y_max_nm: f64,
}

impl<T: Clone + Default> Grid2D<T> {
    pub fn new(
        nx: usize,
        ny: usize,
        x_range: (f64, f64),
        y_range: (f64, f64),
    ) -> crate::error::Result<Self> {
        if nx == 0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "nx",
                value: 0.0,
                reason: "must be positive",
            });
        }
        if ny == 0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "ny",
                value: 0.0,
                reason: "must be positive",
            });
        }
        if x_range.0 >= x_range.1 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "x_range",
                value: x_range.0,
                reason: "x_min must be less than x_max",
            });
        }
        if y_range.0 >= y_range.1 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "y_range",
                value: y_range.0,
                reason: "y_min must be less than y_max",
            });
        }
        Ok(Self {
            data: Array2::default((ny, nx)),
            x_min_nm: x_range.0,
            x_max_nm: x_range.1,
            y_min_nm: y_range.0,
            y_max_nm: y_range.1,
        })
    }

    pub fn nx(&self) -> usize {
        self.data.ncols()
    }

    pub fn ny(&self) -> usize {
        self.data.nrows()
    }

    pub fn pixel_size_x(&self) -> f64 {
        (self.x_max_nm - self.x_min_nm) / self.nx() as f64
    }

    pub fn pixel_size_y(&self) -> f64 {
        (self.y_max_nm - self.y_min_nm) / self.ny() as f64
    }

    /// Physical x-coordinate for column index.
    pub fn x_at(&self, col: usize) -> f64 {
        self.x_min_nm + (col as f64 + 0.5) * self.pixel_size_x()
    }

    /// Physical y-coordinate for row index.
    pub fn y_at(&self, row: usize) -> f64 {
        self.y_min_nm + (row as f64 + 0.5) * self.pixel_size_y()
    }
}

/// 3D grid for volumetric data (e.g., resist dose distribution).
#[derive(Debug, Clone)]
pub struct Grid3D<T> {
    pub data: ndarray::Array3<T>,
    pub x_min_nm: f64,
    pub x_max_nm: f64,
    pub y_min_nm: f64,
    pub y_max_nm: f64,
    pub z_min_nm: f64,
    pub z_max_nm: f64,
}

/// Simulation grid configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Number of grid points (must be power of 2).
    pub size: usize,
    /// Physical pixel size at wafer in nm.
    pub pixel_nm: f64,
}

impl GridConfig {
    pub fn new(size: usize, pixel_nm: f64) -> crate::error::Result<Self> {
        if !size.is_power_of_two() {
            return Err(crate::error::LithographyError::GridSizeNotPowerOfTwo(size));
        }
        if pixel_nm <= 0.0 {
            return Err(crate::error::LithographyError::InvalidParameter {
                name: "pixel_nm",
                value: pixel_nm,
                reason: "must be positive",
            });
        }
        Ok(Self { size, pixel_nm })
    }

    pub fn field_size_nm(&self) -> f64 {
        self.size as f64 * self.pixel_nm
    }

    /// Frequency spacing in 1/nm.
    pub fn freq_step(&self) -> f64 {
        1.0 / self.field_size_nm()
    }
}

/// Polarization state for thin-film calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarization {
    TE,
    TM,
    Unpolarized,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            size: 512,
            pixel_nm: 1.0,
        }
    }
}
