use super::ComputeBackend;
use crate::math::fft2d::Fft2D;
use crate::types::Complex64;
use ndarray::Array2;

/// CPU compute backend using rustfft.
pub struct CpuBackend {
    fft: Fft2D,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { fft: Fft2D::new() }
    }

    /// Access the underlying FFT engine for direct use.
    pub fn fft(&self) -> &Fft2D {
        &self.fft
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn fft2d_forward(&self, data: &mut [Complex64], nx: usize, ny: usize) {
        let mut arr = Array2::from_shape_vec((ny, nx), data.to_vec())
            .expect("invalid dimensions for FFT");
        self.fft.forward(&mut arr);
        data.copy_from_slice(arr.as_slice().expect("array not contiguous"));
    }

    fn fft2d_inverse(&self, data: &mut [Complex64], nx: usize, ny: usize) {
        let mut arr = Array2::from_shape_vec((ny, nx), data.to_vec())
            .expect("invalid dimensions for FFT");
        self.fft.inverse(&mut arr);
        data.copy_from_slice(arr.as_slice().expect("array not contiguous"));
    }
}
