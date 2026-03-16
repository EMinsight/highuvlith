pub mod cpu;

use crate::types::Complex64;

/// Trait abstracting the compute backend for aerial image simulation.
/// Enables CPU, GPU (wgpu), and WASM backends.
pub trait ComputeBackend: Send + Sync {
    /// Forward 2D FFT in-place.
    fn fft2d_forward(&self, data: &mut [Complex64], nx: usize, ny: usize);

    /// Inverse 2D FFT in-place with 1/N normalization.
    fn fft2d_inverse(&self, data: &mut [Complex64], nx: usize, ny: usize);
}
