use thiserror::Error;

#[derive(Debug, Error)]
pub enum LithographyError {
    #[error("invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        name: &'static str,
        value: f64,
        reason: &'static str,
    },

    #[error("grid size {0} must be a power of 2 for FFT")]
    GridSizeNotPowerOfTwo(usize),

    #[error("no diffraction orders pass the pupil (pitch too small or NA too low)")]
    NoDiffractionOrders,

    #[error("material not found: {0}")]
    MaterialNotFound(String),

    #[error("TCC decomposition failed: {0}")]
    TccDecomposition(String),

    #[error("convergence failure after {iterations} iterations (residual: {residual:.2e})")]
    ConvergenceFailure { iterations: usize, residual: f64 },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("numerical error: {0}")]
    NumericalError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, LithographyError>;
