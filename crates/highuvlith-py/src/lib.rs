use pyo3::prelude::*;

mod py_config;
mod py_mnsl;
mod py_results;
mod py_simulation;
mod py_sweep;

use py_config::*;
use py_mnsl::*;
use py_results::*;
use py_simulation::*;
use py_sweep::*;

/// highuvlith native module: VUV lithography simulation engine.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configuration classes
    m.add_class::<PySourceConfig>()?;
    m.add_class::<PyOpticsConfig>()?;
    m.add_class::<PyMaskConfig>()?;
    m.add_class::<PyResistConfig>()?;
    m.add_class::<PyFilmStackConfig>()?;
    m.add_class::<PyProcessConfig>()?;
    m.add_class::<PyGridConfig>()?;

    // Simulation engine
    m.add_class::<PySimulationEngine>()?;

    // Results
    m.add_class::<PyAerialImageResult>()?;
    m.add_class::<PyResistProfileResult>()?;

    // Batch / sweep
    m.add_class::<PyBatchSimulator>()?;
    m.add_class::<PyProcessWindowResult>()?;

    // MNSL module
    register_mnsl_module(m)?;

    Ok(())
}
