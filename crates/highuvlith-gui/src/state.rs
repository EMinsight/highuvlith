use highuvlith_core::aerial::AerialImageEngine;
use highuvlith_core::mask::Mask;
use highuvlith_core::metrics;
use highuvlith_core::optics::ProjectionOptics;
use highuvlith_core::source::{IlluminationShape, VuvSource};
use highuvlith_core::types::GridConfig;
use ndarray::Array2;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone)]
pub struct SimParams {
    pub wavelength_nm: f64,
    pub sigma: f64,
    pub na: f64,
    pub cd_nm: f64,
    pub pitch_nm: f64,
    pub focus_nm: f64,
    pub grid_size: usize,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            wavelength_nm: 157.63,
            sigma: 0.7,
            na: 0.75,
            cd_nm: 65.0,
            pitch_nm: 180.0,
            focus_nm: 0.0,
            grid_size: 128,
        }
    }
}

pub struct SimResult {
    pub intensity: Array2<f64>,
    pub contrast: f64,
    pub i_max: f64,
    pub i_min: f64,
    pub num_kernels: usize,
    pub compute_ms: f64,
    pub cross_section_x: Vec<f64>,
    pub cross_section_i: Vec<f64>,
}

pub struct SimState {
    pub params: SimParams,
    pub result: Arc<Mutex<Option<SimResult>>>,
    pub computing: Arc<Mutex<bool>>,
    dirty: bool,
}

impl SimState {
    pub fn new() -> Self {
        Self {
            params: SimParams::default(),
            result: Arc::new(Mutex::new(None)),
            computing: Arc::new(Mutex::new(false)),
            dirty: true,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn try_compute(&mut self) {
        if !self.dirty {
            return;
        }
        let is_computing = *self.computing.lock().unwrap();
        if is_computing {
            return;
        }
        self.dirty = false;
        *self.computing.lock().unwrap() = true;

        let params = self.params.clone();
        let result_ref = Arc::clone(&self.result);
        let computing_ref = Arc::clone(&self.computing);

        thread::spawn(move || {
            let source = VuvSource {
                wavelength_nm: params.wavelength_nm,
                bandwidth_pm: 1.1,
                spectral_samples: 1,
                spectral_shape: highuvlith_core::source::SpectralShape::Lorentzian,
                pulse_energy_mj: 10.0,
                rep_rate_hz: 4000.0,
                illumination: IlluminationShape::Conventional { sigma: params.sigma },
            };
            let optics = ProjectionOptics::new(params.na);
            let grid = GridConfig {
                size: params.grid_size,
                pixel_nm: 2.0,
            };
            let mask = Mask::line_space(params.cd_nm, params.pitch_nm);

            let start = std::time::Instant::now();
            if let Ok(engine) = AerialImageEngine::new(&source, &optics, grid, 15) {
                let aerial = engine.compute(&mask, params.focus_nm);
                let elapsed = start.elapsed();

                let contrast = metrics::image_contrast(&aerial.data);
                let i_max = aerial.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let i_min = aerial.data.iter().cloned().fold(f64::INFINITY, f64::min);

                let n = aerial.data.ncols();
                let center_row = aerial.data.nrows() / 2;
                let field = params.grid_size as f64 * 2.0;
                let half = field / 2.0;
                let pixel = field / n as f64;
                let cross_section_x: Vec<f64> =
                    (0..n).map(|j| -half + (j as f64 + 0.5) * pixel).collect();
                let cross_section_i: Vec<f64> =
                    (0..n).map(|j| aerial.data[[center_row, j]]).collect();

                let sim_result = SimResult {
                    intensity: aerial.data,
                    contrast,
                    i_max,
                    i_min,
                    num_kernels: engine.num_kernels(),
                    compute_ms: elapsed.as_secs_f64() * 1000.0,
                    cross_section_x,
                    cross_section_i,
                };
                *result_ref.lock().unwrap() = Some(sim_result);
            }
            *computing_ref.lock().unwrap() = false;
        });
    }
}
