//! Moiré Nanosphere Lithographic Reflection (MNSL) simulation.
//!
//! Models enhanced emission control through in-plane twisting and rotation
//! between stacked layers of nanosphere arrays to create Moiré interference
//! patterns for next-generation nanophotonic applications.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

use crate::thinfilm::FilmStack;
use crate::types::{Complex64, Grid2D, GridConfig};

/// Nanosphere packing type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SpherePacking {
    /// Hexagonal Close Packed.
    HCP,
    /// Face-Centered Cubic.
    FCC,
    /// Simple Cubic.
    SimpleCubic,
}

/// Substrate coupling configuration for enhanced emission modeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateCoupling {
    /// Substrate material stack for standing wave calculations.
    pub substrate_stack: FilmStack,
    /// Emission enhancement coupling strength (0.0 to 1.0).
    pub coupling_strength: f64,
    /// Enable near-field enhancement calculation.
    pub enable_nearfield: bool,
}

impl Default for SubstrateCoupling {
    fn default() -> Self {
        Self {
            substrate_stack: FilmStack::default(),
            coupling_strength: 0.5,
            enable_nearfield: true,
        }
    }
}

/// Nanosphere array configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NanosphereArray {
    /// Sphere diameter in nm.
    pub diameter_nm: f64,
    /// Center-to-center spacing in nm.
    pub pitch_nm: f64,
    /// Rotation angle in degrees.
    pub orientation_deg: f64,
    /// Real part of refractive index.
    pub n_real: f64,
    /// Imaginary part of refractive index (extinction coefficient).
    pub n_imag: f64,
    /// Sphere packing type.
    pub packing: SpherePacking,
}

impl NanosphereArray {
    /// Create silica nanosphere array with typical VUV optical constants.
    pub fn silica_spheres(diameter_nm: f64, pitch_nm: f64) -> Self {
        Self {
            diameter_nm,
            pitch_nm,
            orientation_deg: 0.0,
            n_real: 1.56,  // SiO2 at 157nm
            n_imag: 0.001, // low absorption
            packing: SpherePacking::HCP,
        }
    }

    /// Create polystyrene nanosphere array.
    pub fn polystyrene_spheres(diameter_nm: f64, pitch_nm: f64) -> Self {
        Self {
            diameter_nm,
            pitch_nm,
            orientation_deg: 0.0,
            n_real: 1.65,  // PS at 157nm
            n_imag: 0.015, // moderate absorption
            packing: SpherePacking::HCP,
        }
    }

    /// Volume fraction of spheres in the array.
    pub fn volume_fraction(&self) -> f64 {
        let sphere_volume = PI * self.diameter_nm.powi(3) / 6.0;
        match self.packing {
            SpherePacking::HCP => {
                let unit_cell_volume = self.pitch_nm.powi(2) * (3.0_f64.sqrt() / 2.0) * self.pitch_nm;
                2.0 * sphere_volume / unit_cell_volume // 2 spheres per HCP unit cell
            }
            SpherePacking::FCC => {
                let unit_cell_volume = self.pitch_nm.powi(3);
                4.0 * sphere_volume / unit_cell_volume // 4 spheres per FCC unit cell
            }
            SpherePacking::SimpleCubic => {
                let unit_cell_volume = self.pitch_nm.powi(3);
                sphere_volume / unit_cell_volume // 1 sphere per simple cubic unit cell
            }
        }
    }
}

/// Complete MNSL configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnslConfig {
    /// Bottom nanosphere array.
    pub bottom_array: NanosphereArray,
    /// Top nanosphere array.
    pub top_array: NanosphereArray,
    /// Vertical separation between array layers in nm.
    pub separation_nm: f64,
    /// Substrate coupling configuration.
    pub substrate: SubstrateCoupling,
    /// Wavelength for calculations in nm.
    pub wavelength_nm: f64,
}

impl Default for MnslConfig {
    fn default() -> Self {
        Self {
            bottom_array: NanosphereArray::silica_spheres(200.0, 300.0),
            top_array: NanosphereArray::silica_spheres(200.0, 300.0),
            separation_nm: 100.0,
            substrate: SubstrateCoupling::default(),
            wavelength_nm: 157.0, // F2 laser
        }
    }
}

/// MNSL simulation result.
#[derive(Debug)]
pub struct MnslResult {
    /// Enhanced emission intensity pattern.
    pub emission_pattern: Grid2D<f64>,
    /// Underlying Moiré interference pattern.
    pub moire_pattern: Grid2D<f64>,
    /// Local field enhancement factors.
    pub enhancement_factors: Grid2D<f64>,
    /// Calculated Moiré period in nm.
    pub moire_period_nm: f64,
    /// Total integrated emission power (relative units).
    pub total_emission_power: f64,
    /// Peak enhancement factor.
    pub peak_enhancement: f64,
    /// Peak positions (x, y) coordinates in nm.
    pub peak_positions: Vec<(f64, f64)>,
}

/// MNSL simulation engine.
pub struct MnslEngine {
    config: MnslConfig,
    grid: GridConfig,
}

impl MnslEngine {
    /// Create new MNSL simulation engine.
    pub fn new(config: MnslConfig, grid: GridConfig) -> Self {
        Self { config, grid }
    }

    /// Compute complete MNSL emission pattern.
    pub fn compute_emission(&self) -> MnslResult {
        // 1. Generate nanosphere positions for both arrays
        let bottom_positions = self.generate_nanosphere_positions(&self.config.bottom_array);
        let top_positions = self.generate_nanosphere_positions(&self.config.top_array);

        // 2. Apply rotation to top array
        let rotated_top = self.apply_rotation_transform(&top_positions, self.config.top_array.orientation_deg);

        // 3. Compute Moiré interference pattern
        let moire_pattern = self.compute_moire_interference(&bottom_positions, &rotated_top);

        // 4. Calculate local field enhancement factors
        let enhancement_factors = self.compute_field_enhancement(&moire_pattern);

        // 5. Model substrate coupling and enhanced emission
        let emission_pattern = self.compute_substrate_emission(&enhancement_factors);

        // 6. Calculate Moiré period
        let moire_period_nm = self.calculate_moire_period();

        // 7. Find peak positions
        let peak_positions = self.find_emission_peaks(&emission_pattern);

        // 8. Calculate integrated quantities
        let total_emission_power = emission_pattern.data.iter().sum();
        let peak_enhancement = enhancement_factors.data.iter().cloned()
            .fold(0.0f64, f64::max);

        MnslResult {
            emission_pattern,
            moire_pattern,
            enhancement_factors,
            moire_period_nm,
            total_emission_power,
            peak_enhancement,
            peak_positions,
        }
    }

    /// Generate nanosphere positions using DSA-based algorithms.
    fn generate_nanosphere_positions(&self, array: &NanosphereArray) -> Vec<(f64, f64)> {
        let field_size = self.grid.field_size_nm();
        let mut positions = Vec::new();

        match array.packing {
            SpherePacking::HCP => {
                // Hexagonal close packed structure
                let hex_height = array.pitch_nm * (3.0_f64).sqrt() / 2.0;
                let half_field = field_size / 2.0;

                let mut row = 0;
                let mut y = -half_field;
                while y < half_field {
                    let x_offset = if row % 2 == 0 { 0.0 } else { array.pitch_nm / 2.0 };
                    let mut x = -half_field + x_offset;

                    while x < half_field {
                        positions.push((x, y));
                        x += array.pitch_nm;
                    }
                    y += hex_height;
                    row += 1;
                }
            }
            SpherePacking::FCC => {
                // Face-centered cubic (projected to 2D)
                let step = array.pitch_nm / (2.0_f64).sqrt();
                let half_field = field_size / 2.0;

                let mut y = -half_field;
                let mut row = 0;
                while y < half_field {
                    let x_offset = if row % 2 == 0 { 0.0 } else { step / 2.0 };
                    let mut x = -half_field + x_offset;

                    while x < half_field {
                        positions.push((x, y));
                        x += step;
                    }
                    y += step * (3.0_f64).sqrt() / 2.0;
                    row += 1;
                }
            }
            SpherePacking::SimpleCubic => {
                // Simple cubic lattice
                let half_field = field_size / 2.0;
                let mut y = -half_field;
                while y < half_field {
                    let mut x = -half_field;
                    while x < half_field {
                        positions.push((x, y));
                        x += array.pitch_nm;
                    }
                    y += array.pitch_nm;
                }
            }
        }

        positions
    }

    /// Apply rotation transformation to nanosphere positions.
    pub fn apply_rotation_transform(&self, positions: &[(f64, f64)], angle_deg: f64) -> Vec<(f64, f64)> {
        if angle_deg.abs() < 1e-12 {
            return positions.to_vec();
        }

        let angle_rad = angle_deg * PI / 180.0;
        let cos_theta = angle_rad.cos();
        let sin_theta = angle_rad.sin();

        positions.iter().map(|&(x, y)| {
            let x_rot = x * cos_theta - y * sin_theta;
            let y_rot = x * sin_theta + y * cos_theta;
            (x_rot, y_rot)
        }).collect()
    }

    /// Compute Moiré interference pattern from overlapping nanosphere arrays.
    fn compute_moire_interference(&self, bottom: &[(f64, f64)], top: &[(f64, f64)]) -> Grid2D<f64> {
        let n = self.grid.size;
        let mut pattern = Array2::zeros((n, n));
        let half_field = self.grid.field_size_nm() / 2.0;

        for i in 0..n {
            for j in 0..n {
                let x = -half_field + (j as f64 + 0.5) * self.grid.pixel_nm;
                let y = -half_field + (i as f64 + 0.5) * self.grid.pixel_nm;

                // Calculate scattering contributions from both layers
                let bottom_contrib = self.calculate_array_scattering(x, y, bottom, &self.config.bottom_array);
                let top_contrib = self.calculate_array_scattering(x, y, top, &self.config.top_array);

                // Phase difference due to layer separation
                let k = TAU / self.config.wavelength_nm;
                let phase_diff = k * self.config.separation_nm;
                let top_contrib_shifted = top_contrib * Complex64::from_polar(1.0, phase_diff);

                // Interference intensity
                let total_field = bottom_contrib + top_contrib_shifted;
                pattern[[i, j]] = total_field.norm_sqr();
            }
        }

        Grid2D {
            data: pattern,
            x_min_nm: -half_field,
            x_max_nm: half_field,
            y_min_nm: -half_field,
            y_max_nm: half_field,
        }
    }

    /// Calculate scattering contribution from a nanosphere array at given position.
    fn calculate_array_scattering(&self, x: f64, y: f64, positions: &[(f64, f64)], array: &NanosphereArray) -> Complex64 {
        let k = TAU / self.config.wavelength_nm;
        let radius = array.diameter_nm / 2.0;
        let n_sphere = Complex64::new(array.n_real, array.n_imag);

        // Rayleigh scattering approximation for small spheres
        let volume = (4.0 * PI / 3.0) * radius.powi(3);
        let alpha = volume * (n_sphere * n_sphere - 1.0) / (n_sphere * n_sphere + 2.0);

        let mut total_field = Complex64::new(0.0, 0.0);

        for &(sphere_x, sphere_y) in positions {
            let dx = x - sphere_x;
            let dy = y - sphere_y;
            let r = (dx * dx + dy * dy).sqrt();

            if r < radius {
                // Inside sphere: full interaction
                total_field += alpha;
            } else {
                // Outside sphere: scattered field with phase
                let phase = k * r;
                let field_amplitude = alpha / (r * r).max(radius * radius);
                total_field += field_amplitude * Complex64::from_polar(1.0, phase);
            }
        }

        total_field
    }

    /// Compute local field enhancement factors.
    fn compute_field_enhancement(&self, moire_pattern: &Grid2D<f64>) -> Grid2D<f64> {
        let n = self.grid.size;
        let mut enhancement = Array2::zeros((n, n));

        // Calculate mean intensity for normalization
        let mean_intensity = moire_pattern.data.iter().sum::<f64>() / (n * n) as f64;

        for i in 0..n {
            for j in 0..n {
                let local_intensity = moire_pattern.data[[i, j]];
                // Enhancement factor is local intensity relative to mean
                enhancement[[i, j]] = if mean_intensity > 0.0 {
                    (local_intensity / mean_intensity).max(1.0)
                } else {
                    1.0
                };
            }
        }

        Grid2D {
            data: enhancement,
            x_min_nm: moire_pattern.x_min_nm,
            x_max_nm: moire_pattern.x_max_nm,
            y_min_nm: moire_pattern.y_min_nm,
            y_max_nm: moire_pattern.y_max_nm,
        }
    }

    /// Model enhanced emission using substrate coupling and thin-film standing waves.
    fn compute_substrate_emission(&self, enhancement: &Grid2D<f64>) -> Grid2D<f64> {
        let n = self.grid.size;
        let mut emission = Array2::zeros((n, n));

        if !self.config.substrate.enable_nearfield {
            // Simple multiplicative enhancement
            emission = enhancement.data.clone();
        } else {
            // Calculate substrate standing wave enhancement
            let z_points = vec![self.config.separation_nm];
            let standing_wave = self.config.substrate.substrate_stack
                .standing_wave(self.config.wavelength_nm, &z_points);
            let substrate_factor = standing_wave.get(0).copied().unwrap_or(1.0);

            for i in 0..n {
                for j in 0..n {
                    let local_enhancement = enhancement.data[[i, j]];
                    let substrate_enhancement = 1.0 + self.config.substrate.coupling_strength * (substrate_factor - 1.0);
                    emission[[i, j]] = local_enhancement * substrate_enhancement;
                }
            }
        }

        Grid2D {
            data: emission,
            x_min_nm: enhancement.x_min_nm,
            x_max_nm: enhancement.x_max_nm,
            y_min_nm: enhancement.y_min_nm,
            y_max_nm: enhancement.y_max_nm,
        }
    }

    /// Calculate theoretical Moiré period.
    pub fn calculate_moire_period(&self) -> f64 {
        let pitch1 = self.config.bottom_array.pitch_nm;
        let pitch2 = self.config.top_array.pitch_nm;
        let angle_rad = self.config.top_array.orientation_deg * PI / 180.0;

        if angle_rad.abs() < 1e-12 {
            // Parallel layers: period determined by pitch difference
            if (pitch1 - pitch2).abs() < 1e-12 {
                // Same pitch: infinite period
                1e6
            } else {
                pitch1 * pitch2 / (pitch1 - pitch2).abs()
            }
        } else {
            // Rotated layers: period determined by rotation angle
            pitch1 / (2.0 * angle_rad.sin().abs())
        }
    }

    /// Find local maxima in emission pattern (peak positions).
    fn find_emission_peaks(&self, emission: &Grid2D<f64>) -> Vec<(f64, f64)> {
        let n = self.grid.size;
        let mut peaks = Vec::new();
        let threshold = 0.8; // Find peaks above 80% of maximum

        // Find global maximum
        let max_value = emission.data.iter().cloned()
            .fold(0.0f64, f64::max);
        let min_peak_value = threshold * max_value;

        // Search for local maxima
        for i in 1..n-1 {
            for j in 1..n-1 {
                let center = emission.data[[i, j]];
                if center > min_peak_value {
                    // Check if it's a local maximum
                    let is_peak = center > emission.data[[i-1, j]] &&
                                 center > emission.data[[i+1, j]] &&
                                 center > emission.data[[i, j-1]] &&
                                 center > emission.data[[i, j+1]] &&
                                 center > emission.data[[i-1, j-1]] &&
                                 center > emission.data[[i-1, j+1]] &&
                                 center > emission.data[[i+1, j-1]] &&
                                 center > emission.data[[i+1, j+1]];

                    if is_peak {
                        let x = emission.x_at(j);
                        let y = emission.y_at(i);
                        peaks.push((x, y));
                    }
                }
            }
        }

        peaks
    }
}

/// Convenience function for quick MNSL simulation.
pub fn simulate_moire_emission(
    sphere_diameter_nm: f64,
    array_pitch_nm: f64,
    rotation_angle_deg: f64,
    separation_nm: f64,
    grid: GridConfig,
) -> MnslResult {
    let mut config = MnslConfig::default();

    // Configure arrays
    config.bottom_array = NanosphereArray::silica_spheres(sphere_diameter_nm, array_pitch_nm);
    config.top_array = NanosphereArray::silica_spheres(sphere_diameter_nm, array_pitch_nm);
    config.top_array.orientation_deg = rotation_angle_deg;
    config.separation_nm = separation_nm;

    let engine = MnslEngine::new(config, grid);
    engine.compute_emission()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nanosphere_array_volume_fraction() {
        let array = NanosphereArray::silica_spheres(200.0, 300.0);
        let vf = array.volume_fraction();
        assert!(vf > 0.0 && vf < 1.0, "Volume fraction should be between 0 and 1");
    }

    #[test]
    fn test_moire_period_calculation() {
        let config = MnslConfig::default();
        let grid = GridConfig::new(128, 2.0).unwrap();
        let engine = MnslEngine::new(config, grid);

        let period = engine.calculate_moire_period();
        assert!(period > 0.0, "Moiré period should be positive");
    }

    #[test]
    fn test_rotation_transform_identity() {
        let grid = GridConfig::new(128, 2.0).unwrap();
        let config = MnslConfig::default();
        let engine = MnslEngine::new(config, grid);

        let positions = vec![(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)];
        let rotated = engine.apply_rotation_transform(&positions, 0.0);

        for (orig, rot) in positions.iter().zip(rotated.iter()) {
            assert_relative_eq!(orig.0, rot.0, epsilon = 1e-10);
            assert_relative_eq!(orig.1, rot.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rotation_transform_90deg() {
        let grid = GridConfig::new(128, 2.0).unwrap();
        let config = MnslConfig::default();
        let engine = MnslEngine::new(config, grid);

        let positions = vec![(100.0, 0.0)];
        let rotated = engine.apply_rotation_transform(&positions, 90.0);

        assert_relative_eq!(rotated[0].0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated[0].1, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mnsl_simulation_runs() {
        let grid = GridConfig::new(64, 4.0).unwrap();
        let result = simulate_moire_emission(200.0, 300.0, 5.0, 100.0, grid);

        assert!(result.moire_period_nm > 0.0);
        assert!(result.total_emission_power > 0.0);
        assert!(result.peak_enhancement >= 1.0);
        assert_eq!(result.emission_pattern.data.dim(), (64, 64));
    }

    #[test]
    fn test_substrate_coupling_default() {
        let coupling = SubstrateCoupling::default();
        assert!(coupling.coupling_strength >= 0.0 && coupling.coupling_strength <= 1.0);
        assert!(coupling.enable_nearfield);
    }
}