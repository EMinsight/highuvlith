use num::Complex;
use serde::{Deserialize, Serialize};

use crate::types::{Complex64, Polarization};

/// A single layer in a thin-film stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilmLayer {
    pub name: String,
    /// Thickness in nm.
    pub thickness_nm: f64,
    /// Complex refractive index (n + i*k, where k is extinction coefficient).
    #[serde(
        serialize_with = "serialize_complex",
        deserialize_with = "deserialize_complex"
    )]
    pub n: Complex64,
}

/// Thin-film stack for standing wave and reflectance calculations.
/// Layers are ordered from top (superstrate side) to bottom (substrate side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilmStack {
    /// Thin-film layers from top to bottom.
    pub layers: Vec<FilmLayer>,
    /// Complex refractive index of the substrate (semi-infinite bottom medium).
    #[serde(
        serialize_with = "serialize_complex",
        deserialize_with = "deserialize_complex"
    )]
    pub substrate: Complex64,
    /// Superstrate refractive index (vacuum for VUV: n=1.0).
    #[serde(
        serialize_with = "serialize_complex",
        deserialize_with = "deserialize_complex"
    )]
    pub superstrate: Complex64,
}

impl FilmStack {
    /// Create a VUV film stack with vacuum superstrate.
    pub fn new_vuv(layers: Vec<FilmLayer>, substrate: Complex64) -> Self {
        Self {
            layers,
            substrate,
            superstrate: Complex64::new(1.0, 0.0), // vacuum
        }
    }

    /// Compute reflectance at the top surface for normal incidence.
    pub fn reflectance(&self, wavelength_nm: f64, pol: Polarization) -> f64 {
        self.reflectance_at_angle(wavelength_nm, 0.0, pol)
    }

    /// Compute reflectance for arbitrary incidence angle (radians, from normal).
    pub fn reflectance_at_angle(
        &self,
        wavelength_nm: f64,
        angle_rad: f64,
        pol: Polarization,
    ) -> f64 {
        match pol {
            Polarization::TE => {
                let (r, _) = self.transfer_matrix(wavelength_nm, angle_rad, Polarization::TE);
                r.norm_sqr()
            }
            Polarization::TM => {
                let (r, _) = self.transfer_matrix(wavelength_nm, angle_rad, Polarization::TM);
                r.norm_sqr()
            }
            Polarization::Unpolarized => {
                let r_te = self.reflectance_at_angle(wavelength_nm, angle_rad, Polarization::TE);
                let r_tm = self.reflectance_at_angle(wavelength_nm, angle_rad, Polarization::TM);
                (r_te + r_tm) / 2.0
            }
        }
    }

    /// Compute the standing wave intensity pattern inside a film.
    /// Returns intensity values at the given z-positions (measured from top of stack, positive downward).
    pub fn standing_wave(
        &self,
        wavelength_nm: f64,
        z_points: &[f64],
    ) -> Vec<f64> {
        let k0 = 2.0 * std::f64::consts::PI / wavelength_nm;

        z_points
            .iter()
            .map(|&z| {
                let (forward, backward) = self.field_at_depth(wavelength_nm, z, k0);
                // Intensity = |E_forward + E_backward|^2
                (forward + backward).norm_sqr()
            })
            .collect()
    }

    /// Transfer matrix method for the entire stack.
    /// Returns (reflection_coefficient, transmission_coefficient).
    fn transfer_matrix(
        &self,
        wavelength_nm: f64,
        angle_rad: f64,
        pol: Polarization,
    ) -> (Complex64, Complex64) {
        let k0 = 2.0 * std::f64::consts::PI / wavelength_nm;
        let cos_0 = angle_rad.cos();

        // Snell's law for complex media
        let n0_sin = self.superstrate * Complex64::new(angle_rad.sin(), 0.0);

        // Build transfer matrix M = product of layer matrices
        let mut m = [[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                     [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]];

        for layer in &self.layers {
            let cos_j = ((Complex64::new(1.0, 0.0) - (n0_sin / layer.n).powi(2))).sqrt();
            let phase = k0 * layer.n * cos_j * layer.thickness_nm;

            let eta_j = match pol {
                Polarization::TE | Polarization::Unpolarized => layer.n * cos_j,
                Polarization::TM => layer.n / cos_j,
            };

            // Layer matrix
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();
            let l = [
                [cos_phase, Complex64::new(0.0, 1.0) * sin_phase / eta_j],
                [Complex64::new(0.0, 1.0) * eta_j * sin_phase, cos_phase],
            ];

            // Matrix multiply: m = m * l
            let new_m = mat2_mul(&m, &l);
            m = new_m;
        }

        // Substrate admittance
        let cos_s = ((Complex64::new(1.0, 0.0) - (n0_sin / self.substrate).powi(2))).sqrt();
        let eta_s = match pol {
            Polarization::TE | Polarization::Unpolarized => self.substrate * cos_s,
            Polarization::TM => self.substrate / cos_s,
        };

        // Superstrate admittance
        let eta_0 = match pol {
            Polarization::TE | Polarization::Unpolarized => {
                self.superstrate * Complex64::new(cos_0, 0.0)
            }
            Polarization::TM => self.superstrate / Complex64::new(cos_0, 0.0),
        };

        // r = (eta_0 * M11 + eta_0*eta_s*M12 - M21 - eta_s*M22) /
        //     (eta_0 * M11 + eta_0*eta_s*M12 + M21 + eta_s*M22)
        let r_num = eta_0 * m[0][0] + eta_0 * eta_s * m[0][1] - m[1][0] - eta_s * m[1][1];
        let r_den = eta_0 * m[0][0] + eta_0 * eta_s * m[0][1] + m[1][0] + eta_s * m[1][1];

        let r = r_num / r_den;
        let t = Complex64::new(2.0, 0.0) * eta_0 / r_den;

        (r, t)
    }

    /// Compute forward and backward propagating field amplitudes at a given depth.
    fn field_at_depth(
        &self,
        wavelength_nm: f64,
        z: f64,
        k0: f64,
    ) -> (Complex64, Complex64) {
        // Find which layer contains this z-position
        let mut depth = 0.0;
        let mut layer_idx = None;
        let mut z_in_layer = z;

        for (i, layer) in self.layers.iter().enumerate() {
            if z >= depth && z < depth + layer.thickness_nm {
                layer_idx = Some(i);
                z_in_layer = z - depth;
                break;
            }
            depth += layer.thickness_nm;
        }

        let (r_top, _) = self.transfer_matrix(wavelength_nm, 0.0, Polarization::Unpolarized);

        match layer_idx {
            Some(idx) => {
                let n = self.layers[idx].n;
                let phase = k0 * n * z_in_layer;
                let forward = Complex::from_polar(1.0, -phase.re) * (-phase.im).exp();
                let backward = r_top * Complex::from_polar(1.0, phase.re) * (-phase.im).exp();
                (forward, backward)
            }
            None => {
                // In substrate or above stack
                (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))
            }
        }
    }
}

fn mat2_mul(
    a: &[[Complex64; 2]; 2],
    b: &[[Complex64; 2]; 2],
) -> [[Complex64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

impl Default for FilmStack {
    fn default() -> Self {
        // Default VUV stack: resist on silicon
        Self::new_vuv(
            vec![FilmLayer {
                name: "resist".to_string(),
                thickness_nm: 150.0,
                n: Complex64::new(1.65, 0.015),
            }],
            Complex64::new(0.88, 2.10), // Si at 157nm
        )
    }
}

fn serialize_complex<S>(c: &Complex64, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeTuple;
    let mut tup = serializer.serialize_tuple(2)?;
    tup.serialize_element(&c.re)?;
    tup.serialize_element(&c.im)?;
    tup.end()
}

fn deserialize_complex<'de, D>(deserializer: D) -> std::result::Result<Complex64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let (re, im) = <(f64, f64)>::deserialize(deserializer)?;
    Ok(Complex64::new(re, im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bare_substrate_reflectance() {
        let stack = FilmStack::new_vuv(vec![], Complex64::new(1.5, 0.0));
        let r = stack.reflectance(157.0, Polarization::Unpolarized);
        // Fresnel: R = ((n1-n2)/(n1+n2))^2 = ((1-1.5)/(1+1.5))^2 = 0.04
        assert_relative_eq!(r, 0.04, epsilon = 0.001);
    }

    #[test]
    fn test_quarter_wave_ar() {
        // Quarter-wave AR coating: n_film = sqrt(n_substrate), thickness = lambda/(4*n_film)
        let n_sub = 1.5_f64;
        let n_film = n_sub.sqrt();
        let wavelength = 157.0;
        let thickness = wavelength / (4.0 * n_film);

        let stack = FilmStack::new_vuv(
            vec![FilmLayer {
                name: "AR".to_string(),
                thickness_nm: thickness,
                n: Complex64::new(n_film, 0.0),
            }],
            Complex64::new(n_sub, 0.0),
        );
        let r = stack.reflectance(wavelength, Polarization::TE);
        // Should be near zero for ideal AR
        assert!(r < 0.001, "Quarter-wave AR reflectance should be near zero, got {}", r);
    }

    #[test]
    fn test_quarter_wave_is_minimum() {
        let n_sub = 1.5_f64;
        let n_film = n_sub.sqrt();
        let wavelength = 157.0;
        let thickness_qw = wavelength / (4.0 * n_film);

        let stack_qw = FilmStack::new_vuv(
            vec![FilmLayer {
                name: "AR".to_string(),
                thickness_nm: thickness_qw,
                n: Complex64::new(n_film, 0.0),
            }],
            Complex64::new(n_sub, 0.0),
        );

        let stack_off = FilmStack::new_vuv(
            vec![FilmLayer {
                name: "AR".to_string(),
                thickness_nm: thickness_qw * 1.3,
                n: Complex64::new(n_film, 0.0),
            }],
            Complex64::new(n_sub, 0.0),
        );

        let r_qw = stack_qw.reflectance(wavelength, Polarization::TE);
        let r_off = stack_off.reflectance(wavelength, Polarization::TE);
        assert!(r_qw < r_off);
    }

    #[test]
    fn test_standing_wave_not_empty() {
        let stack = FilmStack::default();
        let z_points: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();
        let sw = stack.standing_wave(157.0, &z_points);
        assert_eq!(sw.len(), 100);
        assert!(sw.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_brewster_angle() {
        // At Brewster's angle, TM reflectance should be zero for dielectric
        let n_sub = 1.5;
        let stack = FilmStack::new_vuv(vec![], Complex64::new(n_sub, 0.0));
        let brewster = (n_sub / 1.0).atan(); // atan(n2/n1)
        let r_tm = stack.reflectance_at_angle(157.0, brewster, Polarization::TM);
        assert!(r_tm < 0.001, "TM reflectance at Brewster's angle should be near zero, got {}", r_tm);
    }
}
