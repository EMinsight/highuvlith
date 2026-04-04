use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Development rate model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevelopmentModel {
    /// Simple threshold model: developed if PAC < threshold.
    Threshold { threshold: f64 },
    /// Mack development model:
    /// R(m) = Rmax * (a + 1) * (1 - m)^n / (a + (1-m)^n) + Rmin
    /// where m = normalized PAC concentration, a = (n+1)/(n-1) * (1 - mth)^n
    Mack {
        /// Maximum development rate (nm/s).
        rmax: f64,
        /// Minimum development rate (nm/s).
        rmin: f64,
        /// Threshold PAC concentration.
        mth: f64,
        /// Development selectivity.
        n: f64,
    },
}

impl Default for DevelopmentModel {
    fn default() -> Self {
        Self::Mack {
            rmax: 100.0,
            rmin: 0.1,
            mth: 0.5,
            n: 3.0,
        }
    }
}

/// Photoresist parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResistParams {
    /// Resist thickness in nm.
    pub thickness_nm: f64,
    /// Dill A parameter: bleachable absorption (1/um).
    /// VUV fluoropolymer resists: ~0.1-0.3 /um (much lower than DUV).
    pub dill_a: f64,
    /// Dill B parameter: non-bleachable absorption (1/um).
    /// VUV: ~0.3-0.6 /um.
    pub dill_b: f64,
    /// Dill C parameter: exposure rate constant (cm^2/mJ).
    pub dill_c: f64,
    /// Post-exposure bake (PEB) diffusion length in nm.
    pub peb_diffusion_nm: f64,
    /// Development model.
    pub development: DevelopmentModel,
}

impl ResistParams {
    /// Default VUV fluoropolymer resist parameters.
    pub fn vuv_fluoropolymer() -> Self {
        Self {
            thickness_nm: 150.0,
            dill_a: 0.2,  // Low bleachable absorption (fluoropolymer)
            dill_b: 0.45, // Moderate non-bleachable absorption
            dill_c: 0.02,
            peb_diffusion_nm: 30.0,
            development: DevelopmentModel::default(),
        }
    }

    /// Compute absorption coefficient alpha (1/nm) at a given PAC concentration m.
    /// alpha = A*m + B (in 1/um, convert to 1/nm)
    pub fn absorption(&self, m: f64) -> f64 {
        (self.dill_a * m + self.dill_b) * 1e-3 // convert from 1/um to 1/nm
    }
}

impl Default for ResistParams {
    fn default() -> Self {
        Self::vuv_fluoropolymer()
    }
}

/// Latent image: normalized PAC (photo-active compound) concentration after exposure.
/// m = 1.0 means unexposed, m = 0.0 means fully exposed.
pub struct LatentImage {
    /// PAC concentration at each (y, x) grid point (depth-averaged).
    pub pac: Array2<f64>,
}

/// 1D resist profile after development.
pub struct ResistProfile {
    /// x positions in nm.
    pub x_nm: Vec<f64>,
    /// Remaining resist height at each x position.
    pub height_nm: Vec<f64>,
    /// Original resist thickness.
    pub thickness_nm: f64,
}

/// Compute the latent image from an aerial image and exposure dose.
///
/// Uses the Dill exposure model:
///   m(x,y) = exp(-C * dose * I(x,y))
/// where I is the aerial image intensity.
pub fn expose(aerial_image: &Array2<f64>, dose_mj_cm2: f64, params: &ResistParams) -> LatentImage {
    let pac = aerial_image.mapv(|intensity| {
        // Beer-Lambert through resist (depth-averaged approximation)
        let effective_intensity = intensity * effective_coupling(params);
        // Dill first-order model
        (-params.dill_c * dose_mj_cm2 * effective_intensity).exp()
    });

    LatentImage { pac }
}

/// Apply post-exposure bake diffusion (Gaussian blur of latent image).
pub fn peb_diffuse(latent: &mut LatentImage, diffusion_nm: f64, pixel_nm: f64) {
    if diffusion_nm <= 0.0 {
        return;
    }

    let sigma_pixels = diffusion_nm / pixel_nm;
    let kernel_radius = (3.0 * sigma_pixels).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0; kernel_size];
    let mut sum = 0.0;
    for (i, k_val) in kernel.iter_mut().enumerate().take(kernel_size) {
        let d = i as f64 - kernel_radius as f64;
        *k_val = (-d * d / (2.0 * sigma_pixels * sigma_pixels)).exp();
        sum += *k_val;
    }
    for k in &mut kernel {
        *k /= sum;
    }

    let (ny, nx) = latent.pac.dim();

    // Separable convolution: first along x, then along y
    let mut temp = Array2::zeros((ny, nx));

    // Convolve along x
    for i in 0..ny {
        for j in 0..nx {
            let mut val = 0.0;
            for (k, &k_val) in kernel.iter().enumerate().take(kernel_size) {
                let jj = j as i64 + k as i64 - kernel_radius as i64;
                let jj = jj.clamp(0, nx as i64 - 1) as usize;
                val += latent.pac[[i, jj]] * k_val;
            }
            temp[[i, j]] = val;
        }
    }

    // Convolve along y
    for i in 0..ny {
        for j in 0..nx {
            let mut val = 0.0;
            for (k, &k_val) in kernel.iter().enumerate().take(kernel_size) {
                let ii = i as i64 + k as i64 - kernel_radius as i64;
                let ii = ii.clamp(0, ny as i64 - 1) as usize;
                val += temp[[ii, j]] * k_val;
            }
            latent.pac[[i, j]] = val;
        }
    }
}

/// Compute development rate at each point from the latent image.
pub fn development_rate(latent: &LatentImage, params: &ResistParams) -> Array2<f64> {
    latent.pac.mapv(|m| match &params.development {
        DevelopmentModel::Threshold { threshold } => {
            if m < *threshold {
                1000.0 // fast development (exposed)
            } else {
                0.01 // minimal development (unexposed)
            }
        }
        DevelopmentModel::Mack { rmax, rmin, mth, n } => {
            let m_clamped = m.clamp(0.0, 1.0);
            if (n - 1.0).abs() < 1e-12 {
                // Fallback for n=1 singularity: linear interpolation
                rmax * (1.0 - m_clamped) + rmin
            } else {
                let a = (n + 1.0) / (n - 1.0) * (1.0 - mth).powf(*n);
                let one_minus_m_n = (1.0 - m_clamped).powf(*n);
                rmax * (a + 1.0) * one_minus_m_n / (a + one_minus_m_n) + rmin
            }
        }
    })
}

/// Simulate development to extract the resist profile along the x-axis.
/// Uses a simple vertical development model (1D at each x position).
pub fn develop(
    latent: &LatentImage,
    params: &ResistParams,
    dev_time_s: f64,
    pixel_nm: f64,
) -> ResistProfile {
    let rate = development_rate(latent, params);
    let (ny, nx) = rate.dim();
    let center_row = ny / 2;

    let x_nm: Vec<f64> = (0..nx)
        .map(|j| {
            let field = nx as f64 * pixel_nm;
            -field / 2.0 + (j as f64 + 0.5) * pixel_nm
        })
        .collect();

    // Simple vertical development: if average rate * time > thickness, fully developed
    let height_nm: Vec<f64> = (0..nx)
        .map(|j| {
            let r = rate[[center_row, j]];
            let developed = r * dev_time_s;
            (params.thickness_nm - developed).max(0.0)
        })
        .collect();

    ResistProfile {
        x_nm,
        height_nm,
        thickness_nm: params.thickness_nm,
    }
}

/// Effective coupling factor accounting for depth-averaged absorption in resist.
fn effective_coupling(params: &ResistParams) -> f64 {
    let alpha = params.absorption(1.0); // initial absorption (m=1)
    let thickness = params.thickness_nm;

    if alpha * thickness < 0.01 {
        return 1.0; // optically thin
    }

    // Depth-averaged intensity: (1 - exp(-alpha*d)) / (alpha * d)
    (1.0 - (-alpha * thickness).exp()) / (alpha * thickness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_unexposed_pac_is_one() {
        let aerial = Array2::zeros((64, 64)); // zero intensity = no exposure
        let params = ResistParams::vuv_fluoropolymer();
        let latent = expose(&aerial, 30.0, &params);
        for &m in latent.pac.iter() {
            assert_relative_eq!(m, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_high_dose_pac_near_zero() {
        let aerial = Array2::ones((64, 64)); // uniform max intensity
        let params = ResistParams::vuv_fluoropolymer();
        let latent = expose(&aerial, 1000.0, &params); // very high dose
        for &m in latent.pac.iter() {
            assert!(m < 0.01, "High dose should drive PAC near zero");
        }
    }

    #[test]
    fn test_higher_dose_lower_pac() {
        let aerial = Array2::from_elem((64, 64), 0.5);
        let params = ResistParams::vuv_fluoropolymer();
        let latent_low = expose(&aerial, 10.0, &params);
        let latent_high = expose(&aerial, 50.0, &params);
        assert!(
            latent_high.pac[[32, 32]] < latent_low.pac[[32, 32]],
            "Higher dose should give lower PAC"
        );
    }

    #[test]
    fn test_mack_development_rate() {
        let params = ResistParams::vuv_fluoropolymer();

        // Fully exposed (m=0): rate should be near rmax
        let exposed = Array2::from_elem((1, 1), 0.0);
        let latent_exposed = LatentImage { pac: exposed };
        let rate = development_rate(&latent_exposed, &params);
        assert!(rate[[0, 0]] > 50.0);

        // Unexposed (m=1): rate should be near rmin
        let unexposed = Array2::from_elem((1, 1), 1.0);
        let latent_unexposed = LatentImage { pac: unexposed };
        let rate = development_rate(&latent_unexposed, &params);
        assert!(rate[[0, 0]] < 1.0);
    }

    #[test]
    fn test_beer_lambert_coupling() {
        let params = ResistParams {
            thickness_nm: 150.0,
            dill_a: 0.0,
            dill_b: 0.0, // zero absorption
            dill_c: 0.02,
            ..ResistParams::vuv_fluoropolymer()
        };
        let coupling = effective_coupling(&params);
        assert_relative_eq!(coupling, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_peb_diffusion_broadens() {
        let mut pac = Array2::zeros((64, 64));
        pac[[32, 32]] = 0.5; // delta function
        let mut latent = LatentImage { pac };

        let before_max = latent.pac[[32, 32]];
        peb_diffuse(&mut latent, 10.0, 1.0);
        let after_max = latent.pac[[32, 32]];

        assert!(
            after_max < before_max,
            "PEB diffusion should reduce peak concentration"
        );
    }

    #[test]
    fn test_development_profile() {
        let mut pac = Array2::from_elem((64, 64), 0.95); // almost unexposed
                                                         // Create exposed region in center
        for j in 28..36 {
            for i in 0..64 {
                pac[[i, j]] = 0.1;
            }
        }

        let params = ResistParams::vuv_fluoropolymer();
        let latent = LatentImage { pac };
        // Short development time so unexposed region survives
        let profile = develop(&latent, &params, 1.0, 2.0);

        assert_eq!(profile.x_nm.len(), 64);
        // Exposed region should have lower remaining height
        let center_height = profile.height_nm[32];
        let edge_height = profile.height_nm[10];
        assert!(
            center_height < edge_height,
            "Exposed region height ({}) should be less than unexposed ({})",
            center_height,
            edge_height
        );
    }

    #[test]
    fn test_mack_n_near_one_no_panic() {
        let params = ResistParams {
            development: DevelopmentModel::Mack {
                rmax: 100.0,
                rmin: 0.1,
                mth: 0.5,
                n: 1.001, // near-singular n~1
            },
            ..ResistParams::vuv_fluoropolymer()
        };
        let pac = Array2::from_elem((1, 1), 0.5);
        let latent = LatentImage { pac };
        let rate = development_rate(&latent, &params);
        assert!(
            rate[[0, 0]].is_finite(),
            "Rate should be finite for n near 1"
        );
    }

    #[test]
    fn test_expose_zero_dose_pac_one() {
        let aerial = Array2::from_elem((64, 64), 0.8);
        let params = ResistParams::vuv_fluoropolymer();
        let latent = expose(&aerial, 0.0, &params);
        for &m in latent.pac.iter() {
            assert_relative_eq!(m, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_develop_zero_time_full_thickness() {
        let pac = Array2::from_elem((64, 64), 0.1); // fully exposed
        let params = ResistParams::vuv_fluoropolymer();
        let latent = LatentImage { pac };
        let profile = develop(&latent, &params, 0.0, 2.0);
        for &h in &profile.height_nm {
            assert_relative_eq!(h, params.thickness_nm, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_threshold_model_exposed() {
        let params = ResistParams {
            development: DevelopmentModel::Threshold { threshold: 0.5 },
            ..ResistParams::vuv_fluoropolymer()
        };
        // m = 0.3 is below threshold 0.5 => exposed => fast development rate
        let pac = Array2::from_elem((1, 1), 0.3);
        let latent = LatentImage { pac };
        let rate = development_rate(&latent, &params);
        assert!(
            rate[[0, 0]] > 100.0,
            "Exposed region should have high development rate"
        );

        // m = 0.8 is above threshold => unexposed => slow rate
        let pac2 = Array2::from_elem((1, 1), 0.8);
        let latent2 = LatentImage { pac: pac2 };
        let rate2 = development_rate(&latent2, &params);
        assert!(
            rate2[[0, 0]] < 1.0,
            "Unexposed region should have low development rate"
        );
    }
}
