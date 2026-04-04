/// Evaluate Zernike polynomials using Fringe (University of Arizona) indexing.
///
/// The Fringe indexing maps a single index j to (n, m) radial and azimuthal orders.
/// Convert Fringe index to (n, m) pair.
pub fn fringe_to_nm(j: usize) -> (usize, i32) {
    // Fringe indexing: j = 1, 2, 3, ...
    // Standard mapping tables for common indices
    match j {
        1 => (0, 0),   // Piston
        2 => (1, 1),   // Tilt X
        3 => (1, -1),  // Tilt Y
        4 => (2, 0),   // Defocus
        5 => (2, 2),   // Astigmatism 0
        6 => (2, -2),  // Astigmatism 45
        7 => (3, 1),   // Coma X
        8 => (3, -1),  // Coma Y
        9 => (4, 0),   // Spherical
        10 => (3, 3),  // Trefoil X
        11 => (3, -3), // Trefoil Y
        12 => (4, 2),  // Secondary astigmatism 0
        13 => (4, -2), // Secondary astigmatism 45
        14 => (5, 1),  // Secondary coma X
        15 => (5, -1), // Secondary coma Y
        16 => (6, 0),  // Secondary spherical
        17 => (4, 4),
        18 => (4, -4),
        19 => (5, 3),
        20 => (5, -3),
        21 => (6, 2),
        22 => (6, -2),
        23 => (7, 1),
        24 => (7, -1),
        25 => (8, 0),
        26 => (5, 5),
        27 => (5, -5),
        28 => (6, 4),
        29 => (6, -4),
        30 => (7, 3),
        31 => (7, -3),
        32 => (8, 2),
        33 => (8, -2),
        34 => (9, 1),
        35 => (9, -1),
        36 => (10, 0),
        37 => (12, 0),
        38 => (8, 6),
        39 => (8, -6),
        40 => (10, 0),
        41 => (9, 1),
        42 => (9, -1),
        43 => (9, 3),
        44 => (9, -3),
        45 => (9, 5),
        46 => (9, -5),
        47 => (9, 7),
        48 => (9, -7),
        _ => {
            // Approximate fallback for indices beyond the lookup table (j > 48).
            // This uses a rough sqrt-based estimate for the radial order n and
            // sets m = 0. It is NOT accurate for specific aberration terms;
            // extend the table above if precise high-order terms are needed.
            let n_approx = ((j as f64).sqrt() as usize).max(1);
            (n_approx, 0)
        }
    }
}

/// Evaluate the radial polynomial R_n^m(rho).
pub fn radial_polynomial(n: usize, m_abs: usize, rho: f64) -> f64 {
    if !(n - m_abs).is_multiple_of(2) {
        return 0.0;
    }

    let mut sum = 0.0;
    let s_max = (n - m_abs) / 2;

    for s in 0..=s_max {
        let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
        let num = factorial(n - s);
        let den = factorial(s) * factorial((n + m_abs) / 2 - s) * factorial((n - m_abs) / 2 - s);
        sum += sign * (num as f64 / den as f64) * rho.powi((n - 2 * s) as i32);
    }

    sum
}

/// Evaluate Zernike polynomial Z_j at polar coordinates (rho, theta).
/// Returns 0 if rho > 1 (outside unit circle).
pub fn zernike(j: usize, rho: f64, theta: f64) -> f64 {
    if rho > 1.0 {
        return 0.0;
    }

    let (n, m) = fringe_to_nm(j);
    let m_abs = m.unsigned_abs() as usize;
    let r = radial_polynomial(n, m_abs, rho);

    if m >= 0 {
        r * (m_abs as f64 * theta).cos()
    } else {
        r * (m_abs as f64 * theta).sin()
    }
}

/// Evaluate pupil phase from a set of Zernike coefficients.
/// coefficients: Vec of (fringe_index, coefficient_in_waves).
/// Returns phase in radians.
pub fn pupil_phase(coefficients: &[(usize, f64)], rho: f64, theta: f64) -> f64 {
    let mut phase = 0.0;
    for &(j, coeff) in coefficients {
        phase += coeff * zernike(j, rho, theta);
    }
    phase * 2.0 * std::f64::consts::PI // convert waves to radians
}

fn factorial(n: usize) -> u64 {
    (1..=n as u64).product::<u64>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_piston() {
        // Z1 = 1 everywhere inside unit circle
        assert_relative_eq!(zernike(1, 0.5, 0.0), 1.0);
        assert_relative_eq!(zernike(1, 0.0, 0.0), 1.0);
        assert_relative_eq!(zernike(1, 1.0, 1.23), 1.0);
    }

    #[test]
    fn test_tilt_x() {
        // Z2 = rho * cos(theta)
        assert_relative_eq!(zernike(2, 0.5, 0.0), 0.5);
        assert_relative_eq!(
            zernike(2, 1.0, std::f64::consts::PI / 2.0),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_defocus() {
        // Z4 = 2*rho^2 - 1
        assert_relative_eq!(zernike(4, 0.0, 0.0), -1.0);
        assert_relative_eq!(zernike(4, 1.0, 0.0), 1.0);
        let rho_half = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(zernike(4, rho_half, 0.0), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_spherical() {
        // Z9 = 6*rho^4 - 6*rho^2 + 1
        assert_relative_eq!(zernike(9, 0.0, 0.0), 1.0);
        assert_relative_eq!(zernike(9, 1.0, 0.0), 1.0);
    }

    #[test]
    fn test_outside_unit_circle() {
        assert_relative_eq!(zernike(4, 1.5, 0.0), 0.0);
    }

    #[test]
    fn test_fringe_to_nm_extended_table() {
        // Fringe index 38 maps to (n=8, m=6)
        assert_eq!(fringe_to_nm(38), (8, 6));
        // Fringe index 48 maps to (n=9, m=-7)
        assert_eq!(fringe_to_nm(48), (9, -7));
        // Fringe index 43 maps to (n=9, m=3)
        assert_eq!(fringe_to_nm(43), (9, 3));
    }

    #[test]
    fn test_astigmatism_z5() {
        // Z5 = rho^2 * cos(2*theta), astigmatism 0-degree
        // At 45 degrees (pi/4), cos(2*pi/4) = cos(pi/2) = 0
        let val_45 = zernike(5, 0.5, std::f64::consts::PI / 4.0);
        assert_relative_eq!(val_45, 0.0, epsilon = 1e-14);

        // At 0 degrees, should be rho^2 * cos(0) = rho^2
        let val_0 = zernike(5, 0.5, 0.0);
        assert_relative_eq!(val_0, 0.25, epsilon = 1e-14);

        // At 22.5 degrees, cos(2*22.5deg) = cos(45deg) != 0, so nonzero
        let val_225 = zernike(5, 0.5, std::f64::consts::PI / 8.0);
        assert!(val_225.abs() > 0.01, "Z5 at 22.5deg should be nonzero");
    }

    #[test]
    fn test_zernike_at_origin() {
        // At rho=0, only piston (Z1, n=0,m=0) contributes
        // All other Zernike polynomials should be zero at origin
        // (since they have rho^n factor with n>=1)
        assert_relative_eq!(zernike(1, 0.0, 0.0), 1.0); // piston
        assert_relative_eq!(zernike(2, 0.0, 0.0), 0.0, epsilon = 1e-15); // tilt X
        assert_relative_eq!(zernike(3, 0.0, 0.0), 0.0, epsilon = 1e-15); // tilt Y
                                                                         // Defocus Z4 = 2*rho^2 - 1 => at rho=0: -1.0
        assert_relative_eq!(zernike(4, 0.0, 0.0), -1.0, epsilon = 1e-15);
        assert_relative_eq!(zernike(5, 0.0, 0.0), 0.0, epsilon = 1e-15); // astigmatism
        assert_relative_eq!(zernike(7, 0.0, 0.0), 0.0, epsilon = 1e-15); // coma
                                                                         // Spherical Z9 = 6*rho^4 - 6*rho^2 + 1 => at rho=0: 1.0
        assert_relative_eq!(zernike(9, 0.0, 0.0), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_pupil_phase_multiple_coefficients() {
        // With both defocus (Z4) and astigmatism (Z5), phase should combine
        let coeffs = vec![(4, 0.1), (5, 0.05)]; // defocus + astigmatism
        let phase = pupil_phase(&coeffs, 0.5, 0.0);

        // Individual phases
        let phase_defocus = pupil_phase(&[(4, 0.1)], 0.5, 0.0);
        let phase_astig = pupil_phase(&[(5, 0.05)], 0.5, 0.0);

        // Combined should equal sum
        assert_relative_eq!(phase, phase_defocus + phase_astig, epsilon = 1e-14);

        // Phase should be nonzero
        assert!(phase.abs() > 1e-6, "Combined phase should be nonzero");
    }
}
