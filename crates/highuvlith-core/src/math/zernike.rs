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
        _ => {
            // For higher indices, use a general formula
            // This is an approximation; for production use, extend the table
            let n_approx = ((j as f64).sqrt() as usize).max(1);
            (n_approx, 0)
        }
    }
}

/// Evaluate the radial polynomial R_n^m(rho).
pub fn radial_polynomial(n: usize, m_abs: usize, rho: f64) -> f64 {
    if (n - m_abs) % 2 != 0 {
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
pub fn pupil_phase(
    coefficients: &[(usize, f64)],
    rho: f64,
    theta: f64,
) -> f64 {
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
}
