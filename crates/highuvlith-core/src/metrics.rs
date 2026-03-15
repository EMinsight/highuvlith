use ndarray::Array2;

/// Measure critical dimension (CD) from an aerial image cross-section.
/// Uses the threshold crossing method at the specified intensity threshold.
///
/// Returns the width (in nm) of the feature above/below threshold.
/// For bright-field L/S patterns, this measures the space width.
pub fn measure_cd(
    intensity_profile: &[f64],
    x_nm: &[f64],
    threshold: f64,
) -> Option<f64> {
    if intensity_profile.len() != x_nm.len() || intensity_profile.len() < 2 {
        return None;
    }

    // Find threshold crossings
    let mut crossings = Vec::new();
    for i in 0..intensity_profile.len() - 1 {
        let y0 = intensity_profile[i] - threshold;
        let y1 = intensity_profile[i + 1] - threshold;

        if y0 * y1 < 0.0 {
            // Linear interpolation for crossing position
            let t = y0 / (y0 - y1);
            let x_cross = x_nm[i] + t * (x_nm[i + 1] - x_nm[i]);
            crossings.push(x_cross);
        }
    }

    if crossings.len() >= 2 {
        // Return the width between the two innermost crossings (nearest to center)
        crossings.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find the pair closest to the center of the field
        let center = (x_nm[0] + x_nm[x_nm.len() - 1]) / 2.0;
        let mut best_pair = (0, 1);
        let mut best_dist = f64::INFINITY;

        for i in 0..crossings.len() - 1 {
            let mid = (crossings[i] + crossings[i + 1]) / 2.0;
            let dist = (mid - center).abs();
            if dist < best_dist {
                best_dist = dist;
                best_pair = (i, i + 1);
            }
        }

        Some((crossings[best_pair.1] - crossings[best_pair.0]).abs())
    } else {
        None
    }
}

/// Measure CD from a 2D aerial image at y=0 cross-section.
pub fn measure_cd_2d(
    image: &Array2<f64>,
    x_min_nm: f64,
    x_max_nm: f64,
    threshold: f64,
) -> Option<f64> {
    let nx = image.ncols();
    let ny = image.nrows();
    let center_row = ny / 2;
    let pixel = (x_max_nm - x_min_nm) / nx as f64;

    let x_nm: Vec<f64> = (0..nx)
        .map(|j| x_min_nm + (j as f64 + 0.5) * pixel)
        .collect();

    let profile: Vec<f64> = (0..nx).map(|j| image[[center_row, j]]).collect();

    measure_cd(&profile, &x_nm, threshold)
}

/// Normalized Image Log-Slope (NILS) at a feature edge.
///
/// NILS = (d/dx ln(I)) * w = (w / I) * (dI/dx) at the edge
///
/// Higher NILS indicates better image quality and process latitude.
pub fn nils(
    intensity_profile: &[f64],
    x_nm: &[f64],
    threshold: f64,
) -> Option<f64> {
    if intensity_profile.len() != x_nm.len() || intensity_profile.len() < 3 {
        return None;
    }

    // Find the threshold crossing nearest to center
    let center = (x_nm[0] + x_nm[x_nm.len() - 1]) / 2.0;
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;

    for i in 0..intensity_profile.len() - 1 {
        let y0 = intensity_profile[i] - threshold;
        let y1 = intensity_profile[i + 1] - threshold;
        if y0 * y1 < 0.0 {
            let t = y0 / (y0 - y1);
            let x_cross = x_nm[i] + t * (x_nm[i + 1] - x_nm[i]);
            let dist = (x_cross - center).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
    }

    if best_idx == 0 || best_idx >= intensity_profile.len() - 1 {
        return None;
    }

    // Compute derivative at edge using central difference
    let dx = x_nm[best_idx + 1] - x_nm[best_idx];
    let di = intensity_profile[best_idx + 1] - intensity_profile[best_idx];
    let i_at_edge = threshold;

    if i_at_edge.abs() < 1e-15 {
        return None;
    }

    // Get feature width
    let cd = measure_cd(intensity_profile, x_nm, threshold)?;

    // NILS = w * |dI/dx| / I_threshold
    Some(cd * (di / dx).abs() / i_at_edge)
}

/// Image contrast (modulation): (Imax - Imin) / (Imax + Imin).
pub fn image_contrast(image: &Array2<f64>) -> f64 {
    let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = image.iter().cloned().fold(f64::INFINITY, f64::min);

    if max + min > 0.0 {
        (max - min) / (max + min)
    } else {
        0.0
    }
}

/// Modulation Transfer Function (MTF) at a given spatial frequency.
/// Computed from the aerial image of a line/space pattern at that pitch.
pub fn mtf_from_image(image: &Array2<f64>) -> f64 {
    // Use the center row
    let ny = image.nrows();
    let nx = image.ncols();
    let center = ny / 2;

    let row: Vec<f64> = (0..nx).map(|j| image[[center, j]]).collect();
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = row.iter().cloned().fold(f64::INFINITY, f64::min);

    if max + min > 0.0 {
        (max - min) / (max + min)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cd_measurement_simple() {
        // Simple step function: 0 from -100 to -25, 1 from -25 to 25, 0 from 25 to 100
        let n = 200;
        let x_nm: Vec<f64> = (0..n).map(|i| -100.0 + i as f64).collect();
        let intensity: Vec<f64> = x_nm
            .iter()
            .map(|&x| if x.abs() < 25.0 { 1.0 } else { 0.0 })
            .collect();

        let cd = measure_cd(&intensity, &x_nm, 0.5).unwrap();
        assert_relative_eq!(cd, 50.0, epsilon = 2.0);
    }

    #[test]
    fn test_contrast_full() {
        let image = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        assert_relative_eq!(image_contrast(&image), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_contrast_zero() {
        let image = Array2::from_elem((4, 4), 0.5);
        assert_relative_eq!(image_contrast(&image), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nils_positive() {
        // Smooth feature (Gaussian bump centered at 0)
        let n = 200;
        let x_nm: Vec<f64> = (0..n).map(|i| -100.0 + i as f64).collect();
        let sigma = 25.0;
        let intensity: Vec<f64> = x_nm
            .iter()
            .map(|&x| (-x * x / (2.0 * sigma * sigma)).exp())
            .collect();

        let nils_val = nils(&intensity, &x_nm, 0.5);
        assert!(nils_val.is_some());
        assert!(nils_val.unwrap() > 0.0);
    }
}
