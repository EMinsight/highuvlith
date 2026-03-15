use ndarray::Array2;

/// Bilinear interpolation on a 2D grid.
pub fn bilinear(
    grid: &Array2<f64>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x: f64,
    y: f64,
) -> f64 {
    let (ny, nx) = grid.dim();
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    // Continuous index
    let fi = (x - x_min) / dx - 0.5;
    let fj = (y - y_min) / dy - 0.5;

    let i0 = fi.floor() as i64;
    let j0 = fj.floor() as i64;

    let t = fi - i0 as f64;
    let u = fj - j0 as f64;

    let get = |i: i64, j: i64| -> f64 {
        let ic = i.clamp(0, nx as i64 - 1) as usize;
        let jc = j.clamp(0, ny as i64 - 1) as usize;
        grid[[jc, ic]]
    };

    (1.0 - t) * (1.0 - u) * get(i0, j0)
        + t * (1.0 - u) * get(i0 + 1, j0)
        + (1.0 - t) * u * get(i0, j0 + 1)
        + t * u * get(i0 + 1, j0 + 1)
}

/// Extract a 1D cross-section along x at a given y coordinate.
pub fn cross_section_x(
    grid: &Array2<f64>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    y: f64,
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let step = (x_max - x_min) / (n_points - 1) as f64;
    let xs: Vec<f64> = (0..n_points).map(|i| x_min + i as f64 * step).collect();
    let vals: Vec<f64> = xs
        .iter()
        .map(|&x| bilinear(grid, x_min, x_max, y_min, y_max, x, y))
        .collect();
    (xs, vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bilinear_center() {
        // 2x2 grid: [[1, 2], [3, 4]]
        let grid = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let val = bilinear(&grid, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0);
        assert_relative_eq!(val, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_bilinear_corner() {
        let grid = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // Near top-left corner
        let val = bilinear(&grid, 0.0, 2.0, 0.0, 2.0, 0.5, 0.5);
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
    }
}
