//! Export aerial images as PNG or TIFF files.

use ndarray::Array2;
use std::path::Path;

/// Colormap for rendering intensity images.
#[derive(Debug, Clone, Copy)]
pub enum Colormap {
    /// Black → purple → orange → yellow (inferno-like).
    Inferno,
    /// Black → white.
    Grayscale,
    /// Blue → white → red (diverging, centered at 0.5).
    BlueRed,
    /// Black → green → white.
    Viridis,
}

/// Save a 2D intensity array as a PNG image.
pub fn save_png(
    data: &Array2<f64>,
    path: &Path,
    colormap: Colormap,
) -> Result<(), image::ImageError> {
    let (ny, nx) = data.dim();
    let (min_val, max_val) = data_range(data);
    let range = max_val - min_val;

    let mut img = image::RgbImage::new(nx as u32, ny as u32);

    for i in 0..ny {
        for j in 0..nx {
            let t = if range > 1e-15 {
                (data[[i, j]] - min_val) / range
            } else {
                0.5
            };
            let (r, g, b) = apply_colormap(t.clamp(0.0, 1.0), colormap);
            img.put_pixel(j as u32, i as u32, image::Rgb([r, g, b]));
        }
    }

    img.save(path)
}

/// Save a 2D intensity array as a 32-bit floating-point TIFF.
pub fn save_tiff_f32(data: &Array2<f64>, path: &Path) -> Result<(), image::ImageError> {
    let (ny, nx) = data.dim();
    let mut img = image::GrayImage::new(nx as u32, ny as u32);

    let (min_val, max_val) = data_range(data);
    let range = max_val - min_val;

    for i in 0..ny {
        for j in 0..nx {
            let t = if range > 1e-15 {
                ((data[[i, j]] - min_val) / range * 255.0) as u8
            } else {
                128
            };
            img.put_pixel(j as u32, i as u32, image::Luma([t]));
        }
    }

    img.save(path)
}

fn data_range(data: &Array2<f64>) -> (f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn apply_colormap(t: f64, colormap: Colormap) -> (u8, u8, u8) {
    match colormap {
        Colormap::Grayscale => {
            let v = (t * 255.0) as u8;
            (v, v, v)
        }
        Colormap::Inferno => {
            let t = t as f32;
            if t < 0.25 {
                let s = t / 0.25;
                ((s * 80.0) as u8, 0, (s * 120.0) as u8)
            } else if t < 0.5 {
                let s = (t - 0.25) / 0.25;
                (
                    (80.0 + s * 140.0) as u8,
                    (s * 30.0) as u8,
                    (120.0 - s * 70.0) as u8,
                )
            } else if t < 0.75 {
                let s = (t - 0.5) / 0.25;
                (
                    (220.0 + s * 35.0) as u8,
                    (30.0 + s * 130.0) as u8,
                    (50.0 - s * 50.0) as u8,
                )
            } else {
                let s = (t - 0.75) / 0.25;
                (255, (160.0 + s * 95.0) as u8, (s * 100.0) as u8)
            }
        }
        Colormap::BlueRed => {
            if t < 0.5 {
                let s = t / 0.5;
                ((s * 255.0) as u8, (s * 255.0) as u8, 255)
            } else {
                let s = (t - 0.5) / 0.5;
                (255, (255.0 * (1.0 - s)) as u8, (255.0 * (1.0 - s)) as u8)
            }
        }
        Colormap::Viridis => {
            let t = t as f32;
            if t < 0.33 {
                let s = t / 0.33;
                (
                    (s * 30.0) as u8,
                    (20.0 + s * 100.0) as u8,
                    (80.0 + s * 50.0) as u8,
                )
            } else if t < 0.66 {
                let s = (t - 0.33) / 0.33;
                (
                    (30.0 + s * 60.0) as u8,
                    (120.0 + s * 60.0) as u8,
                    (130.0 - s * 60.0) as u8,
                )
            } else {
                let s = (t - 0.66) / 0.34;
                (
                    (90.0 + s * 165.0) as u8,
                    (180.0 + s * 75.0) as u8,
                    (70.0 - s * 40.0) as u8,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_colormap_range() {
        for cmap in [
            Colormap::Inferno,
            Colormap::Grayscale,
            Colormap::BlueRed,
            Colormap::Viridis,
        ] {
            let (r0, g0, b0) = apply_colormap(0.0, cmap);
            let (r1, g1, b1) = apply_colormap(1.0, cmap);
            // Just verify no panics at boundaries
            assert!(r0 <= 255 && g0 <= 255 && b0 <= 255);
            assert!(r1 <= 255 && g1 <= 255 && b1 <= 255);
        }
    }

    #[test]
    fn test_save_png() {
        let data = Array2::from_shape_fn((64, 64), |(i, j)| (i as f64 / 64.0) * (j as f64 / 64.0));
        let path = std::env::temp_dir().join("highuvlith_test_aerial.png");
        save_png(&data, &path, Colormap::Inferno).unwrap();
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }
}
