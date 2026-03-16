use crate::config::SimConfig;
use highuvlith_core::aerial::AerialImageEngine;
use highuvlith_core::metrics;
use std::path::Path;

pub fn run(
    config_path: &Path,
    output: Option<&Path>,
    focus_override: Option<f64>,
    _dose_override: Option<f64>,
) -> anyhow::Result<()> {
    let config = SimConfig::load(config_path)?;
    let source = config.to_source();
    let optics = config.to_optics();
    let mask = config.to_mask();
    let grid = config.to_grid()?;

    let focus = focus_override.unwrap_or(config.process.focus_nm);

    eprintln!("Source:  λ = {:.2} nm, σ = {:.2}", source.wavelength_nm,
        match &source.illumination {
            highuvlith_core::source::IlluminationShape::Conventional { sigma } => *sigma,
            _ => 0.0,
        });
    eprintln!("Optics:  NA = {:.2}", optics.na);
    eprintln!("Mask:    CD = {:.1} nm, pitch = {:.1} nm", config.mask.cd_nm, config.mask.pitch_nm);
    eprintln!("Grid:    {}×{}, pixel = {:.1} nm", grid.size, grid.size, grid.pixel_nm);
    eprintln!("Focus:   {:.1} nm", focus);

    let engine = AerialImageEngine::new(&source, &optics, grid.clone(), 20)?;
    eprintln!("Engine:  {} SOCS kernels", engine.num_kernels());

    let start = std::time::Instant::now();
    let aerial = engine.compute(&mask, focus);
    let elapsed = start.elapsed();

    let contrast = metrics::image_contrast(&aerial.data);
    let max_i = aerial.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_i = aerial.data.iter().cloned().fold(f64::INFINITY, f64::min);

    eprintln!("Compute: {:.2?}", elapsed);
    eprintln!("Results:");
    eprintln!("  Contrast:  {:.4}", contrast);
    eprintln!("  I_max:     {:.4}", max_i);
    eprintln!("  I_min:     {:.4}", min_i);

    if let Some(out_path) = output {
        let result = serde_json::json!({
            "wavelength_nm": source.wavelength_nm,
            "na": optics.na,
            "cd_nm": config.mask.cd_nm,
            "pitch_nm": config.mask.pitch_nm,
            "focus_nm": focus,
            "grid_size": grid.size,
            "pixel_nm": grid.pixel_nm,
            "contrast": contrast,
            "i_max": max_i,
            "i_min": min_i,
            "compute_ms": elapsed.as_secs_f64() * 1000.0,
            "num_kernels": engine.num_kernels(),
        });
        if out_path.extension().map_or(false, |ext| ext == "png") {
            use highuvlith_core::io::image_export::{save_png, Colormap};
            save_png(&aerial.data, out_path, Colormap::Inferno)
                .map_err(|e| anyhow::anyhow!("Failed to save PNG: {}", e))?;
        } else {
            std::fs::write(out_path, serde_json::to_string_pretty(&result)?)?;
        }
        eprintln!("Output written to {}", out_path.display());
    }

    Ok(())
}
