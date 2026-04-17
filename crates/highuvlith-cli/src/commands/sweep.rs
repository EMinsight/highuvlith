use crate::config::SimConfig;
use highuvlith_core::aerial::AerialImageEngine;
use highuvlith_core::metrics;
use highuvlith_core::source::LithographySource;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

fn parse_range(s: &str) -> anyhow::Result<Vec<f64>> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!("Range must be 'start,stop,steps' (e.g., '-200,200,21')");
    }
    let start: f64 = parts[0].parse()?;
    let stop: f64 = parts[1].parse()?;
    let steps: usize = parts[2].parse()?;
    if steps > 1 && start > stop {
        anyhow::bail!(
            "Invalid range: start ({}) > stop ({}) with steps = {}",
            start,
            stop,
            steps
        );
    }
    if steps < 2 {
        return Ok(vec![start]);
    }
    let step = (stop - start) / (steps - 1) as f64;
    Ok((0..steps).map(|i| start + i as f64 * step).collect())
}

pub fn run(
    config_path: &Path,
    output: Option<&Path>,
    focus_range: &str,
    dose_range: Option<&str>,
) -> anyhow::Result<()> {
    let config = SimConfig::load(config_path)?;
    config.validate()?;
    let source = config.to_source();
    let optics = config.to_optics()?;
    let mask = config.to_mask()?;
    let grid = config.to_grid()?;

    let focuses = parse_range(focus_range)?;
    let doses = if let Some(dr) = dose_range {
        parse_range(dr)?
    } else {
        vec![config.process.dose_mj_cm2]
    };

    let total = focuses.len() * doses.len();
    eprintln!(
        "Sweep: {} focuses × {} doses = {} points",
        focuses.len(),
        doses.len(),
        total
    );

    let engine = AerialImageEngine::new(&source, &optics, grid.clone(), 20)?;
    eprintln!("Engine: {} SOCS kernels", engine.num_kernels());

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let field = grid.field_size_nm();
    let half = field / 2.0;

    let mut results = Vec::with_capacity(total);
    let start = std::time::Instant::now();

    for &dose in &doses {
        for &focus in &focuses {
            let aerial = engine.compute(&mask, focus);
            let contrast = metrics::image_contrast(&aerial.data);
            let cd = metrics::measure_cd_2d(&aerial.data, -half, half, 0.3);

            results.push(serde_json::json!({
                "dose_mj_cm2": dose,
                "focus_nm": focus,
                "contrast": contrast,
                "cd_nm": cd,
            }));
            pb.inc(1);
        }
    }

    pb.finish_with_message("Sweep complete");
    let elapsed = start.elapsed();
    eprintln!(
        "Total time: {:.2?} ({:.1?} per point)",
        elapsed,
        elapsed / total as u32
    );

    if let Some(out_path) = output {
        let output_json = serde_json::json!({
            "config": {
                "wavelength_nm": source.wavelength_nm(),
                "source_type": source.kind_label(),
                "na": optics.na,
                "cd_nm": config.mask.cd_nm,
                "pitch_nm": config.mask.pitch_nm,
            },
            "focuses": focuses,
            "doses": doses,
            "results": results,
        });
        std::fs::write(out_path, serde_json::to_string_pretty(&output_json)?)?;
        eprintln!("Output written to {}", out_path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_range_valid() {
        let vals = parse_range("-200,200,5").unwrap();
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - (-200.0)).abs() < 1e-10);
        assert!((vals[4] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_range_reversed_error() {
        let result = parse_range("200,-200,5");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("start"), "Error should mention start: {}", msg);
    }

    #[test]
    fn test_parse_range_single_step() {
        let vals = parse_range("100,200,1").unwrap();
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_range_bad_format() {
        assert!(parse_range("1,2").is_err());
        assert!(parse_range("1,2,3,4").is_err());
    }
}
