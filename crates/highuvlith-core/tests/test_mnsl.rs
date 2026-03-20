use approx::assert_relative_eq;
use highuvlith_core::{
    mnsl::{MnslConfig, MnslEngine, NanosphereArray, SpherePacking, SubstrateCoupling, simulate_moire_emission},
    types::GridConfig,
};

#[test]
fn test_mnsl_basic_simulation() {
    let config = MnslConfig::default();
    let grid = GridConfig::new(64, 4.0).unwrap();
    let engine = MnslEngine::new(config, grid);

    let result = engine.compute_emission();

    assert!(result.moire_period_nm > 0.0);
    assert!(result.total_emission_power > 0.0);
    assert!(result.peak_enhancement >= 1.0);
    assert_eq!(result.emission_pattern.data.dim(), (64, 64));
}

#[test]
fn test_nanosphere_array_creation() {
    let array = NanosphereArray::silica_spheres(200.0, 300.0);

    assert_eq!(array.diameter_nm, 200.0);
    assert_eq!(array.pitch_nm, 300.0);
    assert_eq!(array.n_real, 1.56);
    assert_eq!(array.packing, SpherePacking::HCP);

    let vf = array.volume_fraction();
    assert!(vf > 0.0 && vf < 1.0);
}

#[test]
fn test_polystyrene_spheres() {
    let array = NanosphereArray::polystyrene_spheres(150.0, 250.0);

    assert_eq!(array.diameter_nm, 150.0);
    assert_eq!(array.pitch_nm, 250.0);
    assert_eq!(array.n_real, 1.65);
    assert_eq!(array.n_imag, 0.015);
}

#[test]
fn test_moire_period_calculation() {
    let mut config = MnslConfig::default();
    config.bottom_array.pitch_nm = 300.0;
    config.top_array.pitch_nm = 300.0;
    config.top_array.orientation_deg = 5.0;

    let grid = GridConfig::new(128, 2.0).unwrap();
    let engine = MnslEngine::new(config, grid);

    let period = engine.calculate_moire_period();

    // For small rotation angles, period ≈ pitch / (2 * sin(angle))
    let expected_period = 300.0 / (2.0 * (5.0 * std::f64::consts::PI / 180.0).sin());
    assert_relative_eq!(period, expected_period, epsilon = 0.1);
}

#[test]
fn test_rotation_transform() {
    let grid = GridConfig::new(64, 2.0).unwrap();
    let config = MnslConfig::default();
    let engine = MnslEngine::new(config, grid);

    let positions = vec![(100.0, 0.0), (0.0, 100.0)];

    // Test 90-degree rotation
    let rotated = engine.apply_rotation_transform(&positions, 90.0);
    assert_relative_eq!(rotated[0].0, 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated[0].1, 100.0, epsilon = 1e-10);
    assert_relative_eq!(rotated[1].0, -100.0, epsilon = 1e-10);
    assert_relative_eq!(rotated[1].1, 0.0, epsilon = 1e-10);

    // Test identity (0-degree rotation)
    let identity = engine.apply_rotation_transform(&positions, 0.0);
    for (orig, rot) in positions.iter().zip(identity.iter()) {
        assert_relative_eq!(orig.0, rot.0, epsilon = 1e-10);
        assert_relative_eq!(orig.1, rot.1, epsilon = 1e-10);
    }
}

#[test]
fn test_substrate_coupling() {
    let coupling = SubstrateCoupling::default();

    assert!(coupling.coupling_strength >= 0.0 && coupling.coupling_strength <= 1.0);
    assert!(coupling.enable_nearfield);
}

#[test]
fn test_convenience_function() {
    let grid = GridConfig::new(64, 4.0).unwrap();
    let result = simulate_moire_emission(200.0, 300.0, 5.0, 100.0, grid);

    assert!(result.moire_period_nm > 0.0);
    assert!(result.peak_enhancement >= 1.0);
    assert_eq!(result.emission_pattern.data.dim(), (64, 64));
}

#[test]
fn test_sphere_packing_volume_fractions() {
    let diameter = 200.0;
    let pitch = 300.0;

    let hcp_array = NanosphereArray {
        diameter_nm: diameter,
        pitch_nm: pitch,
        orientation_deg: 0.0,
        n_real: 1.5,
        n_imag: 0.0,
        packing: SpherePacking::HCP,
    };

    let fcc_array = NanosphereArray {
        packing: SpherePacking::FCC,
        ..hcp_array.clone()
    };

    let cubic_array = NanosphereArray {
        packing: SpherePacking::SimpleCubic,
        ..hcp_array.clone()
    };

    let hcp_vf = hcp_array.volume_fraction();
    let fcc_vf = fcc_array.volume_fraction();
    let cubic_vf = cubic_array.volume_fraction();

    // All should be physical values
    assert!(hcp_vf > 0.0 && hcp_vf < 1.0);
    assert!(fcc_vf > 0.0 && fcc_vf < 1.0);
    assert!(cubic_vf > 0.0 && cubic_vf < 1.0);

    // HCP and FCC should have higher packing efficiency than simple cubic
    assert!(hcp_vf > cubic_vf);
    assert!(fcc_vf > cubic_vf);
}

#[test]
fn test_emission_enhancement_factor() {
    let config = MnslConfig::default();
    let grid = GridConfig::new(64, 4.0).unwrap();
    let engine = MnslEngine::new(config, grid);

    let result = engine.compute_emission();

    // Enhancement factors should be >= 1.0 everywhere
    for &enhancement in result.enhancement_factors.data.iter() {
        assert!(enhancement >= 1.0, "Enhancement factor {} should be >= 1.0", enhancement);
    }

    // There should be some enhancement (not all values equal to 1.0)
    let max_enhancement = result.enhancement_factors.data.iter()
        .cloned()
        .fold(0.0f64, f64::max);
    assert!(max_enhancement > 1.0, "Maximum enhancement should be > 1.0");
}

#[test]
fn test_peak_finding() {
    let grid = GridConfig::new(64, 4.0).unwrap();
    let field_size = grid.field_size_nm();
    let half_field = field_size / 2.0;
    let result = simulate_moire_emission(200.0, 300.0, 5.0, 100.0, grid);

    // Should find some peaks
    assert!(!result.peak_positions.is_empty(), "Should find at least some emission peaks");

    // Peak positions should be within the simulation domain

    for &(x, y) in &result.peak_positions {
        assert!(x >= -half_field && x <= half_field, "Peak x-position {} outside domain", x);
        assert!(y >= -half_field && y <= half_field, "Peak y-position {} outside domain", y);
    }
}

#[test]
fn test_different_sphere_materials() {
    let grid = GridConfig::new(64, 4.0).unwrap();

    // Silica spheres
    let silica_result = simulate_moire_emission(200.0, 300.0, 5.0, 100.0, grid.clone());

    // Create polystyrene configuration manually
    let mut ps_config = MnslConfig::default();
    ps_config.bottom_array = NanosphereArray::polystyrene_spheres(200.0, 300.0);
    ps_config.top_array = NanosphereArray::polystyrene_spheres(200.0, 300.0);
    ps_config.top_array.orientation_deg = 5.0;
    ps_config.separation_nm = 100.0;

    let ps_engine = MnslEngine::new(ps_config, grid.clone());
    let ps_result = ps_engine.compute_emission();

    // Both should produce valid results
    assert!(silica_result.total_emission_power > 0.0);
    assert!(ps_result.total_emission_power > 0.0);

    // Results should be different due to different refractive indices
    assert_ne!(silica_result.total_emission_power, ps_result.total_emission_power);
}

#[test]
fn test_zero_rotation_special_case() {
    let mut config = MnslConfig::default();
    config.top_array.orientation_deg = 0.0; // No rotation

    let grid = GridConfig::new(64, 4.0).unwrap();
    let engine = MnslEngine::new(config, grid);

    let result = engine.compute_emission();

    // Should still produce valid results
    assert!(result.total_emission_power > 0.0);
    assert!(result.peak_enhancement >= 1.0);
}

#[test]
fn test_large_rotation_angle() {
    let mut config = MnslConfig::default();
    config.top_array.orientation_deg = 45.0; // Large rotation

    let grid = GridConfig::new(64, 4.0).unwrap();
    let engine = MnslEngine::new(config, grid);

    let result = engine.compute_emission();

    // Should still produce valid results
    assert!(result.total_emission_power > 0.0);
    assert!(result.peak_enhancement >= 1.0);
}