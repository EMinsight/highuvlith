"""Tests for MNSL module."""
import pytest
import numpy as np
from highuvlith.mnsl import (
    simulate_moire_emission,
    create_nanosphere_array,
    sweep_rotation_angle,
)


class TestSimulateMoireEmission:
    def test_returns_result(self):
        result = simulate_moire_emission(
            sphere_diameter_nm=200.0,
            array_pitch_nm=300.0,
            rotation_angle_deg=5.0,
            grid_size=64,
            pixel_nm=4.0,
        )
        assert hasattr(result, 'emission_pattern')
        assert hasattr(result, 'moire_pattern')
        assert hasattr(result, 'peak_enhancement')

    def test_peak_enhancement_positive(self):
        result = simulate_moire_emission(
            sphere_diameter_nm=200.0,
            array_pitch_nm=300.0,
            rotation_angle_deg=5.0,
            grid_size=64,
            pixel_nm=4.0,
        )
        assert result.peak_enhancement > 0

    def test_emission_pattern_shape(self):
        result = simulate_moire_emission(
            sphere_diameter_nm=200.0,
            array_pitch_nm=300.0,
            rotation_angle_deg=5.0,
            grid_size=64,
            pixel_nm=4.0,
        )
        pattern = result.emission_pattern
        assert pattern.shape == (64, 64)

    def test_moire_period_positive(self):
        result = simulate_moire_emission(
            sphere_diameter_nm=200.0,
            array_pitch_nm=300.0,
            rotation_angle_deg=5.0,
            grid_size=64,
            pixel_nm=4.0,
        )
        assert result.moire_period_nm > 0

    def test_invalid_diameter_negative(self):
        with pytest.raises(ValueError, match="sphere_diameter_nm"):
            simulate_moire_emission(
                sphere_diameter_nm=-100.0,
                array_pitch_nm=300.0,
                rotation_angle_deg=5.0,
            )

    def test_invalid_coupling_out_of_range(self):
        with pytest.raises(ValueError, match="coupling_strength"):
            simulate_moire_emission(
                sphere_diameter_nm=200.0,
                array_pitch_nm=300.0,
                rotation_angle_deg=5.0,
                coupling_strength=1.5,
            )

    def test_invalid_pitch_negative(self):
        with pytest.raises(ValueError, match="array_pitch_nm"):
            simulate_moire_emission(
                sphere_diameter_nm=200.0,
                array_pitch_nm=-300.0,
                rotation_angle_deg=5.0,
            )

    def test_invalid_grid_not_power_of_two(self):
        with pytest.raises(ValueError, match="grid_size"):
            simulate_moire_emission(
                sphere_diameter_nm=200.0,
                array_pitch_nm=300.0,
                rotation_angle_deg=5.0,
                grid_size=100,
            )


class TestCreateNanosphereArray:
    def test_silica_spheres(self):
        config = create_nanosphere_array(diameter_nm=200.0, pitch_nm=300.0, material="silica")
        assert config is not None

    def test_polystyrene_spheres(self):
        config = create_nanosphere_array(diameter_nm=200.0, pitch_nm=300.0, material="polystyrene")
        assert config is not None

    def test_invalid_negative_diameter(self):
        with pytest.raises(ValueError, match="diameter_nm"):
            create_nanosphere_array(diameter_nm=-10.0, pitch_nm=300.0)

    def test_invalid_negative_pitch(self):
        with pytest.raises(ValueError, match="pitch_nm"):
            create_nanosphere_array(diameter_nm=200.0, pitch_nm=-300.0)

    def test_unknown_material(self):
        with pytest.raises(ValueError, match="Unknown material"):
            create_nanosphere_array(diameter_nm=200.0, pitch_nm=300.0, material="diamond")


class TestSweepRotationAngle:
    def test_returns_dict_with_keys(self):
        result = sweep_rotation_angle(
            sphere_diameter_nm=200.0,
            array_pitch_nm=300.0,
            angle_min=0.0,
            angle_max=30.0,
            angle_steps=3,
            grid_size=64,
            pixel_nm=4.0,
        )
        assert 'angles' in result
        assert 'peak_enhancements' in result
        assert 'best_angle_deg' in result

    def test_steps_match_output_length(self):
        result = sweep_rotation_angle(
            sphere_diameter_nm=200.0,
            array_pitch_nm=300.0,
            angle_min=0.0,
            angle_max=30.0,
            angle_steps=5,
            grid_size=64,
            pixel_nm=4.0,
        )
        assert len(result['angles']) == 5
        assert len(result['peak_enhancements']) == 5

    def test_invalid_angle_range(self):
        with pytest.raises(ValueError, match="angle_min"):
            sweep_rotation_angle(
                sphere_diameter_nm=200.0,
                array_pitch_nm=300.0,
                angle_min=30.0,
                angle_max=0.0,
            )

    def test_invalid_angle_steps_too_few(self):
        with pytest.raises(ValueError, match="angle_steps"):
            sweep_rotation_angle(
                sphere_diameter_nm=200.0,
                array_pitch_nm=300.0,
                angle_steps=1,
            )
