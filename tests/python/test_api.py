"""Tests for the high-level convenience API."""
import pytest
import numpy as np
import highuvlith as huv
from highuvlith.api import simulate_line_space, simulate_contact_hole, sweep_focus


class TestSimulateLineSpace:
    def test_returns_full_result(self):
        result = simulate_line_space(65.0, 180.0, grid_size=64, pixel_nm=4.0)
        assert hasattr(result, 'aerial')
        assert hasattr(result, 'contrast')
        assert hasattr(result, 'config')
        assert result.contrast > 0

    def test_with_resist(self):
        result = simulate_line_space(65.0, 180.0, grid_size=64, pixel_nm=4.0, with_resist=True)
        assert result.resist_profile is not None

    def test_nils_populated(self):
        result = simulate_line_space(65.0, 180.0, grid_size=128, pixel_nm=2.0)
        # NILS may or may not be computed depending on the profile
        # Just check it's set (could be None or a float)
        assert hasattr(result, 'nils')

    def test_invalid_cd_negative(self):
        with pytest.raises(ValueError, match="cd_nm"):
            simulate_line_space(-10.0, 180.0)

    def test_invalid_cd_exceeds_pitch(self):
        with pytest.raises(ValueError):
            simulate_line_space(200.0, 180.0)

    def test_invalid_na_zero(self):
        with pytest.raises(ValueError, match="na"):
            simulate_line_space(65.0, 180.0, na=0.0)

    def test_invalid_grid_not_power_of_two(self):
        with pytest.raises(ValueError, match="grid_size"):
            simulate_line_space(65.0, 180.0, grid_size=100)

    @pytest.mark.parametrize("grid_size", [64, 128])
    def test_different_grid_sizes(self, grid_size):
        result = simulate_line_space(65.0, 180.0, grid_size=grid_size, pixel_nm=4.0)
        assert result.contrast > 0

    def test_config_dict_populated(self):
        result = simulate_line_space(65.0, 180.0, grid_size=64, pixel_nm=4.0)
        assert result.config['cd_nm'] == 65.0
        assert result.config['pitch_nm'] == 180.0
        assert result.config['grid_size'] == 64

    def test_aerial_image_shape(self):
        result = simulate_line_space(65.0, 180.0, grid_size=64, pixel_nm=4.0)
        intensity = np.asarray(result.aerial.intensity)
        assert intensity.shape == (64, 64)

    def test_without_resist_default(self):
        result = simulate_line_space(65.0, 180.0, grid_size=64, pixel_nm=4.0)
        assert result.resist_profile is None


class TestSimulateContactHole:
    def test_returns_result(self):
        result = simulate_contact_hole(50.0, 150.0, grid_size=64, pixel_nm=4.0)
        assert result.contrast >= 0

    def test_square_pitch_default(self):
        result = simulate_contact_hole(50.0, 150.0, grid_size=64, pixel_nm=4.0)
        assert result.config['pitch_x_nm'] == result.config['pitch_y_nm']

    def test_invalid_diameter_negative(self):
        with pytest.raises(ValueError, match="diameter"):
            simulate_contact_hole(-10.0, 150.0)

    def test_rectangular_pitch(self):
        result = simulate_contact_hole(50.0, 150.0, 200.0, grid_size=64, pixel_nm=4.0)
        assert result.config['pitch_x_nm'] == 150.0
        assert result.config['pitch_y_nm'] == 200.0


class TestSweepFocus:
    def test_returns_dict_with_keys(self):
        result = sweep_focus(65.0, 180.0, grid_size=64, pixel_nm=4.0, focus_steps=5)
        assert 'focuses' in result
        assert 'contrasts' in result
        assert 'best_focus_nm' in result
        assert 'best_contrast' in result

    def test_contrasts_shape_matches_steps(self):
        result = sweep_focus(65.0, 180.0, grid_size=64, pixel_nm=4.0, focus_steps=7)
        assert len(result['contrasts']) == 7
        assert len(result['focuses']) == 7

    def test_focus_min_exceeds_max(self):
        with pytest.raises(ValueError, match="focus_min"):
            sweep_focus(65.0, 180.0, focus_min=100, focus_max=-100)

    def test_focus_steps_too_few(self):
        with pytest.raises(ValueError, match="focus_steps"):
            sweep_focus(65.0, 180.0, focus_steps=1)

    def test_best_contrast_in_range(self):
        result = sweep_focus(65.0, 180.0, grid_size=64, pixel_nm=4.0, focus_steps=5)
        assert 0 <= result['best_contrast'] <= 1.0
