"""Tests for I/O utilities."""
import pytest
import numpy as np
from pathlib import Path
from highuvlith.io.config_io import save_config, load_config
from highuvlith.io.results_io import save_result, load_result


class TestConfigIO:
    def test_save_and_load_round_trip(self, tmp_path):
        config = {
            "source": {"wavelength_nm": 157.63, "sigma": 0.7},
            "optics": {"na": 0.75},
            "mask": {"cd_nm": 65.0, "pitch_nm": 180.0},
        }
        path = tmp_path / "test_config.toml"
        save_config(config, str(path))
        loaded = load_config(str(path))
        assert loaded["source"]["wavelength_nm"] == 157.63
        assert loaded["optics"]["na"] == 0.75

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.toml"))

    def test_save_nested_dict(self, tmp_path):
        config = {"section": {"key": "value", "num": 42}}
        path = tmp_path / "nested.toml"
        save_config(config, str(path))
        loaded = load_config(str(path))
        assert loaded["section"]["key"] == "value"
        assert loaded["section"]["num"] == 42

    def test_save_with_bool(self, tmp_path):
        config = {"settings": {"enabled": True, "debug": False}}
        path = tmp_path / "bool.toml"
        save_config(config, str(path))
        loaded = load_config(str(path))
        assert loaded["settings"]["enabled"] is True
        assert loaded["settings"]["debug"] is False


class TestResultsIO:
    def test_save_and_load_aerial(self, tmp_path):
        import highuvlith as huv

        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)

        path = tmp_path / "result.npz"
        save_result(result, str(path))
        loaded = load_result(str(path))
        assert "intensity" in loaded
        np.testing.assert_array_equal(loaded["intensity"].shape, (64, 64))

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_result(str(tmp_path / "nonexistent.npz"))

    def test_save_and_load_process_window(self, tmp_path):
        import highuvlith as huv

        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        batch = huv.BatchSimulator(source, optics, mask, grid)
        pw = batch.process_window(
            doses=[20.0, 30.0, 40.0],
            focuses=[-200.0, 0.0, 200.0],
        )

        path = tmp_path / "pw.npz"
        save_result(pw, str(path))
        loaded = load_result(str(path))
        assert "cd_matrix" in loaded
        assert "doses" in loaded
        assert "focuses" in loaded

    def test_saved_intensity_matches_original(self, tmp_path):
        import highuvlith as huv

        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)
        original_intensity = np.asarray(result.intensity).copy()

        path = tmp_path / "result2.npz"
        save_result(result, str(path))
        loaded = load_result(str(path))
        np.testing.assert_array_almost_equal(loaded["intensity"], original_intensity)
