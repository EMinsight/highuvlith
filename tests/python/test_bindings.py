"""Tests for PyO3 binding validation and features."""
import pytest
import numpy as np
import highuvlith as huv


class TestSourceConfig:
    def test_f2_laser_defaults(self):
        source = huv.SourceConfig.f2_laser()
        assert abs(source.wavelength_nm - 157.63) < 1.0

    def test_ar2_laser_defaults(self):
        source = huv.SourceConfig.ar2_laser()
        assert abs(source.wavelength_nm - 126.0) < 1.0

    def test_custom_getters(self):
        source = huv.SourceConfig(wavelength_nm=157.0, sigma_outer=0.5, bandwidth_pm=1.0, spectral_samples=3)
        assert abs(source.sigma_outer - 0.5) < 1e-10
        assert source.spectral_samples == 3
        assert abs(source.bandwidth_pm - 1.0) < 1e-10

    def test_repr(self):
        source = huv.SourceConfig.f2_laser()
        r = repr(source)
        assert '157' in r

    def test_invalid_wavelength(self):
        with pytest.raises(ValueError):
            huv.SourceConfig(wavelength_nm=-1.0)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            huv.SourceConfig(sigma_outer=0.0)

    def test_invalid_sigma_above_one(self):
        with pytest.raises(ValueError):
            huv.SourceConfig(sigma_outer=1.5)


class TestOpticsConfig:
    def test_default_getters(self):
        optics = huv.OpticsConfig()
        assert abs(optics.numerical_aperture - 0.75) < 1e-10
        assert optics.reduction > 0
        assert 0 <= optics.flare_fraction <= 1

    def test_invalid_na(self):
        with pytest.raises(ValueError):
            huv.OpticsConfig(numerical_aperture=0.0)

    def test_invalid_na_above_one(self):
        with pytest.raises(ValueError):
            huv.OpticsConfig(numerical_aperture=1.5)

    def test_rayleigh_resolution(self):
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        res = optics.rayleigh_resolution(157.63)
        # Rayleigh = 0.61 * lambda / NA
        expected = 0.61 * 157.63 / 0.75
        assert abs(res - expected) < 5.0  # allow some tolerance

    def test_repr(self):
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        r = repr(optics)
        assert 'NA=' in r or '0.75' in r


class TestResistConfig:
    def test_vuv_fluoropolymer_getters(self):
        resist = huv.ResistConfig.vuv_fluoropolymer()
        assert resist.thickness_nm > 0
        assert resist.dill_a >= 0
        assert resist.dill_b >= 0
        assert resist.dill_c >= 0
        assert resist.peb_diffusion_nm >= 0

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown"):
            huv.ResistConfig(model="unknown")

    def test_invalid_thickness(self):
        with pytest.raises(ValueError):
            huv.ResistConfig(thickness_nm=-10.0)

    def test_mack_model_default(self):
        resist = huv.ResistConfig(model="mack")
        assert resist.thickness_nm > 0

    def test_threshold_model(self):
        resist = huv.ResistConfig(model="threshold")
        assert resist.thickness_nm > 0

    def test_repr(self):
        resist = huv.ResistConfig.vuv_fluoropolymer()
        r = repr(resist)
        assert 'ResistConfig' in r


class TestGridConfig:
    def test_valid_power_of_two(self):
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        assert grid.size == 128
        assert abs(grid.pixel_nm - 2.0) < 1e-10

    def test_invalid_not_power_of_two(self):
        with pytest.raises(ValueError):
            huv.GridConfig(size=100)

    def test_field_size(self):
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        assert abs(grid.field_size_nm() - 256.0) < 1e-10

    def test_repr(self):
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        r = repr(grid)
        assert '64' in r


class TestMaskConfig:
    def test_line_space(self):
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        r = repr(mask)
        assert 'MaskConfig' in r

    def test_contact_hole(self):
        mask = huv.MaskConfig.contact_hole(50.0, 150.0, 150.0)
        r = repr(mask)
        assert 'MaskConfig' in r

    def test_invalid_cd_negative(self):
        with pytest.raises(ValueError):
            huv.MaskConfig.line_space(-10.0, 180.0)

    def test_invalid_pitch_negative(self):
        with pytest.raises(ValueError):
            huv.MaskConfig.line_space(65.0, -180.0)


class TestSimulationEngine:
    def test_config_getters(self):
        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        # Test that getters work
        assert engine.source.wavelength_nm > 0
        assert engine.optics.numerical_aperture > 0
        assert engine.grid.size == 64

    def test_num_kernels(self):
        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        assert engine.num_kernels() <= 10
        assert engine.num_kernels() > 0

    def test_repr(self):
        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        r = repr(engine)
        assert 'SimulationEngine' in r


class TestAerialImageResult:
    def test_nils(self):
        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)
        nils = result.nils(threshold=0.3)
        # NILS might be None for some configurations, or a positive float
        if nils is not None:
            assert nils > 0

    def test_image_contrast(self):
        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)
        c = result.image_contrast()
        assert 0 <= c <= 1.0

    def test_coord_arrays(self):
        source = huv.SourceConfig.f2_laser()
        optics = huv.OpticsConfig()
        mask = huv.MaskConfig.line_space(65.0, 180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)
        x = np.asarray(result.x_nm)
        y = np.asarray(result.y_nm)
        assert len(x) == 64
        assert len(y) == 64
