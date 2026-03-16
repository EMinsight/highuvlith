"""Analytical validation tests via Python API."""

import numpy as np
import highuvlith as huv


class TestDefocusSymmetry:
    def test_positive_negative_defocus_equal_contrast(self):
        source = huv.SourceConfig.f2_laser(sigma=0.5)
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid)

        c_pos = engine.image_contrast(focus_nm=150.0)
        c_neg = engine.image_contrast(focus_nm=-150.0)
        np.testing.assert_allclose(c_pos, c_neg, rtol=1e-3)


class TestIntensityBounds:
    def test_non_negative_intensity(self):
        source = huv.SourceConfig.f2_laser(sigma=0.7)
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid)

        for focus in [-200.0, 0.0, 200.0]:
            result = engine.compute_aerial_image(focus_nm=focus)
            intensity = np.asarray(result.intensity)
            assert intensity.min() >= -1e-10, (
                f"Negative intensity at focus={focus}: {intensity.min()}"
            )


class TestMonotonicity:
    def test_defocus_reduces_contrast(self):
        source = huv.SourceConfig.f2_laser(sigma=0.7)
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid)

        # Test within moderate defocus range (contrast reversal can occur at extreme defocus)
        c0 = engine.image_contrast(focus_nm=0.0)
        c100 = engine.image_contrast(focus_nm=100.0)
        c200 = engine.image_contrast(focus_nm=200.0)
        assert c0 > c100, f"Contrast at focus=0 ({c0:.4f}) should exceed focus=100 ({c100:.4f})"
        assert c100 > c200, f"Contrast at focus=100 ({c100:.4f}) should exceed focus=200 ({c200:.4f})"


class TestEnergyConservation:
    def test_total_intensity_preserved_with_defocus(self):
        source = huv.SourceConfig.f2_laser(sigma=0.5)
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid)

        e0 = np.asarray(engine.compute_aerial_image(focus_nm=0.0).intensity).sum()
        e200 = np.asarray(engine.compute_aerial_image(focus_nm=200.0).intensity).sum()
        np.testing.assert_allclose(e0, e200, rtol=0.05)
