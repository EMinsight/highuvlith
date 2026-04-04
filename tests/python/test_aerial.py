import numpy as np
import highuvlith as huv


class TestAerialImage:
    def test_returns_correct_shape(self, engine, grid):
        result = engine.compute_aerial_image()
        intensity = np.asarray(result.intensity)
        assert intensity.ndim == 2
        assert intensity.shape == (grid.size, grid.size)

    def test_intensity_non_negative(self, engine):
        result = engine.compute_aerial_image()
        intensity = np.asarray(result.intensity)
        assert intensity.min() >= -1e-10

    def test_defocus_reduces_contrast(self, engine):
        c0 = engine.image_contrast(focus_nm=0.0)
        c100 = engine.image_contrast(focus_nm=100.0)
        c200 = engine.image_contrast(focus_nm=200.0)
        assert c0 > c100 > c200

    def test_cross_section(self, engine):
        result = engine.compute_aerial_image()
        x, intensity = result.cross_section(y_nm=0.0)
        x_arr = np.asarray(x)
        i_arr = np.asarray(intensity)
        assert len(x_arr) == len(i_arr)
        assert x_arr[0] < x_arr[-1]

    def test_symmetric_defocus(self, engine):
        """Positive and negative defocus should give same contrast (symmetric optics)."""
        c_pos = engine.image_contrast(focus_nm=150.0)
        c_neg = engine.image_contrast(focus_nm=-150.0)
        np.testing.assert_allclose(c_pos, c_neg, rtol=0.01)

    def test_polychromatic(self, engine):
        mono = engine.compute_aerial_image(focus_nm=0.0)
        poly = engine.compute_polychromatic(focus_nm=0.0)
        # Polychromatic should have slightly lower contrast due to chromatic aberration
        assert poly.image_contrast() <= mono.image_contrast() + 0.01


class TestResistProfile:
    def test_returns_profile(self, engine):
        profile = engine.compute_resist_profile(dose_mj_cm2=30.0)
        x = np.asarray(profile.x_nm)
        h = np.asarray(profile.height_nm)
        assert len(x) == len(h)
        assert profile.thickness_nm == 150.0

    def test_height_non_negative(self, engine):
        profile = engine.compute_resist_profile(dose_mj_cm2=30.0)
        h = np.asarray(profile.height_nm)
        assert h.min() >= 0.0

    def test_height_bounded_by_thickness(self, engine):
        profile = engine.compute_resist_profile(dose_mj_cm2=30.0)
        h = np.asarray(profile.height_nm)
        assert h.max() <= profile.thickness_nm + 1e-10


class TestEdgeCases:
    def test_small_cd_near_resolution_limit(self):
        """CD=30nm at 157nm/NA=0.75 is near resolution limit; should still compute."""
        source = huv.SourceConfig.f2_laser(sigma=0.7)
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=30.0, pitch_nm=80.0)
        grid = huv.GridConfig(size=128, pixel_nm=2.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)
        intensity = np.asarray(result.intensity)
        assert intensity.shape == (128, 128)
        assert intensity.min() >= -1e-10

    def test_grid_size_64_works(self):
        """Smallest practical grid produces valid results."""
        source = huv.SourceConfig.f2_laser(sigma=0.7)
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        grid = huv.GridConfig(size=64, pixel_nm=4.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=10)
        result = engine.compute_aerial_image(focus_nm=0.0)
        intensity = np.asarray(result.intensity)
        assert intensity.shape == (64, 64)
        contrast = result.image_contrast()
        assert contrast > 0
