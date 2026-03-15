import pytest
import highuvlith as huv


@pytest.fixture
def source():
    return huv.SourceConfig.f2_laser(sigma=0.7)


@pytest.fixture
def optics():
    return huv.OpticsConfig(numerical_aperture=0.75)


@pytest.fixture
def mask():
    return huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)


@pytest.fixture
def resist():
    return huv.ResistConfig.vuv_fluoropolymer()


@pytest.fixture
def grid():
    return huv.GridConfig(size=128, pixel_nm=2.0)


@pytest.fixture
def engine(source, optics, mask, resist, grid):
    return huv.SimulationEngine(source, optics, mask, resist, grid)
