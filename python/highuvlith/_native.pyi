"""Type stubs for the highuvlith._native extension module."""

from typing import Optional

import numpy as np

class SourceConfig:
    """VUV laser source configuration."""

    def __init__(
        self,
        wavelength_nm: float = 157.63,
        sigma_outer: float = 0.7,
        bandwidth_pm: float = 1.1,
        spectral_samples: int = 5,
    ) -> None: ...
    @staticmethod
    def f2_laser(sigma: float = 0.7) -> SourceConfig:
        """Create F2 laser (157nm) source."""
        ...
    @staticmethod
    def ar2_laser(sigma: float = 0.7) -> SourceConfig:
        """Create Ar2 excimer (126nm) source."""
        ...
    @property
    def wavelength_nm(self) -> float: ...
    @property
    def bandwidth_pm(self) -> float: ...
    def __repr__(self) -> str: ...

class OpticsConfig:
    """Projection optics configuration."""

    def __init__(
        self,
        numerical_aperture: float = 0.75,
        reduction: float = 4.0,
        flare_fraction: float = 0.02,
    ) -> None: ...
    def add_aberration(self, fringe_index: int, coefficient_waves: float) -> None:
        """Add a Zernike aberration coefficient."""
        ...
    @property
    def numerical_aperture(self) -> float: ...
    def rayleigh_resolution(self, wavelength_nm: float) -> float: ...
    def __repr__(self) -> str: ...

class MaskConfig:
    """Mask configuration."""

    @staticmethod
    def line_space(cd_nm: float, pitch_nm: float) -> MaskConfig:
        """Create a line/space pattern."""
        ...
    @staticmethod
    def contact_hole(
        diameter_nm: float, pitch_x_nm: float, pitch_y_nm: float
    ) -> MaskConfig:
        """Create a contact hole array."""
        ...
    def __repr__(self) -> str: ...

class ResistConfig:
    """Photoresist configuration."""

    def __init__(
        self,
        thickness_nm: float = 150.0,
        dill_a: float = 0.2,
        dill_b: float = 0.45,
        dill_c: float = 0.02,
        peb_diffusion_nm: float = 30.0,
        model: str = "mack",
    ) -> None: ...
    @staticmethod
    def vuv_fluoropolymer() -> ResistConfig:
        """Create VUV fluoropolymer resist with default parameters."""
        ...
    def __repr__(self) -> str: ...

class FilmStackConfig:
    """Film stack configuration."""

    def __init__(self) -> None: ...
    def add_layer(
        self, name: str, thickness_nm: float, n_real: float, n_imag: float
    ) -> None:
        """Add a layer to the stack."""
        ...
    def set_substrate(self, n_real: float, n_imag: float) -> None:
        """Set the substrate refractive index."""
        ...
    def __repr__(self) -> str: ...

class ProcessConfig:
    """Process parameters."""

    dose_mj_cm2: float
    focus_nm: float
    development_time_s: float

    def __init__(
        self,
        dose_mj_cm2: float = 30.0,
        focus_nm: float = 0.0,
        development_time_s: float = 60.0,
    ) -> None: ...
    def __repr__(self) -> str: ...

class GridConfig:
    """Grid configuration."""

    def __init__(self, size: int = 512, pixel_nm: float = 1.0) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def pixel_nm(self) -> float: ...
    def field_size_nm(self) -> float: ...
    def __repr__(self) -> str: ...

class AerialImageResult:
    """Aerial image simulation result."""

    @property
    def intensity(self) -> np.ndarray: ...
    @property
    def x_nm(self) -> np.ndarray: ...
    @property
    def y_nm(self) -> np.ndarray: ...
    def cross_section(
        self, y_nm: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a 1D cross-section along x at y=y_nm."""
        ...
    def image_contrast(self) -> float:
        """Compute image contrast: (Imax - Imin) / (Imax + Imin)."""
        ...
    def nils(self, threshold: float = 0.3) -> Optional[float]:
        """Compute NILS at y=0 cross-section."""
        ...
    def __repr__(self) -> str: ...

class ResistProfileResult:
    """Resist profile simulation result."""

    @property
    def x_nm(self) -> np.ndarray: ...
    @property
    def height_nm(self) -> np.ndarray: ...
    @property
    def thickness_nm(self) -> float: ...
    def __repr__(self) -> str: ...

class SimulationEngine:
    """Core simulation engine. Precomputes TCC for efficient multi-evaluation."""

    def __init__(
        self,
        source: SourceConfig,
        optics: OpticsConfig,
        mask: MaskConfig,
        resist: Optional[ResistConfig] = None,
        grid: Optional[GridConfig] = None,
        max_kernels: int = 30,
    ) -> None: ...
    def compute_aerial_image(
        self, focus_nm: float = 0.0
    ) -> AerialImageResult:
        """Compute aerial image at the given defocus."""
        ...
    def compute_polychromatic(
        self, focus_nm: float = 0.0
    ) -> AerialImageResult:
        """Compute polychromatic aerial image (accounts for VUV chromatic aberration)."""
        ...
    def measure_cd(
        self,
        dose_mj_cm2: float = 30.0,
        focus_nm: float = 0.0,
        threshold: float = 0.3,
    ) -> float:
        """Measure CD at given dose and focus."""
        ...
    def compute_resist_profile(
        self,
        dose_mj_cm2: float = 30.0,
        focus_nm: float = 0.0,
        dev_time_s: float = 60.0,
    ) -> ResistProfileResult:
        """Compute resist profile at given dose and focus."""
        ...
    def image_contrast(self, focus_nm: float = 0.0) -> float:
        """Get image contrast at given focus."""
        ...
    def num_kernels(self) -> int:
        """Number of SOCS kernels in the decomposition."""
        ...
    def __repr__(self) -> str: ...

class BatchSimulator:
    """Batch simulation with GIL release for parameter sweeps."""

    def __init__(
        self,
        source: SourceConfig,
        optics: OpticsConfig,
        mask: MaskConfig,
        grid: Optional[GridConfig] = None,
        max_kernels: int = 30,
    ) -> None: ...
    def process_window(
        self,
        doses: list[float],
        focuses: list[float],
        cd_threshold: float = 0.3,
        cd_target_nm: float = 65.0,
        cd_tolerance_pct: float = 10.0,
    ) -> ProcessWindowResult:
        """Compute process window: CD vs dose and focus."""
        ...
    def batch_defocus(
        self,
        focuses: list[float],
    ) -> list[tuple[float, np.ndarray]]:
        """Batch compute aerial images at multiple focus values."""
        ...

class ProcessWindowResult:
    """Process window analysis result."""

    @property
    def cd_matrix(self) -> np.ndarray: ...
    @property
    def doses(self) -> np.ndarray: ...
    @property
    def focuses(self) -> np.ndarray: ...
    def depth_of_focus(self) -> float:
        """Depth of focus in nm."""
        ...
    def exposure_latitude(self) -> float:
        """Exposure latitude in percent."""
        ...
    def __repr__(self) -> str: ...
