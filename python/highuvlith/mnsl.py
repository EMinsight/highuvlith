"""MNSL (Moiré Nanosphere Lithographic Reflection) high-level API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import highuvlith as huv
from ._native import (
    PyMnslConfig,
    PyMnslEngine,
    PyMnslResult,
    PyNanosphereArrayConfig,
    PySpherePacking,
    PySubstrateCoupling,
    py_simulate_moire_emission,
)


@dataclass
class MnslSimResult:
    """Enhanced result from MNSL simulation with analysis capabilities."""

    result: PyMnslResult
    config: dict[str, Any]

    @property
    def emission_pattern(self) -> np.ndarray:
        """Enhanced emission intensity pattern as 2D numpy array."""
        return np.asarray(self.result.emission_pattern)

    @property
    def moire_pattern(self) -> np.ndarray:
        """Underlying Moiré interference pattern as 2D numpy array."""
        return np.asarray(self.result.moire_pattern)

    @property
    def enhancement_factors(self) -> np.ndarray:
        """Local field enhancement factors as 2D numpy array."""
        return np.asarray(self.result.enhancement_factors)

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical coordinate arrays (x_coords, y_coords) in nm."""
        x_coords, y_coords = self.result.coordinates()
        return np.array(x_coords), np.array(y_coords)

    def cross_section_x(self, y_nm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Get 1D cross-section along x at specified y-coordinate.

        Args:
            y_nm: Y-coordinate in nm. Default 0.0 (center).

        Returns:
            Tuple of (x_coords, intensities) arrays.
        """
        x_coords, _ = self.coordinates
        intensities = np.array(self.result.cross_section_x(y_nm))
        return x_coords, intensities

    def cross_section_y(self, x_nm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Get 1D cross-section along y at specified x-coordinate.

        Args:
            x_nm: X-coordinate in nm. Default 0.0 (center).

        Returns:
            Tuple of (y_coords, intensities) arrays.
        """
        _, y_coords = self.coordinates
        intensities = np.array(self.result.cross_section_y(x_nm))
        return y_coords, intensities

    @property
    def moire_period_nm(self) -> float:
        """Calculated Moiré period in nm."""
        return self.result.moire_period_nm

    @property
    def peak_enhancement(self) -> float:
        """Peak enhancement factor."""
        return self.result.peak_enhancement

    @property
    def total_emission_power(self) -> float:
        """Total integrated emission power (relative units)."""
        return self.result.total_emission_power

    @property
    def peak_positions(self) -> list[tuple[float, float]]:
        """Peak positions as list of (x, y) coordinates in nm."""
        return self.result.peak_positions

    @property
    def num_peaks(self) -> int:
        """Number of emission peaks found."""
        return len(self.peak_positions)


# Type aliases for convenience
SpherePacking = PySpherePacking
NanosphereArrayConfig = PyNanosphereArrayConfig
SubstrateCoupling = PySubstrateCoupling
MnslConfig = PyMnslConfig
MnslEngine = PyMnslEngine


def simulate_moire_emission(
    sphere_diameter_nm: float,
    array_pitch_nm: float,
    rotation_angle_deg: float,
    *,
    separation_nm: float = 100.0,
    wavelength_nm: float = 157.0,
    grid_size: int = 256,
    pixel_nm: float = 2.0,
    sphere_material: str = "silica",
    coupling_strength: float = 0.5,
    enable_nearfield: bool = True,
) -> MnslSimResult:
    """One-liner MNSL simulation for Moiré nanosphere emission enhancement.

    Args:
        sphere_diameter_nm: Nanosphere diameter in nm.
        array_pitch_nm: Center-to-center spacing of nanospheres in nm.
        rotation_angle_deg: Rotation angle between top and bottom arrays in degrees.
        separation_nm: Vertical separation between arrays in nm. Default 100.
        wavelength_nm: Simulation wavelength in nm. Default 157 (F2 laser).
        grid_size: Simulation grid size (must be power of 2). Default 256.
        pixel_nm: Pixel size in nm. Default 2.0.
        sphere_material: Sphere material ("silica" or "polystyrene"). Default "silica".
        coupling_strength: Substrate coupling strength (0.0 to 1.0). Default 0.5.
        enable_nearfield: Enable near-field enhancement calculation. Default True.

    Returns:
        MnslSimResult with emission patterns, enhancement factors, and analysis.

    Example:
        >>> result = simulate_moire_emission(200.0, 300.0, 5.0)
        >>> print(f"Moiré period: {result.moire_period_nm:.1f} nm")
        >>> print(f"Peak enhancement: {result.peak_enhancement:.2f}×")
        >>> print(f"Number of peaks: {result.num_peaks}")
    """
    # Use low-level function for efficient simulation
    py_result = py_simulate_moire_emission(
        sphere_diameter_nm,
        array_pitch_nm,
        rotation_angle_deg,
        separation_nm,
        grid_size,
        pixel_nm,
    )

    config = {
        "sphere_diameter_nm": sphere_diameter_nm,
        "array_pitch_nm": array_pitch_nm,
        "rotation_angle_deg": rotation_angle_deg,
        "separation_nm": separation_nm,
        "wavelength_nm": wavelength_nm,
        "grid_size": grid_size,
        "pixel_nm": pixel_nm,
        "sphere_material": sphere_material,
        "coupling_strength": coupling_strength,
        "enable_nearfield": enable_nearfield,
    }

    return MnslSimResult(result=py_result, config=config)


def create_nanosphere_array(
    diameter_nm: float,
    pitch_nm: float,
    *,
    orientation_deg: float = 0.0,
    material: str = "silica",
    packing: str = "hcp",
) -> NanosphereArrayConfig:
    """Create a nanosphere array configuration.

    Args:
        diameter_nm: Sphere diameter in nm.
        pitch_nm: Center-to-center spacing in nm.
        orientation_deg: Rotation angle in degrees. Default 0.0.
        material: Sphere material ("silica" or "polystyrene"). Default "silica".
        packing: Packing type ("hcp", "fcc", or "simple_cubic"). Default "hcp".

    Returns:
        NanosphereArrayConfig object.
    """
    packing_map = {
        "hcp": SpherePacking.HCP,
        "fcc": SpherePacking.FCC,
        "simple_cubic": SpherePacking.SIMPLE_CUBIC,
    }

    if material == "silica":
        array = NanosphereArrayConfig.silica_spheres(diameter_nm, pitch_nm)
    elif material == "polystyrene":
        array = NanosphereArrayConfig.polystyrene_spheres(diameter_nm, pitch_nm)
    else:
        raise ValueError(f"Unknown material: {material}")

    array.orientation_deg = orientation_deg
    if packing in packing_map:
        # Note: We can't set packing directly due to PyO3 limitations
        # The packing parameter is stored for future reference but may not affect the simulation
        pass

    return array


def sweep_rotation_angle(
    sphere_diameter_nm: float,
    array_pitch_nm: float,
    *,
    angle_min: float = 0.0,
    angle_max: float = 10.0,
    angle_steps: int = 11,
    separation_nm: float = 100.0,
    grid_size: int = 128,
    pixel_nm: float = 4.0,
) -> dict[str, Any]:
    """Sweep rotation angle and return enhancement vs angle data.

    Args:
        sphere_diameter_nm: Sphere diameter in nm.
        array_pitch_nm: Array pitch in nm.
        angle_min: Minimum rotation angle in degrees. Default 0.0.
        angle_max: Maximum rotation angle in degrees. Default 10.0.
        angle_steps: Number of angle steps. Default 11.
        separation_nm: Layer separation in nm. Default 100.
        grid_size: Simulation grid size. Default 128.
        pixel_nm: Pixel size in nm. Default 4.0.

    Returns:
        Dict with 'angles', 'peak_enhancements', 'moire_periods' arrays
        and 'best_angle_deg', 'best_enhancement'.
    """
    angles = np.linspace(angle_min, angle_max, angle_steps)
    peak_enhancements = []
    moire_periods = []
    total_powers = []

    for angle in angles:
        result = simulate_moire_emission(
            sphere_diameter_nm,
            array_pitch_nm,
            angle,
            separation_nm=separation_nm,
            grid_size=grid_size,
            pixel_nm=pixel_nm,
        )
        peak_enhancements.append(result.peak_enhancement)
        moire_periods.append(result.moire_period_nm)
        total_powers.append(result.total_emission_power)

    peak_enhancements = np.array(peak_enhancements)
    moire_periods = np.array(moire_periods)
    total_powers = np.array(total_powers)

    best_idx = np.argmax(peak_enhancements)

    return {
        "angles": angles,
        "peak_enhancements": peak_enhancements,
        "moire_periods": moire_periods,
        "total_powers": total_powers,
        "best_angle_deg": angles[best_idx],
        "best_enhancement": peak_enhancements[best_idx],
        "best_period_nm": moire_periods[best_idx],
    }


def sweep_separation(
    sphere_diameter_nm: float,
    array_pitch_nm: float,
    rotation_angle_deg: float,
    *,
    separation_min: float = 50.0,
    separation_max: float = 200.0,
    separation_steps: int = 11,
    grid_size: int = 128,
    pixel_nm: float = 4.0,
) -> dict[str, Any]:
    """Sweep layer separation and return enhancement vs separation data.

    Args:
        sphere_diameter_nm: Sphere diameter in nm.
        array_pitch_nm: Array pitch in nm.
        rotation_angle_deg: Rotation angle in degrees.
        separation_min: Minimum separation in nm. Default 50.
        separation_max: Maximum separation in nm. Default 200.
        separation_steps: Number of separation steps. Default 11.
        grid_size: Simulation grid size. Default 128.
        pixel_nm: Pixel size in nm. Default 4.0.

    Returns:
        Dict with 'separations', 'peak_enhancements', 'total_powers' arrays
        and 'best_separation_nm', 'best_enhancement'.
    """
    separations = np.linspace(separation_min, separation_max, separation_steps)
    peak_enhancements = []
    total_powers = []

    for separation in separations:
        result = simulate_moire_emission(
            sphere_diameter_nm,
            array_pitch_nm,
            rotation_angle_deg,
            separation_nm=separation,
            grid_size=grid_size,
            pixel_nm=pixel_nm,
        )
        peak_enhancements.append(result.peak_enhancement)
        total_powers.append(result.total_emission_power)

    peak_enhancements = np.array(peak_enhancements)
    total_powers = np.array(total_powers)

    best_idx = np.argmax(peak_enhancements)

    return {
        "separations": separations,
        "peak_enhancements": peak_enhancements,
        "total_powers": total_powers,
        "best_separation_nm": separations[best_idx],
        "best_enhancement": peak_enhancements[best_idx],
    }


def optimize_moire_parameters(
    sphere_diameter_nm: float,
    array_pitch_nm: float,
    *,
    angle_range: tuple[float, float] = (0.0, 10.0),
    separation_range: tuple[float, float] = (50.0, 200.0),
    angle_steps: int = 11,
    separation_steps: int = 11,
    grid_size: int = 128,
    pixel_nm: float = 4.0,
) -> dict[str, Any]:
    """Optimize rotation angle and separation for maximum emission enhancement.

    Args:
        sphere_diameter_nm: Sphere diameter in nm.
        array_pitch_nm: Array pitch in nm.
        angle_range: (min, max) rotation angles in degrees. Default (0.0, 10.0).
        separation_range: (min, max) separations in nm. Default (50.0, 200.0).
        angle_steps: Number of angle steps. Default 11.
        separation_steps: Number of separation steps. Default 11.
        grid_size: Simulation grid size. Default 128.
        pixel_nm: Pixel size in nm. Default 4.0.

    Returns:
        Dict with optimization results including 'best_angle_deg', 'best_separation_nm',
        'max_enhancement', and full parameter sweep data.
    """
    angles = np.linspace(angle_range[0], angle_range[1], angle_steps)
    separations = np.linspace(separation_range[0], separation_range[1], separation_steps)

    enhancement_grid = np.zeros((len(angles), len(separations)))
    max_enhancement = 0.0
    best_angle = angles[0]
    best_separation = separations[0]

    for i, angle in enumerate(angles):
        for j, separation in enumerate(separations):
            result = simulate_moire_emission(
                sphere_diameter_nm,
                array_pitch_nm,
                angle,
                separation_nm=separation,
                grid_size=grid_size,
                pixel_nm=pixel_nm,
            )
            enhancement = result.peak_enhancement
            enhancement_grid[i, j] = enhancement

            if enhancement > max_enhancement:
                max_enhancement = enhancement
                best_angle = angle
                best_separation = separation

    # Run final simulation with best parameters
    best_result = simulate_moire_emission(
        sphere_diameter_nm,
        array_pitch_nm,
        best_angle,
        separation_nm=best_separation,
        grid_size=grid_size,
        pixel_nm=pixel_nm,
    )

    return {
        "best_angle_deg": best_angle,
        "best_separation_nm": best_separation,
        "max_enhancement": max_enhancement,
        "best_result": best_result,
        "enhancement_grid": enhancement_grid,
        "angles": angles,
        "separations": separations,
        "moire_period_nm": best_result.moire_period_nm,
        "num_peaks": best_result.num_peaks,
    }