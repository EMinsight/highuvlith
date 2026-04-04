"""High-level convenience API for common simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import highuvlith as huv


@dataclass
class FullResult:
    """Aggregated result from a single simulation run."""

    aerial: huv.AerialImageResult
    contrast: float
    config: dict[str, Any]
    resist_profile: huv.ResistProfileResult | None = None
    cd_nm: float | None = None
    nils: float | None = None


def simulate_line_space(
    cd_nm: float,
    pitch_nm: float,
    *,
    wavelength_nm: float = 157.63,
    na: float = 0.75,
    sigma: float = 0.7,
    focus_nm: float = 0.0,
    dose_mj_cm2: float = 30.0,
    grid_size: int = 256,
    pixel_nm: float = 1.0,
    with_resist: bool = False,
    max_kernels: int = 20,
) -> FullResult:
    """One-liner simulation of a line/space pattern.

    Args:
        cd_nm: Line width at wafer scale (nm).
        pitch_nm: Line+space pitch (nm).
        wavelength_nm: Source wavelength (nm). Default 157.63 (F2 laser).
        na: Numerical aperture. Default 0.75.
        sigma: Partial coherence factor. Default 0.7.
        focus_nm: Defocus (nm). Default 0 (best focus).
        dose_mj_cm2: Exposure dose (mJ/cm²). Default 30.
        grid_size: Simulation grid size (must be power of 2). Default 256.
        pixel_nm: Pixel size (nm). Default 1.0.
        with_resist: If True, also compute resist profile.
        max_kernels: Max SOCS kernels for TCC decomposition. Default 20.

    Returns:
        FullResult with aerial image, contrast, and optionally resist profile.
    """
    from highuvlith._validation import (
        _validate_positive,
        _validate_power_of_two,
        _validate_range,
    )

    _validate_positive("cd_nm", cd_nm)
    _validate_positive("pitch_nm", pitch_nm)
    if cd_nm >= pitch_nm:
        raise ValueError(f"cd_nm ({cd_nm}) must be less than pitch_nm ({pitch_nm})")
    _validate_range("na", na, 0.01, 1.0)
    _validate_range("sigma", sigma, 0.01, 1.0)
    _validate_power_of_two("grid_size", grid_size)
    _validate_positive("pixel_nm", pixel_nm)

    source = huv.SourceConfig(
        wavelength_nm=wavelength_nm,
        sigma_outer=sigma,
    )
    optics = huv.OpticsConfig(numerical_aperture=na)
    mask = huv.MaskConfig.line_space(cd_nm=cd_nm, pitch_nm=pitch_nm)
    resist = huv.ResistConfig.vuv_fluoropolymer()
    grid = huv.GridConfig(size=grid_size, pixel_nm=pixel_nm)

    engine = huv.SimulationEngine(
        source, optics, mask, resist, grid, max_kernels=max_kernels
    )

    aerial = engine.compute_aerial_image(focus_nm=focus_nm)
    contrast = aerial.image_contrast()

    config = {
        "wavelength_nm": wavelength_nm,
        "na": na,
        "sigma": sigma,
        "cd_nm": cd_nm,
        "pitch_nm": pitch_nm,
        "focus_nm": focus_nm,
        "dose_mj_cm2": dose_mj_cm2,
        "grid_size": grid_size,
        "pixel_nm": pixel_nm,
    }

    result = FullResult(aerial=aerial, contrast=contrast, config=config)

    result.nils = aerial.nils(threshold=0.3)

    if with_resist:
        result.resist_profile = engine.compute_resist_profile(
            dose_mj_cm2=dose_mj_cm2, focus_nm=focus_nm
        )

    return result


def simulate_contact_hole(
    diameter_nm: float,
    pitch_x_nm: float,
    pitch_y_nm: float | None = None,
    *,
    wavelength_nm: float = 157.63,
    na: float = 0.75,
    sigma: float = 0.7,
    focus_nm: float = 0.0,
    grid_size: int = 256,
    pixel_nm: float = 1.0,
    max_kernels: int = 20,
) -> FullResult:
    """Simulate a contact hole array pattern.

    Args:
        diameter_nm: Hole diameter at wafer (nm).
        pitch_x_nm: X-direction pitch (nm).
        pitch_y_nm: Y-direction pitch (nm). Defaults to pitch_x_nm.
        wavelength_nm: Source wavelength (nm).
        na: Numerical aperture.
        sigma: Partial coherence factor.
        focus_nm: Defocus (nm).
        grid_size: Simulation grid size.
        pixel_nm: Pixel size (nm).
        max_kernels: Max SOCS kernels.

    Returns:
        FullResult with aerial image and contrast.
    """
    from highuvlith._validation import (
        _validate_positive,
        _validate_power_of_two,
        _validate_range,
    )

    _validate_positive("diameter_nm", diameter_nm)
    _validate_positive("pitch_x_nm", pitch_x_nm)
    if pitch_y_nm is not None:
        _validate_positive("pitch_y_nm", pitch_y_nm)
    _validate_range("na", na, 0.01, 1.0)
    _validate_range("sigma", sigma, 0.01, 1.0)
    _validate_power_of_two("grid_size", grid_size)

    if pitch_y_nm is None:
        pitch_y_nm = pitch_x_nm

    source = huv.SourceConfig(wavelength_nm=wavelength_nm, sigma_outer=sigma)
    optics = huv.OpticsConfig(numerical_aperture=na)
    mask = huv.MaskConfig.contact_hole(diameter_nm, pitch_x_nm, pitch_y_nm)
    grid = huv.GridConfig(size=grid_size, pixel_nm=pixel_nm)

    engine = huv.SimulationEngine(source, optics, mask, grid=grid, max_kernels=max_kernels)
    aerial = engine.compute_aerial_image(focus_nm=focus_nm)

    config = {
        "wavelength_nm": wavelength_nm,
        "na": na,
        "sigma": sigma,
        "diameter_nm": diameter_nm,
        "pitch_x_nm": pitch_x_nm,
        "pitch_y_nm": pitch_y_nm,
        "focus_nm": focus_nm,
        "grid_size": grid_size,
        "pixel_nm": pixel_nm,
    }

    return FullResult(
        aerial=aerial,
        contrast=aerial.image_contrast(),
        config=config,
    )


def sweep_focus(
    cd_nm: float = 65.0,
    pitch_nm: float = 180.0,
    *,
    wavelength_nm: float = 157.63,
    na: float = 0.75,
    sigma: float = 0.7,
    focus_min: float = -300.0,
    focus_max: float = 300.0,
    focus_steps: int = 21,
    grid_size: int = 128,
    pixel_nm: float = 2.0,
) -> dict[str, Any]:
    """Sweep focus and return contrast vs focus data.

    Returns:
        Dict with 'focuses', 'contrasts' arrays and 'best_focus_nm'.
    """
    from highuvlith._validation import (
        _validate_positive,
        _validate_power_of_two,
        _validate_range,
    )

    if focus_min >= focus_max:
        raise ValueError(
            f"focus_min ({focus_min}) must be less than focus_max ({focus_max})"
        )
    if focus_steps < 2:
        raise ValueError(f"focus_steps must be >= 2, got {focus_steps}")
    _validate_positive("cd_nm", cd_nm)
    _validate_positive("pitch_nm", pitch_nm)
    _validate_range("na", na, 0.01, 1.0)
    _validate_range("sigma", sigma, 0.01, 1.0)
    _validate_power_of_two("grid_size", grid_size)

    source = huv.SourceConfig(wavelength_nm=wavelength_nm, sigma_outer=sigma)
    optics = huv.OpticsConfig(numerical_aperture=na)
    mask = huv.MaskConfig.line_space(cd_nm=cd_nm, pitch_nm=pitch_nm)
    grid = huv.GridConfig(size=grid_size, pixel_nm=pixel_nm)

    batch = huv.BatchSimulator(source, optics, mask, grid)
    focuses = np.linspace(focus_min, focus_max, focus_steps).tolist()
    results = batch.batch_defocus(focuses=focuses)

    contrasts = []
    for _f, img in results:
        arr = np.asarray(img)
        i_max = arr.max()
        i_min = arr.min()
        contrasts.append((i_max - i_min) / (i_max + i_min) if (i_max + i_min) > 0 else 0.0)

    contrasts = np.array(contrasts)
    best_idx = np.argmax(contrasts)

    return {
        "focuses": np.array(focuses),
        "contrasts": contrasts,
        "best_focus_nm": focuses[best_idx],
        "best_contrast": contrasts[best_idx],
    }
