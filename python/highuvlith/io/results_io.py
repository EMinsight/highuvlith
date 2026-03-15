"""Results serialization (NPZ)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_result(
    result,
    path: str | Path,
) -> None:
    """Save simulation result to .npz file."""
    path = Path(path)
    data = {}

    if hasattr(result, "intensity"):
        data["intensity"] = np.asarray(result.intensity)
    if hasattr(result, "x_nm"):
        data["x_nm"] = np.asarray(result.x_nm)
    if hasattr(result, "y_nm"):
        data["y_nm"] = np.asarray(result.y_nm)
    if hasattr(result, "height_nm"):
        data["height_nm"] = np.asarray(result.height_nm)
    if hasattr(result, "cd_matrix"):
        data["cd_matrix"] = np.asarray(result.cd_matrix)
    if hasattr(result, "doses"):
        data["doses"] = np.asarray(result.doses)
    if hasattr(result, "focuses"):
        data["focuses"] = np.asarray(result.focuses)

    np.savez(path, **data)


def load_result(path: str | Path) -> dict[str, np.ndarray]:
    """Load simulation result from .npz file."""
    path = Path(path)
    return dict(np.load(path))
