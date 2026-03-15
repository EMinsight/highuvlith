"""Aerial image visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highuvlith import AerialImageResult


def plot_aerial(
    result: AerialImageResult,
    *,
    title: str | None = None,
    cmap: str = "inferno",
    show_colorbar: bool = True,
    ax=None,
):
    """Plot 2D aerial image intensity."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = result.x_nm
    y = result.y_nm
    intensity = result.intensity

    im = ax.imshow(
        intensity,
        extent=[x[0], x[-1], y[-1], y[0]],
        cmap=cmap,
        aspect="equal",
    )
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(title or "Aerial Image Intensity")

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Intensity")

    return ax


def plot_cross_section(
    result: AerialImageResult,
    *,
    y_nm: float = 0.0,
    threshold: float | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot 1D cross-section of aerial image along x at given y."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    x, intensity = result.cross_section(y_nm=y_nm)

    ax.plot(x, intensity, "b-", linewidth=1.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(title or f"Aerial Image Cross-Section (y={y_nm:.1f} nm)")
    ax.grid(True, alpha=0.3)

    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"Threshold={threshold}")
        ax.legend()

    return ax
