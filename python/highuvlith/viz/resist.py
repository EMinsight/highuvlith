"""Resist profile visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highuvlith import ResistProfileResult


def plot_resist_profile(
    result: ResistProfileResult,
    *,
    title: str | None = None,
    ax=None,
):
    """Plot developed resist profile."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    x = result.x_nm
    height = result.height_nm
    thickness = result.thickness_nm

    # Plot as filled region
    ax.fill_between(x, 0, height, alpha=0.6, color="steelblue", label="Resist")
    ax.plot(x, height, "b-", linewidth=1.5)

    # Show original thickness
    ax.axhline(
        y=thickness,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Original thickness ({thickness:.0f} nm)",
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Resist Height (nm)")
    ax.set_ylim(0, thickness * 1.1)
    ax.set_title(title or "Developed Resist Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
