"""Process window visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highuvlith import ProcessWindowResult


def plot_bossung(
    pw: ProcessWindowResult,
    *,
    title: str | None = None,
    ax=None,
):
    """Plot Bossung curves (CD vs focus at multiple doses)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    doses = pw.doses
    focuses = pw.focuses
    cd_matrix = pw.cd_matrix

    for i, dose in enumerate(doses):
        cds = cd_matrix[i, :]
        ax.plot(focuses, cds, "-o", markersize=3, label=f"{dose:.1f} mJ/cm²")

    ax.set_xlabel("Focus (nm)")
    ax.set_ylabel("CD (nm)")
    ax.set_title(title or "Bossung Curves")
    ax.legend(title="Dose", fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_ed_window(
    pw: ProcessWindowResult,
    *,
    cd_target_nm: float = 65.0,
    cd_tolerance_pct: float = 10.0,
    title: str | None = None,
    ax=None,
):
    """Plot exposure-defocus (ED) process window contour."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    doses = pw.doses
    focuses = pw.focuses
    cd_matrix = pw.cd_matrix

    cd_min = cd_target_nm * (1 - cd_tolerance_pct / 100)
    cd_max = cd_target_nm * (1 + cd_tolerance_pct / 100)

    F, D = np.meshgrid(focuses, doses)

    # Contour fill showing CD
    cf = ax.contourf(F, D, cd_matrix, levels=20, cmap="RdYlGn_r")
    plt.colorbar(cf, ax=ax, label="CD (nm)")

    # Process window boundary
    ax.contour(F, D, cd_matrix, levels=[cd_min, cd_max], colors="black", linewidths=2)

    ax.set_xlabel("Focus (nm)")
    ax.set_ylabel("Dose (mJ/cm²)")
    ax.set_title(
        title
        or f"ED Window (target={cd_target_nm:.0f}nm ± {cd_tolerance_pct:.0f}%)"
    )

    return ax
