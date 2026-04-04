"""MNSL visualization utilities."""

from __future__ import annotations

from typing import Any


try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..mnsl import MnslSimResult


def plot_emission_pattern(
    result: MnslSimResult,
    *,
    backend: str = "matplotlib",
    show_peaks: bool = True,
    colormap: str = "plasma",
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> Any:
    """Plot 2D emission intensity pattern with optional peak markers.

    Args:
        result: MNSL simulation result.
        backend: Plotting backend ("matplotlib" or "plotly").
        show_peaks: Whether to mark peak positions. Default True.
        colormap: Colormap name. Default "plasma".
        figsize: Figure size for matplotlib. Default (10, 8).
        title: Custom plot title. If None, auto-generated.

    Returns:
        Figure object (matplotlib Figure or plotly Figure).
    """
    if backend == "matplotlib":
        return _plot_emission_matplotlib(result, show_peaks, colormap, figsize, title)
    elif backend == "plotly":
        return _plot_emission_plotly(result, show_peaks, colormap, title)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _plot_emission_matplotlib(
    result: MnslSimResult,
    show_peaks: bool,
    colormap: str,
    figsize: tuple[float, float],
    title: str | None,
) -> Any:
    """Matplotlib implementation of emission pattern plot."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig, ax = plt.subplots(figsize=figsize)

    x_coords, y_coords = result.coordinates
    emission = result.emission_pattern

    # Create 2D plot
    im = ax.imshow(
        emission,
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        origin="lower",
        cmap=colormap,
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Emission Intensity", rotation=270, labelpad=20)

    # Mark peaks if requested
    if show_peaks and result.peak_positions:
        peak_x, peak_y = zip(*result.peak_positions)
        ax.scatter(peak_x, peak_y, c="white", marker="x", s=100, linewidths=2)

    ax.set_xlabel("X Position (nm)")
    ax.set_ylabel("Y Position (nm)")

    if title is None:
        title = (
            f"MNSL Emission Pattern\n"
            f"Period: {result.moire_period_nm:.1f} nm, "
            f"Enhancement: {result.peak_enhancement:.2f}×"
        )
    ax.set_title(title)

    plt.tight_layout()
    return fig


def _plot_emission_plotly(
    result: MnslSimResult,
    show_peaks: bool,
    colormap: str,
    title: str | None,
) -> Any:
    """Plotly implementation of emission pattern plot."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotly backend")

    x_coords, y_coords = result.coordinates
    emission = result.emission_pattern

    fig = go.Figure()

    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=emission,
            x=x_coords,
            y=y_coords,
            colorscale=colormap,
            colorbar=dict(title="Emission Intensity"),
        )
    )

    # Mark peaks if requested
    if show_peaks and result.peak_positions:
        peak_x, peak_y = zip(*result.peak_positions)
        fig.add_trace(
            go.Scatter(
                x=peak_x,
                y=peak_y,
                mode="markers",
                marker=dict(symbol="x", size=10, color="white", line=dict(width=2)),
                name="Peaks",
                showlegend=True,
            )
        )

    if title is None:
        title = (
            f"MNSL Emission Pattern<br>"
            f"Period: {result.moire_period_nm:.1f} nm, "
            f"Enhancement: {result.peak_enhancement:.2f}×"
        )

    fig.update_layout(
        title=title,
        xaxis_title="X Position (nm)",
        yaxis_title="Y Position (nm)",
        width=800,
        height=600,
    )

    return fig


def plot_moire_analysis(
    result: MnslSimResult,
    *,
    backend: str = "matplotlib",
    figsize: tuple[float, float] = (15, 10),
) -> Any:
    """Plot comprehensive Moiré pattern analysis with multiple subplots.

    Args:
        result: MNSL simulation result.
        backend: Plotting backend ("matplotlib" or "plotly").
        figsize: Figure size for matplotlib. Default (15, 10).

    Returns:
        Figure object with subplots showing emission, Moiré pattern, and enhancement factors.
    """
    if backend == "matplotlib":
        return _plot_moire_analysis_matplotlib(result, figsize)
    elif backend == "plotly":
        return _plot_moire_analysis_plotly(result)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _plot_moire_analysis_matplotlib(result: MnslSimResult, figsize: tuple[float, float]) -> Any:
    """Matplotlib implementation of Moiré analysis plot."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    x_coords, y_coords = result.coordinates
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # Emission pattern
    im1 = ax1.imshow(result.emission_pattern, extent=extent, origin="lower", cmap="plasma")
    ax1.set_title("Emission Pattern")
    ax1.set_xlabel("X (nm)")
    ax1.set_ylabel("Y (nm)")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Moiré pattern
    im2 = ax2.imshow(result.moire_pattern, extent=extent, origin="lower", cmap="RdBu_r")
    ax2.set_title("Moiré Interference")
    ax2.set_xlabel("X (nm)")
    ax2.set_ylabel("Y (nm)")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Enhancement factors
    im3 = ax3.imshow(result.enhancement_factors, extent=extent, origin="lower", cmap="hot")
    ax3.set_title("Enhancement Factors")
    ax3.set_xlabel("X (nm)")
    ax3.set_ylabel("Y (nm)")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Cross-sections
    x_cross, emission_cross = result.cross_section_x(0.0)
    ax4.plot(x_cross, emission_cross, "b-", linewidth=2, label="Emission")
    ax4.set_xlabel("X Position (nm)")
    ax4.set_ylabel("Intensity")
    ax4.set_title("Cross-section at Y=0")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    return fig


def _plot_moire_analysis_plotly(result: MnslSimResult) -> Any:
    """Plotly implementation of Moiré analysis plot."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotly backend")

    x_coords, y_coords = result.coordinates

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Emission Pattern",
            "Moiré Interference",
            "Enhancement Factors",
            "Cross-section at Y=0",
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
        ],
    )

    # Emission pattern
    fig.add_trace(
        go.Heatmap(
            z=result.emission_pattern,
            x=x_coords,
            y=y_coords,
            colorscale="plasma",
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Moiré pattern
    fig.add_trace(
        go.Heatmap(
            z=result.moire_pattern,
            x=x_coords,
            y=y_coords,
            colorscale="RdBu",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Enhancement factors
    fig.add_trace(
        go.Heatmap(
            z=result.enhancement_factors,
            x=x_coords,
            y=y_coords,
            colorscale="hot",
            showscale=True,
        ),
        row=2,
        col=1,
    )

    # Cross-section
    x_cross, emission_cross = result.cross_section_x(0.0)
    fig.add_trace(
        go.Scatter(x=x_cross, y=emission_cross, mode="lines", name="Emission"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"MNSL Analysis - Period: {result.moire_period_nm:.1f} nm",
        width=1200,
        height=800,
        showlegend=False,
    )

    return fig


def plot_rotation_sweep(
    sweep_data: dict[str, Any],
    *,
    backend: str = "matplotlib",
    figsize: tuple[float, float] = (12, 8),
) -> Any:
    """Plot results from rotation angle parameter sweep.

    Args:
        sweep_data: Results from sweep_rotation_angle() function.
        backend: Plotting backend ("matplotlib" or "plotly").
        figsize: Figure size for matplotlib. Default (12, 8).

    Returns:
        Figure object showing enhancement vs rotation angle.
    """
    if backend == "matplotlib":
        return _plot_rotation_sweep_matplotlib(sweep_data, figsize)
    elif backend == "plotly":
        return _plot_rotation_sweep_plotly(sweep_data)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _plot_rotation_sweep_matplotlib(sweep_data: dict[str, Any], figsize: tuple[float, float]) -> Any:
    """Matplotlib implementation of rotation sweep plot."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    angles = sweep_data["angles"]
    enhancements = sweep_data["peak_enhancements"]
    periods = sweep_data["moire_periods"]
    best_angle = sweep_data["best_angle_deg"]
    best_enhancement = sweep_data["best_enhancement"]

    # Enhancement vs angle
    ax1.plot(angles, enhancements, "b-o", markersize=6, linewidth=2)
    ax1.axvline(best_angle, color="r", linestyle="--", alpha=0.7)
    ax1.scatter([best_angle], [best_enhancement], color="red", s=100, zorder=5)
    ax1.set_xlabel("Rotation Angle (degrees)")
    ax1.set_ylabel("Peak Enhancement Factor")
    ax1.set_title("Enhancement vs Rotation Angle")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        best_angle,
        best_enhancement * 1.05,
        f"Best: {best_angle:.1f}°",
        ha="center",
        fontweight="bold",
    )

    # Moiré period vs angle
    ax2.plot(angles, periods, "g-s", markersize=6, linewidth=2)
    ax2.axvline(best_angle, color="r", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Rotation Angle (degrees)")
    ax2.set_ylabel("Moiré Period (nm)")
    ax2.set_title("Moiré Period vs Rotation Angle")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _plot_rotation_sweep_plotly(sweep_data: dict[str, Any]) -> Any:
    """Plotly implementation of rotation sweep plot."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotly backend")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Enhancement vs Rotation Angle", "Moiré Period vs Rotation Angle"],
    )

    angles = sweep_data["angles"]
    enhancements = sweep_data["peak_enhancements"]
    periods = sweep_data["moire_periods"]
    best_angle = sweep_data["best_angle_deg"]
    best_enhancement = sweep_data["best_enhancement"]

    # Enhancement vs angle
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=enhancements,
            mode="lines+markers",
            name="Enhancement",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Best point
    fig.add_trace(
        go.Scatter(
            x=[best_angle],
            y=[best_enhancement],
            mode="markers",
            marker=dict(size=12, color="red"),
            name=f"Best: {best_angle:.1f}°",
        ),
        row=1,
        col=1,
    )

    # Moiré period vs angle
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=periods,
            mode="lines+markers",
            name="Period",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Rotation Angle (degrees)", row=1, col=1)
    fig.update_yaxes(title_text="Peak Enhancement Factor", row=1, col=1)
    fig.update_xaxes(title_text="Rotation Angle (degrees)", row=1, col=2)
    fig.update_yaxes(title_text="Moiré Period (nm)", row=1, col=2)

    fig.update_layout(
        title="MNSL Rotation Angle Sweep",
        width=1000,
        height=500,
        showlegend=True,
    )

    return fig


def plot_optimization_heatmap(
    optimization_data: dict[str, Any],
    *,
    backend: str = "matplotlib",
    figsize: tuple[float, float] = (10, 8),
) -> Any:
    """Plot 2D optimization heatmap for angle vs separation.

    Args:
        optimization_data: Results from optimize_moire_parameters() function.
        backend: Plotting backend ("matplotlib" or "plotly").
        figsize: Figure size for matplotlib. Default (10, 8).

    Returns:
        Figure object showing enhancement heatmap.
    """
    if backend == "matplotlib":
        return _plot_optimization_heatmap_matplotlib(optimization_data, figsize)
    elif backend == "plotly":
        return _plot_optimization_heatmap_plotly(optimization_data)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _plot_optimization_heatmap_matplotlib(
    optimization_data: dict[str, Any], figsize: tuple[float, float]
) -> Any:
    """Matplotlib implementation of optimization heatmap."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for matplotlib backend")

    fig, ax = plt.subplots(figsize=figsize)

    angles = optimization_data["angles"]
    separations = optimization_data["separations"]
    enhancement_grid = optimization_data["enhancement_grid"]
    best_angle = optimization_data["best_angle_deg"]
    best_separation = optimization_data["best_separation_nm"]

    # Create heatmap
    extent = [separations[0], separations[-1], angles[0], angles[-1]]
    im = ax.imshow(
        enhancement_grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="plasma",
    )

    # Mark optimum
    ax.plot(best_separation, best_angle, "w*", markersize=20, markeredgecolor="black")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Peak Enhancement Factor", rotation=270, labelpad=20)

    ax.set_xlabel("Layer Separation (nm)")
    ax.set_ylabel("Rotation Angle (degrees)")
    ax.set_title(
        f"MNSL Parameter Optimization\n"
        f"Optimum: {best_angle:.1f}° × {best_separation:.0f} nm "
        f"(Enhancement: {optimization_data['max_enhancement']:.2f}×)"
    )

    plt.tight_layout()
    return fig


def _plot_optimization_heatmap_plotly(optimization_data: dict[str, Any]) -> Any:
    """Plotly implementation of optimization heatmap."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotly backend")

    angles = optimization_data["angles"]
    separations = optimization_data["separations"]
    enhancement_grid = optimization_data["enhancement_grid"]
    best_angle = optimization_data["best_angle_deg"]
    best_separation = optimization_data["best_separation_nm"]

    fig = go.Figure()

    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=enhancement_grid,
            x=separations,
            y=angles,
            colorscale="plasma",
            colorbar=dict(title="Peak Enhancement Factor"),
        )
    )

    # Mark optimum
    fig.add_trace(
        go.Scatter(
            x=[best_separation],
            y=[best_angle],
            mode="markers",
            marker=dict(symbol="star", size=20, color="white", line=dict(width=2, color="black")),
            name=f"Optimum: {best_angle:.1f}° × {best_separation:.0f} nm",
        )
    )

    fig.update_layout(
        title=(
            f"MNSL Parameter Optimization<br>"
            f"Optimum: {best_angle:.1f}° × {best_separation:.0f} nm "
            f"(Enhancement: {optimization_data['max_enhancement']:.2f}×)"
        ),
        xaxis_title="Layer Separation (nm)",
        yaxis_title="Rotation Angle (degrees)",
        width=800,
        height=600,
    )

    return fig