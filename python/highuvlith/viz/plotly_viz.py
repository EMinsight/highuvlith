"""Interactive Plotly-based visualizations (requires plotly)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from highuvlith import AerialImageResult, ProcessWindowResult


def plot_aerial_plotly(
    result: AerialImageResult,
    *,
    title: str | None = None,
    colorscale: str = "Inferno",
):
    """Interactive aerial image heatmap with hover coordinates and intensity."""
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=np.asarray(result.intensity),
            x=np.asarray(result.x_nm),
            y=np.asarray(result.y_nm),
            colorscale=colorscale,
            colorbar=dict(title="Intensity"),
            hovertemplate="x: %{x:.1f} nm<br>y: %{y:.1f} nm<br>I: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or "Aerial Image",
        xaxis_title="x (nm)",
        yaxis_title="y (nm)",
        width=700,
        height=600,
    )
    return fig


def plot_cross_section_plotly(
    result: AerialImageResult,
    *,
    y_nm: float = 0.0,
    threshold: float | None = None,
    title: str | None = None,
):
    """Interactive 1D cross-section with plotly."""
    import numpy as np
    import plotly.graph_objects as go

    x, intensity = result.cross_section(y_nm=y_nm)
    x_arr = np.asarray(x)
    i_arr = np.asarray(intensity)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=i_arr,
            mode="lines",
            name="Intensity",
            line=dict(color="blue", width=2),
        )
    )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold={threshold}",
        )

    fig.update_layout(
        title=title or f"Cross-Section at y={y_nm:.1f} nm",
        xaxis_title="x (nm)",
        yaxis_title="Intensity",
        width=900,
        height=400,
    )
    return fig


def plot_bossung_plotly(
    pw: ProcessWindowResult,
    *,
    title: str | None = None,
):
    """Interactive Bossung curves (CD vs focus at multiple doses)."""
    import numpy as np
    import plotly.graph_objects as go

    doses = np.asarray(pw.doses)
    focuses = np.asarray(pw.focuses)
    cd_matrix = np.asarray(pw.cd_matrix)

    fig = go.Figure()
    for i, dose in enumerate(doses):
        fig.add_trace(
            go.Scatter(
                x=focuses,
                y=cd_matrix[i, :],
                mode="lines+markers",
                name=f"{dose:.1f} mJ/cm²",
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title=title or "Bossung Curves",
        xaxis_title="Focus (nm)",
        yaxis_title="CD (nm)",
        width=900,
        height=600,
        legend_title="Dose",
    )
    return fig
