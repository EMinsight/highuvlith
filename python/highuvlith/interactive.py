"""Interactive Jupyter widgets for parameter exploration (requires ipywidgets)."""

from __future__ import annotations


def interactive_aerial(
    wavelength_range: tuple[float, float] = (126.0, 170.0),
    na_range: tuple[float, float] = (0.3, 0.95),
    sigma_range: tuple[float, float] = (0.1, 1.0),
    cd_range: tuple[float, float] = (30.0, 200.0),
    pitch_range: tuple[float, float] = (60.0, 500.0),
    focus_range: tuple[float, float] = (-500.0, 500.0),
    grid_size: int = 128,
):
    """Create an interactive aerial image explorer with ipywidgets sliders.

    Displays a 2D aerial image and 1D cross-section that update live
    as parameters are adjusted.

    Requires: ``pip install highuvlith[notebook]``
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    import highuvlith as huv

    wavelength = widgets.FloatSlider(
        value=157.63,
        min=wavelength_range[0],
        max=wavelength_range[1],
        step=0.1,
        description="λ (nm):",
        continuous_update=False,
    )
    na = widgets.FloatSlider(
        value=0.75,
        min=na_range[0],
        max=na_range[1],
        step=0.01,
        description="NA:",
        continuous_update=False,
    )
    sigma = widgets.FloatSlider(
        value=0.7,
        min=sigma_range[0],
        max=sigma_range[1],
        step=0.05,
        description="σ:",
        continuous_update=False,
    )
    cd = widgets.FloatSlider(
        value=65.0,
        min=cd_range[0],
        max=cd_range[1],
        step=1.0,
        description="CD (nm):",
        continuous_update=False,
    )
    pitch = widgets.FloatSlider(
        value=180.0,
        min=pitch_range[0],
        max=pitch_range[1],
        step=5.0,
        description="Pitch (nm):",
        continuous_update=False,
    )
    focus = widgets.FloatSlider(
        value=0.0,
        min=focus_range[0],
        max=focus_range[1],
        step=10.0,
        description="Focus (nm):",
        continuous_update=False,
    )

    output = widgets.Output()

    def update(_change=None):
        with output:
            output.clear_output(wait=True)
            src = huv.SourceConfig(
                wavelength_nm=wavelength.value, sigma_outer=sigma.value
            )
            opt = huv.OpticsConfig(numerical_aperture=na.value)
            msk = huv.MaskConfig.line_space(cd_nm=cd.value, pitch_nm=pitch.value)
            grid = huv.GridConfig(size=grid_size, pixel_nm=2.0)
            engine = huv.SimulationEngine(src, opt, msk, grid=grid)
            result = engine.compute_aerial_image(focus_nm=focus.value)

            intensity = np.asarray(result.intensity)
            x_arr, i_arr = result.cross_section(y_nm=0.0)
            x_arr = np.asarray(x_arr)
            i_arr = np.asarray(i_arr)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Aerial image
            im = ax1.imshow(
                intensity,
                extent=[
                    np.asarray(result.x_nm)[0],
                    np.asarray(result.x_nm)[-1],
                    np.asarray(result.y_nm)[-1],
                    np.asarray(result.y_nm)[0],
                ],
                cmap="inferno",
                aspect="equal",
            )
            ax1.set_xlabel("x (nm)")
            ax1.set_ylabel("y (nm)")
            ax1.set_title("Aerial Image")
            plt.colorbar(im, ax=ax1, label="Intensity")

            # Cross-section
            ax2.plot(x_arr, i_arr, "b-", linewidth=1.5)
            ax2.set_xlabel("x (nm)")
            ax2.set_ylabel("Intensity")
            ax2.set_title(f"Cross-Section (contrast={result.image_contrast():.3f})")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="Threshold")
            ax2.legend()

            plt.tight_layout()
            plt.show()

    for w in [wavelength, na, sigma, cd, pitch, focus]:
        w.observe(update, names="value")

    controls = widgets.VBox(
        [
            widgets.HTML("<h3>VUV Lithography Simulator</h3>"),
            wavelength,
            na,
            sigma,
            cd,
            pitch,
            focus,
        ]
    )
    display(widgets.HBox([controls, output]))
    update()


def interactive_focus_sweep(
    cd_nm: float = 65.0,
    pitch_nm: float = 180.0,
    grid_size: int = 128,
):
    """Interactive focus sweep showing contrast vs focus curve.

    Requires: ``pip install highuvlith[notebook]``
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    import highuvlith as huv

    wavelength = widgets.FloatSlider(
        value=157.63, min=126.0, max=170.0, step=0.1,
        description="λ (nm):", continuous_update=False,
    )
    na = widgets.FloatSlider(
        value=0.75, min=0.3, max=0.95, step=0.01,
        description="NA:", continuous_update=False,
    )
    sigma = widgets.FloatSlider(
        value=0.7, min=0.1, max=1.0, step=0.05,
        description="σ:", continuous_update=False,
    )

    output = widgets.Output()

    def update(_change=None):
        with output:
            output.clear_output(wait=True)
            result = huv.sweep_focus(
                cd_nm=cd_nm,
                pitch_nm=pitch_nm,
                wavelength_nm=wavelength.value,
                na=na.value,
                sigma=sigma.value,
                focus_steps=21,
                grid_size=grid_size,
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result["focuses"], result["contrasts"], "bo-", linewidth=2, markersize=4)
            ax.axvline(x=result["best_focus_nm"], color="r", linestyle="--",
                       alpha=0.7, label=f'Best focus: {result["best_focus_nm"]:.0f} nm')
            ax.set_xlabel("Focus (nm)")
            ax.set_ylabel("Image Contrast")
            ax.set_title(f"Focus Sweep — CD={cd_nm}nm, Pitch={pitch_nm}nm")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.show()

    for w in [wavelength, na, sigma]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<h3>Focus Sweep</h3>"),
        wavelength, na, sigma,
    ])
    display(widgets.HBox([controls, output]))
    update()
