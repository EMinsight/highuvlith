#!/usr/bin/env python3
"""
Example: Moiré Nanosphere Lithographic Reflection (MNSL) Simulation

This example demonstrates the MNSL capabilities of highuvlith for
enhanced emission control through nanosphere array rotation.
"""

import numpy as np
import matplotlib.pyplot as plt

import highuvlith as huv

def main():
    """Run MNSL simulation examples."""

    print("=== MNSL (Moiré Nanosphere Lithographic Reflection) Example ===\n")

    # Example 1: Basic MNSL Simulation
    print("1. Basic MNSL Simulation")
    print("-" * 40)

    # Simulate Moiré emission with silica nanospheres
    result = huv.simulate_moire_emission(
        sphere_diameter_nm=200.0,     # 200 nm diameter silica spheres
        array_pitch_nm=300.0,         # 300 nm center-to-center spacing
        rotation_angle_deg=5.0,       # 5 degree rotation between layers
        separation_nm=100.0,          # 100 nm vertical separation
        grid_size=256,                # 256×256 simulation grid
        pixel_nm=2.0,                 # 2 nm pixel size
    )

    print(f"Moiré period: {result.moire_period_nm:.1f} nm")
    print(f"Peak enhancement: {result.peak_enhancement:.2f}×")
    print(f"Total emission power: {result.total_emission_power:.2e}")
    print(f"Number of emission peaks: {result.num_peaks}")
    print()

    # Example 2: Parameter Sweep
    print("2. Rotation Angle Optimization")
    print("-" * 40)

    sweep_data = huv.sweep_rotation_angle(
        sphere_diameter_nm=200.0,
        array_pitch_nm=300.0,
        angle_min=0.0,
        angle_max=10.0,
        angle_steps=11,
        separation_nm=100.0,
        grid_size=128,
        pixel_nm=4.0,
    )

    print(f"Best rotation angle: {sweep_data['best_angle_deg']:.1f}°")
    print(f"Best enhancement: {sweep_data['best_enhancement']:.2f}×")
    print(f"Moiré period at optimum: {sweep_data['best_period_nm']:.1f} nm")
    print()

    # Example 3: Full Parameter Optimization
    print("3. Full Parameter Optimization")
    print("-" * 40)

    optimization = huv.optimize_moire_parameters(
        sphere_diameter_nm=200.0,
        array_pitch_nm=300.0,
        angle_range=(0.0, 10.0),
        separation_range=(50.0, 200.0),
        angle_steps=11,
        separation_steps=11,
        grid_size=128,
        pixel_nm=4.0,
    )

    print(f"Optimal rotation angle: {optimization['best_angle_deg']:.1f}°")
    print(f"Optimal separation: {optimization['best_separation_nm']:.1f} nm")
    print(f"Maximum enhancement: {optimization['max_enhancement']:.2f}×")
    print(f"Moiré period: {optimization['moire_period_nm']:.1f} nm")
    print()

    # Example 4: Material Comparison
    print("4. Material Comparison")
    print("-" * 40)

    # Compare silica vs polystyrene spheres
    silica_result = huv.simulate_moire_emission(
        200.0, 300.0, 5.0,
        sphere_material="silica",
        grid_size=128, pixel_nm=4.0
    )

    # For polystyrene, we need to use the low-level API
    ps_array_bottom = huv.create_nanosphere_array(
        200.0, 300.0, material="polystyrene"
    )
    ps_array_top = huv.create_nanosphere_array(
        200.0, 300.0, orientation_deg=5.0, material="polystyrene"
    )

    print(f"Silica spheres:")
    print(f"  Enhancement: {silica_result.peak_enhancement:.2f}×")
    print(f"  Total power: {silica_result.total_emission_power:.2e}")
    print()

    # Example 5: Cross-sections and Analysis
    print("5. Detailed Analysis")
    print("-" * 40)

    # Get cross-sections
    x_coords, x_emission = result.cross_section_x(y_nm=0.0)
    y_coords, y_emission = result.cross_section_y(x_nm=0.0)

    print(f"Cross-section statistics:")
    print(f"  X-direction FWHM: {estimate_fwhm(x_coords, x_emission):.1f} nm")
    print(f"  Y-direction FWHM: {estimate_fwhm(y_coords, y_emission):.1f} nm")
    print()

    # Create visualizations if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        print("6. Creating Visualizations")
        print("-" * 40)

        # Plot emission pattern
        fig1 = huv.viz.plot_emission_pattern(result, show_peaks=True)
        plt.savefig("mnsl_emission_pattern.png", dpi=300, bbox_inches="tight")
        print("Saved: mnsl_emission_pattern.png")

        # Plot comprehensive analysis
        fig2 = huv.viz.plot_moire_analysis(result)
        plt.savefig("mnsl_analysis.png", dpi=300, bbox_inches="tight")
        print("Saved: mnsl_analysis.png")

        # Plot rotation sweep
        fig3 = huv.viz.plot_rotation_sweep(sweep_data)
        plt.savefig("mnsl_rotation_sweep.png", dpi=300, bbox_inches="tight")
        print("Saved: mnsl_rotation_sweep.png")

        # Plot optimization heatmap
        fig4 = huv.viz.plot_optimization_heatmap(optimization)
        plt.savefig("mnsl_optimization.png", dpi=300, bbox_inches="tight")
        print("Saved: mnsl_optimization.png")

        plt.show()

    except ImportError:
        print("matplotlib not available - skipping visualizations")

    print("\n=== MNSL Example Complete ===")

def estimate_fwhm(coords, intensities):
    """Estimate Full Width Half Maximum of a 1D intensity profile."""
    max_intensity = np.max(intensities)
    half_max = max_intensity / 2.0

    # Find indices where intensity > half_max
    above_half = intensities > half_max
    indices = np.where(above_half)[0]

    if len(indices) < 2:
        return 0.0

    # FWHM is approximately the span of the region above half maximum
    coord_span = coords[indices[-1]] - coords[indices[0]]
    return coord_span

if __name__ == "__main__":
    main()