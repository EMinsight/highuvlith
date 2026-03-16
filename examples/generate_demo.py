#!/usr/bin/env python3
"""Generate demo videos showcasing VUV lithography simulation.

Produces VP9 (.webm) and MPEG-4 (.mp4) animations of contrived
lithographic application sequences.

Usage:
    python examples/generate_demo.py

Outputs:
    examples/demo_vp9.webm
    examples/demo_mpeg4.mp4
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import highuvlith as huv

# --- Constants ---
FPS = 24
DPI = 150
FIG_W, FIG_H = 12.8, 7.2  # inches → 1920×1080 at 150 DPI
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#f0a030"
CMAP = "inferno"

BANNER = "highuvlith  ·  VUV Lithography Simulator  ·  λ = 157.63 nm"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#555555",
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": "#333333",
        "font.size": 12,
        "axes.titlesize": 14,
        "figure.titlesize": 16,
    })


def save_frame(fig, frames_dir: Path, idx: int):
    fig.savefig(frames_dir / f"{idx:04d}.png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def add_banner(fig, text: str = BANNER):
    fig.text(0.5, 0.01, text, ha="center", va="bottom",
             fontsize=9, color="#888888", family="monospace")


# --- Title & transition cards ---

def render_title_card(frames_dir: Path, start: int, text: str,
                      subtitle: str = "", num_frames: int = 48) -> int:
    for i in range(num_frames):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        alpha = min(1.0, i / 12.0)  # fade in
        if i > num_frames - 12:
            alpha = max(0.0, (num_frames - i) / 12.0)  # fade out
        fig.text(0.5, 0.52, text, ha="center", va="center",
                 fontsize=36, fontweight="bold", color=TEXT_COLOR, alpha=alpha)
        if subtitle:
            fig.text(0.5, 0.40, subtitle, ha="center", va="center",
                     fontsize=18, color=ACCENT_COLOR, alpha=alpha)
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)
    return start + num_frames


def render_transition(frames_dir: Path, start: int, num_frames: int = 12) -> int:
    for i in range(num_frames):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)
    return start + num_frames


def render_section_title(frames_dir: Path, start: int, title: str,
                         num_frames: int = 30) -> int:
    for i in range(num_frames):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        alpha = min(1.0, i / 8.0)
        if i > num_frames - 8:
            alpha = max(0.0, (num_frames - i) / 8.0)
        fig.text(0.5, 0.5, title, ha="center", va="center",
                 fontsize=28, color=ACCENT_COLOR, alpha=alpha)
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)
    return start + num_frames


# --- Sequence 1: Through-Focus ---

def render_through_focus(frames_dir: Path, start: int) -> int:
    print("  Rendering Sequence 1: Through-Focus...")
    source = huv.SourceConfig.f2_laser(sigma=0.7)
    optics = huv.OpticsConfig(numerical_aperture=0.75)
    mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
    grid = huv.GridConfig(size=128, pixel_nm=2.0)
    engine = huv.SimulationEngine(source, optics, mask, grid=grid)

    focuses = np.linspace(-400, 400, 120)

    for i, focus in enumerate(focuses):
        result = engine.compute_aerial_image(focus_nm=float(focus))
        intensity = np.asarray(result.intensity)
        x, cross = result.cross_section(y_nm=0.0)
        x = np.asarray(x)
        cross = np.asarray(cross)
        contrast = result.image_contrast()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                                        gridspec_kw={"width_ratios": [1, 1.2]})
        fig.suptitle("Through-Focus Aerial Image", fontsize=18, fontweight="bold")

        # 2D aerial image
        ax1.imshow(intensity, cmap=CMAP, vmin=0, vmax=0.7,
                   extent=[x[0], x[-1], x[-1], x[0]], aspect="equal")
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        ax1.set_title("Aerial Image")

        # Cross-section
        ax2.plot(x, cross, color="#4fc3f7", linewidth=2)
        ax2.fill_between(x, 0, cross, alpha=0.15, color="#4fc3f7")
        ax2.axhline(y=0.3, color="#ff5555", linestyle="--", alpha=0.5, linewidth=1)
        ax2.set_xlim(x[0], x[-1])
        ax2.set_ylim(0, 0.8)
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("Intensity")
        ax2.set_title("Cross-Section at y = 0")
        ax2.grid(True, alpha=0.2)

        # Annotations
        ax2.text(0.97, 0.95, f"Focus: {focus:+.0f} nm",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=14, fontweight="bold", color=ACCENT_COLOR)
        ax2.text(0.97, 0.87, f"Contrast: {contrast:.3f}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=13, color=TEXT_COLOR)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)

    print(f"    → {len(focuses)} frames")
    return start + len(focuses)


# --- Sequence 2: Wavelength Comparison ---

def render_wavelength_comparison(frames_dir: Path, start: int) -> int:
    print("  Rendering Sequence 2: Wavelength Comparison (157nm vs 193nm)...")
    cds = np.linspace(80, 30, 100)
    grid = huv.GridConfig(size=128, pixel_nm=2.0)

    for i, cd in enumerate(cds):
        pitch = cd * 2.5

        # 157nm VUV
        src_vuv = huv.SourceConfig(wavelength_nm=157.63, sigma_outer=0.7)
        opt_vuv = huv.OpticsConfig(numerical_aperture=0.75)
        mask_vuv = huv.MaskConfig.line_space(cd_nm=float(cd), pitch_nm=float(pitch))
        eng_vuv = huv.SimulationEngine(src_vuv, opt_vuv, mask_vuv, grid=grid)
        res_vuv = eng_vuv.compute_aerial_image()
        x_vuv, cs_vuv = res_vuv.cross_section(y_nm=0.0)
        c_vuv = res_vuv.image_contrast()

        # 193nm DUV (simulated — same optics, different wavelength)
        src_duv = huv.SourceConfig(wavelength_nm=193.0, sigma_outer=0.7)
        try:
            eng_duv = huv.SimulationEngine(src_duv, opt_vuv, mask_vuv, grid=grid)
            res_duv = eng_duv.compute_aerial_image()
            x_duv, cs_duv = res_duv.cross_section(y_nm=0.0)
            c_duv = res_duv.image_contrast()
        except Exception:
            x_duv, cs_duv = np.asarray(x_vuv), np.ones_like(np.asarray(cs_vuv)) * 0.5
            c_duv = 0.0

        x_vuv, cs_vuv = np.asarray(x_vuv), np.asarray(cs_vuv)
        x_duv, cs_duv = np.asarray(x_duv), np.asarray(cs_duv)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
        fig.suptitle(f"Wavelength Comparison — CD = {cd:.1f} nm, Pitch = {pitch:.0f} nm",
                     fontsize=18, fontweight="bold")

        ax1.plot(x_vuv, cs_vuv, color="#ff9800", linewidth=2)
        ax1.fill_between(x_vuv, 0, cs_vuv, alpha=0.15, color="#ff9800")
        ax1.set_xlim(x_vuv[0], x_vuv[-1])
        ax1.set_ylim(0, 0.85)
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("Intensity")
        ax1.set_title("λ = 157.63 nm (VUV)", color="#ff9800")
        ax1.grid(True, alpha=0.2)
        ax1.text(0.97, 0.95, f"Contrast: {c_vuv:.3f}",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=13, color="#ff9800", fontweight="bold")

        ax2.plot(x_duv, cs_duv, color="#4fc3f7", linewidth=2)
        ax2.fill_between(x_duv, 0, cs_duv, alpha=0.15, color="#4fc3f7")
        ax2.set_xlim(x_duv[0], x_duv[-1])
        ax2.set_xlabel("x (nm)")
        ax2.set_title("λ = 193 nm (DUV)", color="#4fc3f7")
        ax2.grid(True, alpha=0.2)
        ax2.text(0.97, 0.95, f"Contrast: {c_duv:.3f}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=13, color="#4fc3f7", fontweight="bold")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)

    print(f"    → {len(cds)} frames")
    return start + len(cds)


# --- Sequence 3: NA Exploration ---

def render_na_exploration(frames_dir: Path, start: int) -> int:
    print("  Rendering Sequence 3: NA Exploration...")
    nas = np.linspace(0.5, 0.9, 90)
    grid = huv.GridConfig(size=128, pixel_nm=2.0)

    for i, na in enumerate(nas):
        source = huv.SourceConfig.f2_laser(sigma=0.7)
        optics = huv.OpticsConfig(numerical_aperture=float(na))
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid)

        result = engine.compute_aerial_image()
        intensity = np.asarray(result.intensity)
        x, cross = result.cross_section(y_nm=0.0)
        x, cross = np.asarray(x), np.asarray(cross)
        contrast = result.image_contrast()

        # Focus sweep for DOF visualization
        focus_pts = np.linspace(-300, 300, 11)
        contrasts = []
        for f in focus_pts:
            contrasts.append(engine.image_contrast(focus_nm=float(f)))

        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
        fig.suptitle("NA Exploration — Resolution vs Depth of Focus",
                     fontsize=18, fontweight="bold")

        # Top: aerial image
        ax_aerial = fig.add_subplot(gs[0, :])
        ax_aerial.imshow(intensity, cmap=CMAP, vmin=0, vmax=0.7,
                         extent=[x[0], x[-1], x[-1], x[0]], aspect="auto")
        ax_aerial.set_xlabel("x (nm)")
        ax_aerial.set_ylabel("y (nm)")
        ax_aerial.set_title(f"Aerial Image — NA = {na:.2f}")

        # Bottom-left: cross-section
        ax_cs = fig.add_subplot(gs[1, 0])
        ax_cs.plot(x, cross, color="#4fc3f7", linewidth=2)
        ax_cs.fill_between(x, 0, cross, alpha=0.15, color="#4fc3f7")
        ax_cs.set_xlim(x[0], x[-1])
        ax_cs.set_ylim(0, 0.8)
        ax_cs.set_xlabel("x (nm)")
        ax_cs.set_ylabel("Intensity")
        ax_cs.set_title(f"Cross-Section (Contrast: {contrast:.3f})")
        ax_cs.grid(True, alpha=0.2)

        # Bottom-right: contrast vs focus
        ax_dof = fig.add_subplot(gs[1, 1])
        ax_dof.plot(focus_pts, contrasts, "o-", color=ACCENT_COLOR, linewidth=2, markersize=4)
        ax_dof.set_xlabel("Focus (nm)")
        ax_dof.set_ylabel("Contrast")
        ax_dof.set_title("Process Window")
        ax_dof.set_ylim(0, 1)
        ax_dof.grid(True, alpha=0.2)

        # Rayleigh resolution annotation
        rayleigh = 0.61 * 157.63 / na
        fig.text(0.98, 0.97, f"Rayleigh: {rayleigh:.0f} nm",
                 ha="right", va="top", fontsize=12, color=ACCENT_COLOR)

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)

    print(f"    → {len(nas)} frames")
    return start + len(nas)


# --- Sequence 4: Sigma Sweep ---

def render_sigma_sweep(frames_dir: Path, start: int) -> int:
    print("  Rendering Sequence 4: Partial Coherence (σ) Sweep...")
    sigmas = np.linspace(0.1, 0.95, 80)
    grid = huv.GridConfig(size=128, pixel_nm=2.0)

    for i, sigma in enumerate(sigmas):
        source = huv.SourceConfig(wavelength_nm=157.63, sigma_outer=float(sigma))
        optics = huv.OpticsConfig(numerical_aperture=0.75)
        mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
        engine = huv.SimulationEngine(source, optics, mask, grid=grid)

        result = engine.compute_aerial_image()
        intensity = np.asarray(result.intensity)
        x, cross = result.cross_section(y_nm=0.0)
        x, cross = np.asarray(x), np.asarray(cross)
        contrast = result.image_contrast()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                                        gridspec_kw={"width_ratios": [1, 1.2]})
        fig.suptitle("Partial Coherence Sweep", fontsize=18, fontweight="bold")

        ax1.imshow(intensity, cmap=CMAP, vmin=0, vmax=0.7,
                   extent=[x[0], x[-1], x[-1], x[0]], aspect="equal")
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        ax1.set_title("Aerial Image")

        ax2.plot(x, cross, color="#76ff03", linewidth=2)
        ax2.fill_between(x, 0, cross, alpha=0.15, color="#76ff03")
        ax2.set_xlim(x[0], x[-1])
        ax2.set_ylim(0, 0.8)
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("Intensity")
        ax2.set_title("Cross-Section at y = 0")
        ax2.grid(True, alpha=0.2)

        ax2.text(0.97, 0.95, f"σ = {sigma:.2f}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=16, fontweight="bold", color=ACCENT_COLOR)
        ax2.text(0.97, 0.86, f"Contrast: {contrast:.3f}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=13, color=TEXT_COLOR)

        label = "coherent" if sigma < 0.2 else "partially coherent" if sigma < 0.8 else "incoherent"
        ax2.text(0.97, 0.78, f"({label})",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=11, color="#888888", style="italic")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)

    print(f"    → {len(sigmas)} frames")
    return start + len(sigmas)


# --- Sequence 5: Resist Development ---

def render_resist_development(frames_dir: Path, start: int) -> int:
    print("  Rendering Sequence 5: Resist Development Time-Lapse...")
    source = huv.SourceConfig.f2_laser(sigma=0.7)
    optics = huv.OpticsConfig(numerical_aperture=0.75)
    mask = huv.MaskConfig.line_space(cd_nm=65.0, pitch_nm=180.0)
    resist = huv.ResistConfig.vuv_fluoropolymer()
    grid = huv.GridConfig(size=256, pixel_nm=1.0)
    engine = huv.SimulationEngine(source, optics, mask, resist, grid)

    dev_times = np.linspace(0.1, 120.0, 60)

    for i, dt in enumerate(dev_times):
        profile = engine.compute_resist_profile(dose_mj_cm2=30.0, focus_nm=0.0,
                                                 dev_time_s=float(dt))
        x = np.asarray(profile.x_nm)
        h = np.asarray(profile.height_nm)
        thickness = profile.thickness_nm

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.suptitle("Resist Development Time-Lapse", fontsize=18, fontweight="bold")

        # Substrate
        ax.axhspan(-10, 0, color="#555555", alpha=0.8)
        ax.text(0.5, -5, "Si Substrate", ha="center", va="center",
                fontsize=10, color="#aaaaaa")

        # Resist profile
        ax.fill_between(x, 0, h, color="#1e88e5", alpha=0.7, label="Resist")
        ax.plot(x, h, color="#42a5f5", linewidth=1.5)

        # Original thickness line
        ax.axhline(y=thickness, color="#888888", linestyle="--", alpha=0.5,
                   linewidth=1, label=f"Original ({thickness:.0f} nm)")

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-15, thickness * 1.15)
        ax.set_xlabel("x (nm)", fontsize=13)
        ax.set_ylabel("Height (nm)", fontsize=13)
        ax.set_title(f"Developed Resist Profile — CD = 65 nm, Pitch = 180 nm")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

        ax.text(0.97, 0.80, f"Dev Time: {dt:.1f} s",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=16, fontweight="bold", color=ACCENT_COLOR)
        ax.text(0.97, 0.72, f"Dose: 30 mJ/cm²",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color=TEXT_COLOR)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        add_banner(fig)
        save_frame(fig, frames_dir, start + i)

    print(f"    → {len(dev_times)} frames")
    return start + len(dev_times)


# --- Video encoding ---

def encode_video(frames_dir: Path, total_frames: int, output_path: Path,
                 codec: str, extra_args: list[str] | None = None):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "%04d.png"),
        "-frames:v", str(total_frames),
        "-pix_fmt", "yuv420p",
        "-c:v", codec,
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(output_path))

    print(f"  Encoding {output_path.name} ({codec})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-500:]}")
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    → {output_path.name}: {size_mb:.1f} MB")


# --- Main ---

def main():
    setup_style()
    out_dir = Path(__file__).parent
    frames_dir = Path(tempfile.mkdtemp(prefix="highuvlith_demo_"))
    print(f"Rendering frames to {frames_dir}")

    idx = 0

    # Title card
    idx = render_title_card(frames_dir, idx, "highuvlith",
                            subtitle="VUV Laser Lithography Simulator", num_frames=48)
    idx = render_transition(frames_dir, idx)

    # Sequence 1
    idx = render_section_title(frames_dir, idx, "1 · Through-Focus")
    idx = render_through_focus(frames_dir, idx)
    idx = render_transition(frames_dir, idx)

    # Sequence 2
    idx = render_section_title(frames_dir, idx, "2 · VUV vs DUV Wavelength")
    idx = render_wavelength_comparison(frames_dir, idx)
    idx = render_transition(frames_dir, idx)

    # Sequence 3
    idx = render_section_title(frames_dir, idx, "3 · Numerical Aperture")
    idx = render_na_exploration(frames_dir, idx)
    idx = render_transition(frames_dir, idx)

    # Sequence 4
    idx = render_section_title(frames_dir, idx, "4 · Partial Coherence")
    idx = render_sigma_sweep(frames_dir, idx)
    idx = render_transition(frames_dir, idx)

    # Sequence 5
    idx = render_section_title(frames_dir, idx, "5 · Resist Development")
    idx = render_resist_development(frames_dir, idx)

    # End card
    idx = render_transition(frames_dir, idx)
    idx = render_title_card(frames_dir, idx, "highuvlith",
                            subtitle="github.com/martinpeck/highuvlith", num_frames=48)

    total_frames = idx
    duration = total_frames / FPS
    print(f"\nTotal: {total_frames} frames ({duration:.1f}s at {FPS}fps)")

    # Encode VP9
    vp9_path = out_dir / "demo_vp9.webm"
    encode_video(frames_dir, total_frames, vp9_path,
                 "libvpx-vp9", ["-b:v", "2M", "-threads", "4"])

    # Encode MPEG-4
    mp4_path = out_dir / "demo_mpeg4.mp4"
    encode_video(frames_dir, total_frames, mp4_path,
                 "mpeg4", ["-q:v", "3"])

    # Cleanup
    shutil.rmtree(frames_dir)
    print(f"\nDone! Videos saved to:")
    print(f"  {vp9_path}")
    print(f"  {mp4_path}")


if __name__ == "__main__":
    main()
