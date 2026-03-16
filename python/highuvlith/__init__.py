"""highuvlith: VUV laser lithography simulation framework."""

from highuvlith._native import (
    SourceConfig,
    OpticsConfig,
    MaskConfig,
    ResistConfig,
    FilmStackConfig,
    ProcessConfig,
    GridConfig,
    SimulationEngine,
    AerialImageResult,
    ResistProfileResult,
    BatchSimulator,
    ProcessWindowResult,
)

from highuvlith.api import (
    FullResult,
    simulate_line_space,
    simulate_contact_hole,
    sweep_focus,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "SourceConfig",
    "OpticsConfig",
    "MaskConfig",
    "ResistConfig",
    "FilmStackConfig",
    "ProcessConfig",
    "GridConfig",
    # Simulation
    "SimulationEngine",
    "BatchSimulator",
    # Results
    "AerialImageResult",
    "ResistProfileResult",
    "ProcessWindowResult",
    # High-level API
    "FullResult",
    "simulate_line_space",
    "simulate_contact_hole",
    "sweep_focus",
]
