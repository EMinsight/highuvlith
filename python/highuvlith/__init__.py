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
    # MNSL classes
    PyMnslConfig as MnslConfig,
    PyMnslEngine as MnslEngine,
    PyMnslResult as MnslResult,
    PyNanosphereArrayConfig as NanosphereArrayConfig,
    PySpherePacking as SpherePacking,
    PySubstrateCoupling as SubstrateCoupling,
    py_simulate_moire_emission,
)

from highuvlith.api import (
    FullResult,
    simulate_line_space,
    simulate_contact_hole,
    sweep_focus,
)

from highuvlith.mnsl import (
    MnslSimResult,
    simulate_moire_emission,
    create_nanosphere_array,
    sweep_rotation_angle,
    sweep_separation,
    optimize_moire_parameters,
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
    # MNSL Configuration
    "MnslConfig",
    "MnslEngine",
    "MnslResult",
    "NanosphereArrayConfig",
    "SpherePacking",
    "SubstrateCoupling",
    "py_simulate_moire_emission",
    # MNSL High-level API
    "MnslSimResult",
    "simulate_moire_emission",
    "create_nanosphere_array",
    "sweep_rotation_angle",
    "sweep_separation",
    "optimize_moire_parameters",
]
