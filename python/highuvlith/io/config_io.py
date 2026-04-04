"""Configuration serialization (TOML)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to a TOML file."""

    path = Path(path)

    def _to_toml_str(d: dict, indent: int = 0) -> str:
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}[{key}]")
                lines.append(_to_toml_str(value, indent + 1))
            elif isinstance(value, list):
                formatted = ", ".join(repr(v) for v in value)
                lines.append(f"{prefix}{key} = [{formatted}]")
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key} = {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"{prefix}{key} = {value}")
            elif isinstance(value, str):
                lines.append(f'{prefix}{key} = "{value}"')
        return "\n".join(lines)

    path.write_text(_to_toml_str(config) + "\n")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a configuration dictionary from a TOML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)
