"""Internal validation helpers."""


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_range(name: str, value: float, lo: float, hi: float) -> None:
    if value < lo or value > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def _validate_power_of_two(name: str, value: int) -> None:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of 2, got {value}")
