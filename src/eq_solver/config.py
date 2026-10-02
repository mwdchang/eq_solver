"""Load and validate the shared TOML configuration."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARAM_NAMES = (
    "infection_rate",
    "reinfection_factor",
    "recovery_rate",
    "death_rate",
    "movement_rate",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ParamSpec:
    # Shared parameters have shape (), per-cell parameters have shape (rows, cols)
    value: np.ndarray
    guess: np.ndarray
    bounds: tuple[float, float]

    @property
    def per_cell(self) -> bool:
        return self.value.ndim == 2


@dataclass(frozen=True)
class Config:
    rows: int
    cols: int
    initial_state: np.ndarray  # (4, rows, cols): S, I, R, D
    params: dict[str, ParamSpec]

    def values(self) -> dict[str, np.ndarray]:
        return {name: spec.value for name, spec in self.params.items()}


def _as_scalar_or_grid(raw, rows: int, cols: int, what: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0 or arr.shape == (rows, cols):
        return arr
    raise ConfigError(
        f"{what}: expected a single value or a {rows}x{cols} grid "
        f"([[row0], [row1], ...]), got shape {arr.shape}"
    )


def _parse_param(name: str, raw: dict, rows: int, cols: int) -> ParamSpec:
    if "value" not in raw:
        raise ConfigError(f"params.{name}: missing 'value'")
    value = _as_scalar_or_grid(raw["value"], rows, cols, f"params.{name}.value")

    low, high = (float(b) for b in raw.get("bounds", (0.0, 1.0)))
    if not low < high:
        raise ConfigError(f"params.{name}.bounds: low must be below high, got {[low, high]}")

    guess = _as_scalar_or_grid(raw.get("guess", value), rows, cols, f"params.{name}.guess")
    if value.ndim == 0 and guess.ndim == 2:
        raise ConfigError(f"params.{name}: a grid guess needs a per-cell (grid) value")
    guess = np.broadcast_to(guess, value.shape).copy()
    if np.any(guess <= low) or np.any(guess >= high):
        raise ConfigError(f"params.{name}.guess must lie strictly inside bounds {[low, high]}")

    return ParamSpec(value=value, guess=guess, bounds=(low, high))


def _initial_state(raw: dict, rows: int, cols: int) -> np.ndarray:
    population = _as_scalar_or_grid(raw.get("population", 1000), rows, cols, "initial.population")
    infected = np.zeros((rows, cols))

    seeds = raw.get("infected", [])
    if isinstance(seeds, dict):
        seeds = [seeds]
    for seed in seeds:
        r, c = seed["cell"]
        if not (0 <= r < rows and 0 <= c < cols):
            raise ConfigError(f"initial.infected: cell {[r, c]} is outside the {rows}x{cols} grid")
        infected[r, c] += float(seed["count"])

    susceptible = np.broadcast_to(population, (rows, cols)) - infected
    if np.any(susceptible < 0):
        raise ConfigError("initial.infected: more infected than population in a cell")

    zeros = np.zeros((rows, cols))
    return np.stack([susceptible, infected, zeros, zeros])


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    grid = raw.get("grid", {})
    rows, cols = int(grid.get("rows", 2)), int(grid.get("cols", 2))

    raw_params = raw.get("params", {})
    unknown = set(raw_params) - set(PARAM_NAMES)
    if unknown:
        raise ConfigError(f"unknown parameters: {sorted(unknown)}; expected {list(PARAM_NAMES)}")
    missing = set(PARAM_NAMES) - set(raw_params)
    if missing:
        raise ConfigError(f"missing parameters: {sorted(missing)}")

    params = {name: _parse_param(name, raw_params[name], rows, cols) for name in PARAM_NAMES}
    return Config(
        rows=rows,
        cols=cols,
        initial_state=_initial_state(raw.get("initial", {}), rows, cols),
        params=params,
    )
