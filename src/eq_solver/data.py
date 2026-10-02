"""Synthetic datasets and CSV read/write. CSV columns: day,row,col,S,I,R,D."""

import csv
from pathlib import Path

import numpy as np

from eq_solver.model import COMPARTMENTS

HEADER = ("day", "row", "col", *COMPARTMENTS)


def add_noise(states: np.ndarray, noise: float, seed: int = 0) -> np.ndarray:
    """Relative Gaussian noise, clipped at zero. The initial day is kept exact."""
    rng = np.random.default_rng(seed)
    noisy = states * (1.0 + noise * rng.standard_normal(states.shape))
    noisy[0] = states[0]
    return np.clip(noisy, 0.0, None)


def write_csv(path: str | Path, states: np.ndarray) -> None:
    """states: [days, 4, rows, cols]"""
    days, _, rows, cols = states.shape
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for t in range(days):
            for r in range(rows):
                for c in range(cols):
                    writer.writerow([t, r, c, *(f"{v:.6f}" for v in states[t, :, r, c])])


def read_csv(path: str | Path, rows: int, cols: int) -> np.ndarray:
    """Returns [days, 4, rows, cols]. Every day from 0 to the last must be present for every cell."""
    with open(path, newline="") as f:
        records = list(csv.DictReader(f))
    if not records or tuple(records[0]) != HEADER:
        raise ValueError(f"{path}: expected header {','.join(HEADER)}")

    days = max(int(rec["day"]) for rec in records) + 1
    states = np.full((days, len(COMPARTMENTS), rows, cols), np.nan)
    for rec in records:
        t, r, c = int(rec["day"]), int(rec["row"]), int(rec["col"])
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{path}: cell ({r}, {c}) is outside the {rows}x{cols} grid")
        states[t, :, r, c] = [float(rec[k]) for k in COMPARTMENTS]

    if np.isnan(states).any():
        raise ValueError(f"{path}: missing rows; need every day 0..{days - 1} for every cell")
    return states
