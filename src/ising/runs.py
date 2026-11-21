"""Helpers shared by the scripts: temperature grids and saved configurations.

Exercise 5.11 asks each run to start from a configuration saved by an
earlier run at the same temperature.  The file name must encode T exactly,
so temperatures are built from integers and always written with the same
format.  (Recomputing T as 1 / beta can change the last digit and silently
miss the saved file.)
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "data" / "configs"
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"


def temperature_grid(start, stop, step):
    """Temperatures start, start + step, ..., up to stop, rounded to 6 decimals."""
    n = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(n), 6)


def temperature_label(T):
    """The one format used for temperatures in file names."""
    return f"{T:.4f}"


def config_path(algorithm, L, T):
    return CONFIG_DIR / f"{algorithm}_L{L}_T{temperature_label(T)}.npy"


def load_config(algorithm, L, T):
    """The configuration saved by the last run at this (L, T), or None."""
    path = config_path(algorithm, L, T)
    return np.load(path) if path.exists() else None


def save_config(algorithm, L, T, spins):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config_path(algorithm, L, T), spins.astype(np.int8))