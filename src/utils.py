"""Shared project utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Persist an object with pickle."""
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("wb") as handle:
        pickle.dump(obj, handle)
    return target


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object from disk."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float while keeping a default fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def splitter(*arrays: Any, test_size: float = 0.2, random_state: int = 42, shuffle: bool = True):
    """Split arrays into an 80:20 train/test partition."""
    if not arrays:
        raise ValueError("At least one array must be provided for splitting.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    return train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )


splitter = splitter


__all__ = [
    "project_root",
    "ensure_directory",
    "save_pickle",
    "load_pickle",
    "safe_float",
]
