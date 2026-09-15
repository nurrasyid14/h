"""Reusable preprocessing functions for IndoToxic data."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root from this module location."""
    return Path(__file__).resolve().parents[1]
