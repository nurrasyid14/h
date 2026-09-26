"""Reusable preprocessing functions for IndoToxic data."""

from __future__ import annotations

import re
from pathlib import Path


def project_root() -> Path:
    """Return the repository root from this module location."""
    return Path(__file__).resolve().parents[1]


def clean_text(text: str | None) -> str:
    """Normalize text for downstream modeling and dashboard use."""
    if text is None:
        return ""

    value = str(text).lower()
    value = value.replace("\n", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_text(text: str | None) -> str:
    """Basic Indonesian text normalization with simple token cleanup."""
    value = clean_text(text)
    if not value:
        return ""

    replacements = {
        "gue": "saya",
        "gua": "saya",
        "aku": "saya",
        "saya": "saya",
        "kamu": "anda",
        "elo": "anda",
        "lu": "anda",
        "gak": "tidak",
        "nggak": "tidak",
        "enggak": "tidak",
        "ga": "tidak",
        "bgt": "sangat",
        "banget": "sangat",
        "marah": "marah",
        "benci": "benci",
    }

    tokens = []
    for token in value.split():
        tokens.append(replacements.get(token, token))
    return " ".join(tokens)
