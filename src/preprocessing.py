"""Reusable preprocessing functions for IndoToxic data."""

from __future__ import annotations

import re
import json
from pathlib import Path
from collections.abc import Iterable
from functools import lru_cache


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
    """Basic Indonesian text normalization using ``data/slang_dict.json``."""
    value = clean_text(text)
    if not value:
        return ""

    return " ".join(load_slang_lookup().get(token, token) for token in value.split())


@lru_cache(maxsize=1)
def load_slang_lookup(path: Path | None = None) -> dict[str, str]:
    """Load slang-to-standard-word mappings from the project JSON file."""
    path = path or project_root() / "data" / "slang_dict.json"
    with path.open(encoding="utf-8") as source:
        roots = json.load(source)
    return {
        slang.casefold(): root.casefold()
        for root, slangs in roots.items()
        for slang in slangs
    }


def load_json_expressions(path: Path, key: str = "entries") -> set[str]:
    """Load and normalize string expressions from one of the lexicon JSON files."""
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    values: Iterable[object] = data.get(key, [])
    if key == "entries":
        values = (entry.get("expression", "") for entry in values if isinstance(entry, dict))
    return {
        re.sub(r"\s+", " ", str(value).casefold()).strip()
        for value in values
        if str(value).strip()
    }


def contains_expression(text: str | None, expressions: Iterable[str]) -> bool:
    """Match normalized whole-word phrases without treating substrings as matches."""
    if text is None:
        return False
    normalized = re.sub(r"[^\w\s]", " ", str(text).casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    pattern = "|".join(
        r"\s+".join(re.escape(token) for token in phrase.split())
        for phrase in sorted(expressions, key=len, reverse=True)
    )
    return bool(pattern and re.search(rf"(?<!\w)(?:{pattern})(?!\w)", normalized))
