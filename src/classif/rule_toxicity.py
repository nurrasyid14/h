"""Rule-based toxicity classification using the project profanity lexicon."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.preprocessing import clean_text


class RuleToxicityClassifier:
    """Flag text containing an exact lexicon word or phrase."""

    def __init__(self, lexicon_path: str | Path | None = None) -> None:
        if lexicon_path is None:
            lexicon_path = Path(__file__).resolve().parents[2] / "data" / "umpatan.json"

        with Path(lexicon_path).open(encoding="utf-8") as handle:
            entries = json.load(handle)["entries"]

        grouped_entries: dict[str, set[str]] = {}
        for entry in entries:
            expression = clean_text(entry.get("expression"))
            if expression:
                grouped_entries.setdefault(expression, set()).add(entry.get("region", ""))

        self._patterns = [
            (
                expression,
                sorted(region for region in regions if region),
                re.compile(
                    r"(?<![a-z0-9])"
                    + r"\s+".join(re.escape(token) for token in expression.split())
                    + r"(?![a-z0-9])"
                ),
            )
            for expression, regions in grouped_entries.items()
        ]

    def classify(self, text: str | None) -> dict[str, Any]:
        """Return the binary rule decision and all matching lexicon entries."""
        normalized = clean_text(text)
        matches = [
            {"expression": expression, "regions": regions}
            for expression, regions, pattern in self._patterns
            if pattern.search(normalized)
        ]
        toxic = bool(matches)
        return {
            "toxic": toxic,
            "label": "toxic" if toxic else "non_toxic",
            "matches": matches,
        }


__all__ = ["RuleToxicityClassifier"]