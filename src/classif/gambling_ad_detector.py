"""Detect candidate gambling promotions from keyword-and-number patterns."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.preprocessing import clean_text


DEFAULT_PREFIX_TERMS = (
    "menang",
    "suka",
    "juara",
    "singapore",
    "sidney",
    "sydney",
)


class GamblingAdDetector:
    """Flag posts matching configured terms immediately followed by 2–3 digits."""

    def __init__(
        self,
        prefix_terms: Iterable[str] = DEFAULT_PREFIX_TERMS,
        number_reference_path: str | Path | None = None,
    ) -> None:
        normalized_terms = sorted(
            {clean_text(term) for term in prefix_terms if clean_text(term)},
            key=len,
            reverse=True,
        )
        if not normalized_terms:
            raise ValueError("At least one prefix term is required.")

        alternatives = "|".join(re.escape(term) for term in normalized_terms)
        self._pattern = re.compile(
            rf"(?<![a-z0-9])(?P<term>{alternatives})\s*(?P<number>\d{{2,3}})(?!\d)"
        )

        if number_reference_path is None:
            number_reference_path = Path(__file__).resolve().parents[2] / "data" / "togelnumbers.json"
        self._number_meanings: dict[str, list[dict[str, Any]]] = {}
        reference_path = Path(number_reference_path)
        if reference_path.exists():
            with reference_path.open(encoding="utf-8") as handle:
                entries = json.load(handle).get("entries", [])
            for entry in entries:
                number = str(entry.get("number", ""))
                meaning = entry.get("meaning")
                if (
                    re.fullmatch(r"\d{2,3}", number)
                    and meaning
                    and entry.get("review_status") == "verified"
                ):
                    self._number_meanings.setdefault(number, []).append(
                        {
                            "meaning": meaning,
                            "meaning_status": entry.get("meaning_status", "unknown"),
                            "source_ids": entry.get("source_ids", []),
                        }
                    )

    def classify(self, text: str | None) -> dict[str, Any]:
        """Return candidate matches; a match is a signal, not proof of gambling."""
        normalized = clean_text(text)
        matches = [
            {
                "term": match.group("term"),
                "number": match.group("number"),
                "meanings": self._number_meanings.get(match.group("number"), []),
            }
            for match in self._pattern.finditer(normalized)
        ]
        flagged = bool(matches)
        return {
            "suspected_gambling_promo": flagged,
            "label": "suspected_gambling_promo" if flagged else "no_pattern",
            "matches": matches,
        }


__all__ = ["DEFAULT_PREFIX_TERMS", "GamblingAdDetector"]