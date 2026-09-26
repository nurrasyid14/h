"""Utility for assigning a simple toxicity score from text content."""

from __future__ import annotations

from src.preprocessing import normalize_text


class ToxicityMeter:
    """Heuristic toxicity meter for Indonesian text."""

    _OFFENSIVE_TERMS = {
        "benci",
        "marah",
        "menghina",
        "sakit",
        "hancurkan",
        "serang",
        "bunuh",
        "mencela",
        "menyerang",
        "ancam",
        "bajingan",
        "brengsek",
        "kontol",
        "jawir",
    }

    def score(self, text: str | None) -> float:
        """Return a toxicity score in [0, 1]."""
        normalized = normalize_text(text)
        if not normalized:
            return 0.0

        tokens = normalized.split()
        total_tokens = max(1, len(tokens))
        hits = sum(1 for token in tokens if token in self._OFFENSIVE_TERMS)
        score = hits / total_tokens

        severe_hits = sum(1 for token in tokens if token in {"hancurkan", "serang", "bunuh", "ancam"})
        if severe_hits:
            score = min(1.0, score + 0.25)
        return round(min(1.0, score), 4)

    def classify(self, text: str | None) -> str:
        """Return a lexical toxicity label."""
        return "toxic" if self.score(text) >= 0.2 else "non_toxic"


__all__ = ["ToxicityMeter"]
