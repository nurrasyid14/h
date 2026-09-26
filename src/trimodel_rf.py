"""Orchestration utilities for the IndoToxic tri-model classifier stack."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.classif.mood_rf import MoodRF
from src.classif.sentiment_rf import SentimentRF
from src.classif.subtopic_rf import SubtopicRF


class TriModelRF:
    """Coordinate mood, sentiment, and subtopic predictions in one object."""

    def __init__(
        self,
        mood_kwargs: dict[str, Any] | None = None,
        sentiment_kwargs: dict[str, Any] | None = None,
        subtopic_kwargs: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.mood_model = MoodRF(random_state=random_state, **(mood_kwargs or {}))
        self.sentiment_model = SentimentRF(random_state=random_state, **(sentiment_kwargs or {}))
        self.subtopic_model = SubtopicRF(random_state=random_state, **(subtopic_kwargs or {}))
        self.subtopic_labels_: np.ndarray | None = None
        self._is_fitted = False

    def fit(
        self,
        X: Any,
        mood_y: Any,
        sentiment_y: Any,
        subtopic_y: Any,
        labels: Sequence[str] | None = None,
    ) -> "TriModelRF":
        """Fit all three classifiers on the same feature matrix."""
        self.mood_model.fit(X, mood_y)
        self.sentiment_model.fit(X, sentiment_y)
        self.subtopic_model.fit(X, subtopic_y, labels=labels)
        self.subtopic_labels_ = self.subtopic_model.labels_
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> dict[str, Any]:
        """Return a stacked prediction payload for each model."""
        if not self._is_fitted:
            raise RuntimeError("TriModelRF must be fitted before calling predict().")

        return {
            "mood": self.mood_model.predict(X),
            "sentiment": self.sentiment_model.predict(X),
            "subtopic": self.subtopic_model.predict(X),
            "subtopic_labels": self.subtopic_labels_,
        }

    def transform(self, X: Any) -> dict[str, Any]:
        """Return predictions and probabilities for all model heads."""
        if not self._is_fitted:
            raise RuntimeError("TriModelRF must be fitted before calling transform().")

        return {
            "mood": self.mood_model.predict(X),
            "mood_proba": self.mood_model.predict_proba(X),
            "sentiment": self.sentiment_model.predict(X),
            "sentiment_proba": self.sentiment_model.predict_proba(X),
            "subtopic": self.subtopic_model.predict(X),
            "subtopic_proba": self.subtopic_model.predict_proba(X),
            "subtopic_labels": self.subtopic_labels_,
        }


__all__ = ["TriModelRF"]
