"""Training entry point for IndoToxic classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.classif.mood_rf import MoodRF
from src.classif.sentiment_rf import SentimentRF
from src.classif.subtopic_rf import SubtopicRF
from src.utils import ensure_directory, save_pickle


def train_models(
    X: Any,
    mood_y: Any,
    sentiment_y: Any,
    subtopic_y: Any,
    labels: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the core tri-model pipeline and optionally save the fitted artifacts."""
    mood_model = MoodRF(random_state=random_state)
    sentiment_model = SentimentRF(random_state=random_state)
    subtopic_model = SubtopicRF(random_state=random_state)

    mood_model.fit(X, mood_y)
    sentiment_model.fit(X, sentiment_y)
    subtopic_model.fit(X, subtopic_y, labels=labels)

    payload = {
        "mood": mood_model,
        "sentiment": sentiment_model,
        "subtopic": subtopic_model,
    }

    if output_dir is not None:
        directory = ensure_directory(output_dir)
        for name, model in payload.items():
            save_pickle(model, directory / f"{name}_model.pkl")

    return payload


__all__ = ["train_models"]
