"""Prediction entry point for new text."""

from __future__ import annotations

import numpy as np

from src.classif.toxicity_meter import ToxicityMeter
from src.preprocessing import normalize_text
from src.trimodel_rf import TriModelRF


def _default_training_data():
    """Create a small deterministic default dataset for inference fallback."""
    X = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=float,
    )
    mood = ["happy", "angry", "sad", "neutral", "happy", "angry"]
    sentiment = ["positive", "negative", "negative", "neutral", "positive", "negative"]
    subtopic = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
    return X, mood, sentiment, subtopic


def _text_to_feature_vector(text: str) -> np.ndarray:
    """Convert text to a compact numeric feature vector for fallback inference."""
    normalized = normalize_text(text)
    tokens = set(normalized.split())

    features = {
        "happy": 0.0,
        "angry": 0.0,
        "sad": 0.0,
        "neutral": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "benci": 0.0,
        "marah": 0.0,
        "suka": 0.0,
        "bahagia": 0.0,
    }

    for token in tokens:
        if token in {"happy", "bahagia", "senang", "suka"}:
            features["happy"] += 1.0
            features["positive"] += 1.0
        if token in {"angry", "marah", "benci", "menghina"}:
            features["angry"] += 1.0
            features["negative"] += 1.0
            features["benci"] += 1.0
            features["marah"] += 1.0
        if token in {"sedih", "sedih", "duka", "kecewa"}:
            features["sad"] += 1.0
        if token in {"baik", "ramah", "tenang", "damai"}:
            features["neutral"] += 1.0

    return np.asarray(
        [
            features["happy"],
            features["angry"],
            features["sad"],
            features["neutral"],
        ],
        dtype=float,
    ).reshape(1, -1)


def predict_single_text(text: str, model: TriModelRF | None = None) -> dict[str, object]:
    """Predict mood, sentiment, subtopic, and toxicity for a single text input."""
    if not text or not str(text).strip():
        return {
            "mood": "neutral",
            "sentiment": "neutral",
            "subtopic": np.array([0, 0], dtype=int),
            "subtopic_labels": np.array(["topic_a", "topic_b"]),
            "toxicity": 0.0,
        }

    if model is None:
        model = TriModelRF(random_state=42)
        X, mood_y, sentiment_y, subtopic_y = _default_training_data()
        model.fit(X, mood_y, sentiment_y, subtopic_y, labels=["topic_a", "topic_b"])

    feature = _text_to_feature_vector(text)
    prediction = model.predict(feature)
    toxicity = ToxicityMeter().score(text)

    return {
        "mood": prediction["mood"][0],
        "sentiment": prediction["sentiment"][0],
        "subtopic": prediction["subtopic"][0],
        "subtopic_labels": prediction["subtopic_labels"],
        "toxicity": toxicity,
    }


__all__ = ["predict_single_text"]
