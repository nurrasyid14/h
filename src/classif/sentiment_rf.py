from __future__ import annotations
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class SentimentRF:
    """Random Forest classifier for sentiment prediction."""
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.classes_: np.ndarray | None = None

    def fit(self, X: Any, y: Any) -> "SentimentRF":
        """
        Train the sentiment classifier.

        Parameters:
        -   X:Training features.
        -   y:Sentiment labels.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict sentiment labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Return probability distribution over sentiment classes.
        Columns correspond to ``classes_``.
        """
        return self.model.predict_proba(X)

    def transform(self, X: Any) -> dict[str, Any]:
        """
        Generate sentiment features for downstream models.
        Returns dict Contains the predicted sentiment, class probabilities,and class names.
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        return {
            "sentiment": predictions,
            "sentiment_proba": probabilities,
            "sentiment_classes": self.classes_,
        }

    def get_feature_names(self) -> list[str]:
        """
        Return names of the probability features generated
        by this classifier.
        """
        if self.classes_ is None:
            raise RuntimeError("Model must be fitted first.")

        return [
            f"sentiment_{str(label)}_proba"
            for label in self.classes_
        ]

    def get_params(self) -> dict[str, Any]:
        """Return model parameters."""
        return self.model.get_params()