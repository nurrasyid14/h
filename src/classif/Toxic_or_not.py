"""Binary toxicity classification with a linear SVM and lexicon rules."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.classif.rule_toxicity import RuleToxicityClassifier
from src.preprocessing import clean_text


class ToxicOrNot:
    """Predict toxicity with an annotated-data SVM and flag lexicon review cases."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict[int, float] | None = "balanced",
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        random_state: int = 42,
        lexicon_path: str | None = None,
        calibration_cv: int = 3,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        self.svm = CalibratedClassifierCV(
            estimator=LinearSVC(C=C, class_weight=class_weight, random_state=random_state),
            method="sigmoid",
            cv=calibration_cv,
        )
        self.rule_classifier = RuleToxicityClassifier(lexicon_path=lexicon_path)
        self._engineered_feature_count = 0
        self._is_fitted = False

    def _features(
        self,
        texts: list[str],
        engineered_features: Any | None,
        fit_vectorizer: bool,
    ) -> Any:
        normalized_texts = [clean_text(text) for text in texts]
        if fit_vectorizer:
            text_features = self.vectorizer.fit_transform(normalized_texts)
        else:
            text_features = self.vectorizer.transform(normalized_texts)

        if engineered_features is None:
            if self._engineered_feature_count:
                raise ValueError("engineered_features are required because the model was fitted with them.")
            return text_features

        engineered = csr_matrix(engineered_features, dtype=float)
        if engineered.shape[0] != len(texts):
            raise ValueError("engineered_features must have one row per text.")
        if fit_vectorizer:
            self._engineered_feature_count = engineered.shape[1]
        elif engineered.shape[1] != self._engineered_feature_count:
            raise ValueError("engineered_features must have the same columns used during fit.")
        return hstack((text_features, engineered), format="csr")

    def fit(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        engineered_features: Any | None = None,
    ) -> "ToxicOrNot":
        """Fit on text, optional engineered features, and binary toxicity labels."""
        normalized_texts = [clean_text(text) for text in texts]
        target = np.asarray(list(labels), dtype=int)
        if len(normalized_texts) != len(target):
            raise ValueError("texts and labels must have the same number of rows.")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("toxicity labels must be binary: 0 or 1.")
        if np.unique(target).size != 2:
            raise ValueError("training data must contain both toxicity classes.")

        features = self._features(normalized_texts, engineered_features, fit_vectorizer=True)
        self.svm.fit(features, target)
        self._is_fitted = True
        return self

    def predict(
        self,
        texts: Iterable[str],
        engineered_features: Any | None = None,
    ) -> dict[str, Any]:
        """Return SVM class, calibrated confidence, rule signals, and OR experiment."""
        if not self._is_fitted:
            raise RuntimeError("ToxicOrNot must be fitted before prediction.")

        text_list = list(texts)
        features = self._features(text_list, engineered_features, fit_vectorizer=False)
        svm_prediction = self.svm.predict(features).astype(int)
        class_probabilities = self.svm.predict_proba(features)
        confidence = class_probabilities[np.arange(len(svm_prediction)), svm_prediction]
        rule_results = [self.rule_classifier.classify(text) for text in text_list]
        rule_prediction = np.asarray([result["toxic"] for result in rule_results], dtype=int)

        return {
            "svm_prediction": svm_prediction,
            "rule_prediction": rule_prediction,
            "final_prediction": svm_prediction.copy(),
            "rule_review": ((rule_prediction == 1) & (svm_prediction == 0)).astype(int),
            "or_experimental_prediction": np.maximum(svm_prediction, rule_prediction),
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "rule_matches": [result["matches"] for result in rule_results],
        }


__all__ = ["ToxicOrNot"]