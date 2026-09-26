"""Evaluation utilities for classification models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Metrics:
    """Compute standard classification metrics from ground truth and predictions."""

    def __init__(self, y_true: Iterable[int] | np.ndarray, y_pred: Iterable[int] | np.ndarray) -> None:
        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()

        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.precision = precision_score(
            self.y_true,
            self.y_pred,
            zero_division=0,
        )
        self.recall = recall_score(
            self.y_true,
            self.y_pred,
            zero_division=0,
        )
        self.f1 = f1_score(
            self.y_true,
            self.y_pred,
            zero_division=0,
        )

    def gap(self, train_true: Iterable[int] | np.ndarray, train_pred: Iterable[int] | np.ndarray) -> float:
        """Return absolute difference between training and validation accuracy."""
        train_metrics = Metrics(train_true, train_pred)
        return abs(train_metrics.accuracy - self.accuracy)


def metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    train_true: Iterable[int] | np.ndarray | None = None,
    train_pred: Iterable[int] | np.ndarray | None = None,
) -> tuple[float, float, float, float, str, float]:
    """Return accuracy, precision, recall, F1, status, and train/test gap."""
    model_metrics = Metrics(y_true, y_pred)

    if train_true is None or train_pred is None:
        gap = 0.0
        status = "n/a"
    else:
        gap = model_metrics.gap(train_true, train_pred)
        if gap < 0.10:
            status = "good"
        elif gap < 0.25:
            status = "warning"
        else:
            status = "poor"

    return (
        float(model_metrics.accuracy),
        float(model_metrics.precision),
        float(model_metrics.recall),
        float(model_metrics.f1),
        status,
        float(gap),
    )
