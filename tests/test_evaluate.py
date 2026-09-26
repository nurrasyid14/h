import numpy as np
import pytest

from src.evaluate import Metrics, metrics


def test_metrics_returns_standard_binary_scores_and_gap():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    train_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    train_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0])

    acc, prec, rec, f1, status, gap = metrics(y_true, y_pred, train_true, train_pred)

    assert acc == pytest.approx(4 / 6)
    assert prec == pytest.approx(2 / 3)
    assert rec == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)
    assert gap == pytest.approx(1 / 3)
    assert status == "poor"

    metrics_obj = Metrics(y_true, y_pred)
    assert metrics_obj.accuracy == pytest.approx(4 / 6)
    assert metrics_obj.precision == pytest.approx(2 / 3)
    assert metrics_obj.recall == pytest.approx(2 / 3)
    assert metrics_obj.f1 == pytest.approx(2 / 3)
    assert metrics_obj.gap(train_true, train_pred) == pytest.approx(1 / 3)
