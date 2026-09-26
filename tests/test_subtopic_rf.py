import numpy as np

from src.classif.subtopic_rf import SubtopicRF


def test_subtopic_rf_multilabel_probability_interface():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=int,
    )
    labels = ["terpolarisasi", "tionghoa", "jewish"]

    model = SubtopicRF(random_state=42)
    model.fit(X, y, labels=labels)

    assert model.predict(X).shape == (6, 3)
    assert model.predict_proba(X).shape == (6, 3)
    assert np.all((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1))

    transformed = model.transform(X)
    assert transformed["subtopic"].shape == (6, 3)
    assert transformed["subtopic_proba"].shape == (6, 3)
    assert list(transformed["subtopic_labels"]) == labels

    feature_names = model.get_feature_names()
    assert feature_names == [
        "subtopic_terpolarisasi_proba",
        "subtopic_tionghoa_proba",
        "subtopic_jewish_proba",
    ]
