# Source package guide

This package contains the project’s reusable Python code for preprocessing, model training, evaluation, and inference. At the moment, the repository is in a transitional state: some modules are already implemented as concrete classifier wrappers, while others remain stubs that describe the intended production architecture.

## 1. Project layer and module precedents

The central pattern in this project is a tri-model pipeline:

- Mood classifier
- Sentiment classifier
- Subtopic classifier

The actual concrete precedent is in `src/classif/`:

- `src/classif/mood_rf.py`
- `src/classif/sentiment_rf.py`
- `src/classif/subtopic_rf.py`

These classes follow the same pattern:

- `fit(X, y)` trains the RandomForest model
- `predict(X)` returns class predictions
- `predict_proba(X)` returns class probabilities
- `transform(X)` emits model outputs in dictionary format for downstream use
- `get_feature_names()` creates feature names for probability columns
- `get_params()` exposes the estimator configuration

Among them, `SubtopicRF` is the most complete example because it supports multilabel output through `MultiOutputClassifier` and handles labels explicitly.

This is the reusable blueprint for the orchestration layer that should later sit in `src/trimodel_rf.py` and the training entry point in `src/train.py`.

## 2. Current source modules

### `src/evaluate.py`

This is the active evaluation module.

It defines:

- `Metrics`: a classifier metrics wrapper
- `metrics(...)`: convenience function that returns accuracy, precision, recall, F1, a status string, and a train/test gap

The implementation is intentionally lightweight and compatible with the project’s existing binary-classification use case.

Typical usage:

```python
from src.evaluate import metrics

acc, prec, rec, f1, status, gap = metrics(y_true, y_pred, train_true, train_pred)
```

This is the reference evaluation layer the project should use for model diagnostics and validation reports.

### `src/preprocessing.py`

This file is currently only a small helper module that exposes the repository root via `project_root()`.

It does not yet contain the full preprocessing pipeline. The actual preprocessing logic is still concentrated in the notebooks, especially:

- `notebooks/02_preprocessing.ipynb`
- `notebooks/data_understanding.ipynb`

The preprocessing notebook handles tasks such as:

- empty-text and spam removal
- duplicate handling
- label-conflict cleanup
- emoji extraction
- text cleaning
- slang normalization
- topic preparation and encoding

The source package still needs a real production-quality preprocessing pipeline to move this logic out of notebooks.

### `src/train.py`

This file is a stub. It is intended to be the training entry point for the project’s classifier stack, but no actual model-fitting workflow is implemented yet.

Conceptually, it should:

1. load processed features
2. split train/validation data
3. instantiate mood / sentiment / subtopic models
4. fit the models
5. persist trained artifacts under `models/`
6. record metrics and training diagnostics

At present, the project has no serialized model outputs under `models/`, so this file is not yet wired to a runnable end-to-end training job.

### `src/predict.py`

This file is also a stub and represents the inference entry point for new text.

The intended behavior is:

- accept raw text input
- apply the same preprocessing pipeline as training
- run the mood / sentiment / subtopic models
- return structured predictions and probabilities

This is the module that should eventually call the saved trained models and expose a clean API for the dashboard or an external service. It is not yet connected to actual model artifacts.

### `src/trimodel_rf.py`

This module is the intended orchestration layer for the three Random Forest classifiers.

The intended pattern is:

1. prepare inputs
2. call `MoodRF`, `SentimentRF`, and `SubtopicRF`
3. combine their outputs into a unified prediction payload
4. optionally expose per-model probabilities and labels
5. feed the combined result into later filtering or dashboard logic

The orchestrator is not implemented yet, and the classif modules are the real precedent for how it should behave.

### `src/utils.py`

This is a shared utilities placeholder.

It should eventually contain common functionality such as:

- project root resolution
- repeated preprocessing helpers
- validation helpers
- data schema checks
- saving and loading model objects
- serialization of predictions and metrics

Right now it is only a shell and does not yet contain shared logic used by the rest of the project.

## 3. Classifier implementations in `src/classif/`

These are the modules with the clearest implementation status.

### `src/classif/mood_rf.py`

A RandomForest classifier wrapper for mood prediction.

### `src/classif/sentiment_rf.py`

A RandomForest classifier wrapper for sentiment prediction.

### `src/classif/subtopic_rf.py`

A multilabel RandomForest wrapper using `MultiOutputClassifier`.

This is the most important precedent for the source layer because it correctly handles:

- 2D label matrices
- explicit label naming
- per-output probability extraction
- transform output for downstream use

### `src/classif/toxicity_meter.py`

This file is empty. It is intended to represent the toxicity assessment layer, but it is not implemented yet.

The project roadmap clearly expects this module to provide toxicity scoring or toxicity classification logic using the project’s toxicity labels.

## 4. Missing pieces in the source layer

The current source package is incomplete in the following ways:

1. `train.py` has no actual training workflow
2. `predict.py` has no inference implementation
3. `trimodel_rf.py` is not a working orchestration layer
4. `utils.py` is not yet populated with reusable helpers
5. `toxicity_meter.py` is empty
6. data preprocessing is still notebook-driven instead of source-driven
7. model persistence under `models/` is missing

In short, the source package contains the intended architecture, but not yet the full end-to-end implementation.

## 5. Recommended direction

The healthy progression is:

1. Move the preprocessing logic into a reusable source module
2. Complete the tri-model orchestration in `trimodel_rf.py`
3. Build the real training pipeline in `train.py`
4. Implement a stable prediction API in `predict.py`
5. Add toxicity classification and meter logic in `src/classif/toxicity_meter.py`
6. Keep evaluation centralized in `src/evaluate.py`
7. Persist metrics and model artifacts so the dashboard and reports can consume them consistently

## 6. Source-layer summary

The project’s strongest implementation precedent is not the notebooks; it is the classifier wrapper pattern in `src/classif/`.

The next step is to promote that pattern into a real source pipeline:

- reusable preprocessing functions
- tri-model orchestration
- training entry point
- inference entry point
- standardized evaluation

That is the missing bridge between the notebook experiments and a production-ready IndoToxic modeling workflow.
