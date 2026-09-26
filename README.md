# IndoToxic 2024

Indonesian toxicity analysis project. This repository currently contains a raw annotated dataset, a preprocessing notebook and intermediate CSV outputs, three Random Forest classifier wrappers, and a dashboard scaffold.

## Roadmap Status

Marks describe the state of the artifacts in this repository, not just whether a filename exists.

| # | Stage | Status | Evidence and remaining work |
| --- | --- | --- | --- |
| 0 | EDA 1 | ⚠️ Partial / blocked | `notebooks/01_EDA.ipynb` is empty. `notebooks/data_understanding.ipynb` contains exploratory checks for schema, duplicates, toxicity, topics, category counts, and null values, but it reads `data/raw/indotoxic2024_annotated_data-3.jsonl`; that file is absent. The available raw dataset is CSV. |
| 1 | Prep 1 | ⚠️ Draft implemented | `notebooks/02_preprocessing.ipynb` contains spam/empty-text removal, duplicate and label-conflict handling, emoji extraction, text cleaning, slang normalization, topic-list preparation, and optional stopword removal, stemming, and topic encoding. Its saved execution state is unexecuted. A processed CSV exists, but the notebook saves `df` rather than `df_processed`, so the exported file does not contain `text_processed`. |
| 2 | EDA 2 | ⬜ Not started | No post-preprocessing EDA notebook or report is present. |
| 3 | Feature Engineerings | ⚠️ Partial | Preprocessing code creates `emoji`, `text_clean`, and `topic_list`; it defines stopword/stemming and multi-label encoding helpers. The final processed features are not persisted in the current processed CSV, and there is no separate feature-engineering module/pipeline. |
| 4 | Trimodel (classif/rf files) + orchestration | ⚠️ Partial | `src/classif/sentiment_rf.py`, `mood_rf.py`, and `subtopic_rf.py` provide classifier wrappers. `src/trimodel_rf.py` and `src/train.py` are stubs, and `models/` contains no trained model artifacts. |
| 5 | Toxicity classif and meters | ⬜ Not started | `src/classif/toxicity_meter.py` is empty. No toxicity classifier or toxicity-meter implementation/model is present. Toxicity labels are present in the raw and interim data. |
| 6 | Filtering | ⚠️ Partial | Prep removes spam, blank text, duplicate text, and conflicting toxicity labels. A separate downstream filtering/refinement stage is not implemented. |
| 7 | Dashboard building | ⚠️ Scaffold | Four dashboard templates are present, but `dashboard/app.py`, `routes.py`, and the inference/chart services contain only module docstrings; the dashboard is not wired to trained models. |

## Notebooks

| Notebook | Current contents | State / issue |
| --- | --- | --- |
| `notebooks/01_EDA.ipynb` | No cells | Empty. |
| `notebooks/data_understanding.ipynb` | Raw-data loading and exploratory checks | Expects a missing JSONL input; use or generate the available raw CSV instead. |
| `notebooks/02_preprocessing.ipynb` | Cleaning and preprocessing workflow | Cells are not marked as executed. The output cell writes `df` with the default CSV index, rather than writing `df_processed` with `index=False`. |
| `notebooks/03_Modeling.ipynb` | No cells | Empty. |
| `notebooks/04_Evaluation.ipynb` | No cells | Empty. |

## Data Inventory

| Folder / file | Current contents | Observed size |
| --- | --- | --- |
| `data/raw/indotoxic2024_annotated_data-3.csv` | Annotated source CSV; 17 columns | 43,692 rows; about 20.3 MB |
| `data/interim/label_conflict.csv` | Rows associated with conflicting toxicity labels; same 17-column schema | 8,955 rows; about 5.8 MB |
| `data/processed/data_cleaned_before_encode.csv` | Cleaned text and topic-list fields are present; no encoded topic columns or `text_processed` | 23,009 rows; about 16.9 MB; includes an unnamed index column |

These are the only data files currently present under `data/`. The counts above are a snapshot of the checked-in workspace; rerunning preprocessing may change them. The `label_conflict.csv` row count is a row count, not necessarily a count of unique conflicting texts.

## Current Next Steps

1. Point the EDA notebook at the existing raw CSV and complete `01_EDA.ipynb`.
2. Rerun preprocessing, save the intended final dataframe (`df_processed`) without its index, and verify the resulting columns and row counts.
3. Add EDA 2 and persist/validate the feature-engineering output before training.
4. Implement and test trimodel orchestration, then toxicity classification/meters and downstream filtering.
5. Connect dashboard routes and services to saved models and verify the end-to-end workflow.