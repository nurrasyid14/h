# IndoToxic 2024

Indonesian toxicity analysis using annotated comments, text preprocessing, a multi-head Random Forest model, an SVM toxicity classifier, and transparent lexicon/pattern signals.

## Project Status

Statuses describe current artifacts and remaining work.

| Stage | Status | Current state and remaining work |
| --- | --- | --- |
| EDA 1 | Partial / blocked | `notebooks/01_EDA.ipynb` is empty. `notebooks/data_understanding.ipynb` expects a JSONL file that is absent; the available raw source is CSV. |
| Preprocessing | **Complete; revision needed** | `notebooks/02_preprocessing.ipynb` and `data/processed/data_cleaned_before_encode.csv` exist. Revisit the export/feature decisions and verify the final saved columns and row counts before treating this as a stable training input. |
| EDA 2 | Not started | No completed post-preprocessing EDA report is present. |
| Feature engineering | Partial | The modeling notebook currently fits TF-IDF on its training partition. A reusable, persisted preprocessing/feature pipeline still needs refinement. |
| Modeling | **Complete; needs some refinements** | `notebooks/03_Modeling.ipynb` contains a duplicate-grouped split, TriModel Random Forest heads, and a final `ToxicOrNot` calibrated LinearSVC. Out-of-fold TriModel proto-class columns and exact-match rule features feed the final toxicity model. The notebook's saved execution state is unexecuted, so rerun it to refresh metrics and artifacts. Mood and sentiment remain toxicity-derived proxies. |
| Toxicity decision | Implemented; refine policy | The final SVM consumes text, TriModel proto-classes, and rule hit/count features; its calibrated predicted-class confidence is evaluated with held-out Brier score. Lexicon disagreement remains a review signal, not an automatic override. |
| Gambling-ad detection | Heuristic implemented | X-tag search is optional and requires authorized API access. The keyword-plus-number detector flags candidates, not confirmed ads. `data/togelnumbers.json` records source metadata, but has no verified number meanings yet. |
| Dashboarding | **Complete** | Streamlit multi-page dashboard implemented (`dashboard/app.py`) with live AI inference, EDA, batch CSV moderation, model diagnostics, and regional lexicon audit. |

## Current Flow

The flowchart source is [charts/chart30092026.mermaid.js](charts/chart30092026.mermaid.js).

The processed comments feed the notebook's duplicate-safe train/test split. The TriModel generates mood/sentiment proxy labels and subtopics; out-of-fold proto-class columns are appended with lexicon hit/count features. A calibrated LinearSVC then predicts the annotated toxicity target and confidence. Conflicting rule signals are surfaced for review. Optional X posts can also be scanned for gambling-promotion patterns. The interactive Streamlit dashboard connects directly to these trained models and data.

## Running the Dashboard

To launch the interactive Streamlit dashboard:

```bash
cd h
streamlit run dashboard/app.py
```
or from the project root:
```bash
streamlit run app.py
```

## Data Snapshot

| File | Contents |
| --- | --- |
| `data/raw/indotoxic2024_annotated_data-3.csv` | Annotated source dataset. |
| `data/interim/label_conflict.csv` | Rows collected for label-conflict review. |
| `data/processed/data_cleaned_before_encode.csv` | Current cleaned modeling input with text and topic-list fields. |
| `data/umpatan.json` | Indonesian/regional expressions used for exact-match review signals. Entries may be contextual and are not automatic SVM overrides. |
| `data/togelnumbers.json` | Source registry for possible number-meaning references; meanings are currently empty/unverified. |

`models/` contains serialized artifacts: `mood_model.pkl`, `sentiment_model.pkl`, `subtopic_model.pkl`, `trimodel_model.pkl`, `vectorizer_model.pkl`, `toxic_or_not_model.pkl`, `all_topics.json`, and `evaluation_metrics.json`.

## Next Work

1. Revise preprocessing output and verify the persisted schema.
2. Complete EDA 1 and the dedicated evaluation workflow notebook if required for academic submission.
3. Review lexicon context/provenance and add number meanings only from authorized, verified sources.
4. Expand live X/social media scrapers with verified API tokens.