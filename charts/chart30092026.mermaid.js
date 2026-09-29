"use strict";

const chart = String.raw`flowchart TD
    RAW["Annotated raw CSV"] --> PREP["Preprocessing<br/>Complete; revision needed"]
    PREP --> CLEAN["Cleaned processed CSV"]

    CLEAN --> SPLIT["Modeling notebook<br/>Stratified grouped train/test split"]
    SPLIT --> TFIDF["Training-only TF-IDF features"]
    TFIDF --> TRI["TriModel Random Forest<br/>Mood/sentiment proxies + subtopics"]
    TRI --> PROTO["Out-of-fold proto-class columns<br/>for toxicity-model training"]
    SPLIT --> TEXT_FEATURES["ToxicOrNot training-only text TF-IDF"]
    UMPATAN["umpatan.json<br/>Exact phrase lexicon"] --> RULES["Rule features<br/>hit + match count"]
    PROTO --> SVM["ToxicOrNot<br/>Calibrated LinearSVC"]
    RULES --> SVM
    TEXT_FEATURES --> SVM
    SVM --> PRIMARY["Primary toxicity label<br/>+ calibrated confidence"]
    SVM --> REVIEW["Rule/SVM disagreement<br/>Manual review signal"]
    RULES --> REVIEW
    SVM --> EXPERIMENT["Experimental SVM OR rules<br/>Compare only; not primary"]
    RULES --> EXPERIMENT

    TRI --> EVAL["Held-out evaluation<br/>Review metrics and errors"]
    PRIMARY --> EVAL
    REVIEW --> EVAL
    EXPERIMENT --> EVAL
    EVAL --> SAVE["Rerun notebook to refresh<br/>model artifacts and metrics"]

    TAGS["Optional X recent search<br/>Hashtag query; API token required"] --> POSTS["Fetched public posts"]
    POSTS --> GAMBLE["Gambling-ad heuristic<br/>Keyword + 2-3 digits"]
    TOGEL["togelnumbers.json<br/>No verified meanings yet"] -.-> GAMBLE
    GAMBLE --> CANDIDATE["Suspected promotion candidates<br/>Not confirmed ads"]

    SAVE -.-> DASH["Dashboarding: NULL<br/>Templates only; not wired"]
    EVAL -.-> DASH
`;

module.exports = chart;