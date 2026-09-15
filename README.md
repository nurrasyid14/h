<<<<<<< HEAD
# *Smart System Workshop: Classical ML for Sentiment Analysis**
3rd Smt Text Mining Recap
===
## **Overviews**

### **Objectives**
---
## **Methodologies**
### **Architecture**
```
project/
│
├── data/
│   ├── raw/
│   │   └── indotoxic2024_annotated_data-3.jsonl
│   │
│   ├── interim/
│   │   ├── cleaned.csv
│   │   └── no_spam.csv
│   │
│   └── processed/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── tfidf.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   └── label_encoder.joblib
│
├── reports/
│   ├── figures/
│   │   ├── label_distribution.png
│   │   ├── confusion_matrix.png
│   │   ├── wordcloud.png
│   │   └── feature_importance.png
│   │
│   └── metrics.json
│
├── dashboard/
│   ├── app.py
│   ├── routes.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── analytics.html
│   │   ├── prediction.html
│   │   └── evaluation.html
│   │
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   │
│   └── services/
│       ├── inference.py
│       └── charts.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── requirements.txt
├── README.md
└── run.py
```
---
## **Workflow**
---
## **Algorithm**

---
## **Output Expectation**

---
## **References**
=======
# IndoToxic 2024

Project structure for preprocessing, modeling, evaluation, and dashboard development.

Run preprocessing from `prepro.ipynb`, then use the generated files under `data/processed/`.
>>>>>>> master
