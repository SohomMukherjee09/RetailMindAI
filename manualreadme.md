# RetailMind AI

### An Incremental, End-to-End Machine Learning Platform for Customer Intelligence

**Project status:** In active development\
**Current increments documented:** Day 1 --- End-to-End ML Pipeline &
ML-as-a-Service foundation; Day 2 --- Model comparison (in progress)\
**Domain:** Customer analytics / churn prediction\
**Primary task:** Binary classification

------------------------------------------------------------------------

## 1. Executive Summary

RetailMind AI is an incrementally developed machine-learning platform
intended to turn customer data into actionable business intelligence.
Its first use case is customer churn prediction: estimating whether a
customer is likely to discontinue a service so that a retention team can
prioritise timely, evidence-based engagement.

The initial increment establishes the machine-learning foundation: data
ingestion, preprocessing, model training, evaluation, persistence and a
reusable prediction interface. The second increment investigates
alternative classifiers---Decision Tree, Random Forest and Support
Vector Machine (SVM)---and compares them with the Logistic Regression
baseline.

The project is being developed as a 35-day sequence of integrated
increments. Each increment adds a defined capability to one continuing
codebase. The roadmap is an intended development plan, not a claim that
all future capabilities already exist.

## 2. Business Problem

Customer churn can reduce recurring revenue and increase the cost of
acquiring replacement customers. A retention team needs a way to
identify customers who may be at risk before they leave.

RetailMind AI frames this as a supervised binary-classification problem:

-   **Class 0 --- No churn:** the customer did not churn in the
    dataset's observation period.
-   **Class 1 --- Churn:** the customer churned in the dataset's
    observation period.

The model produces a prediction from customer attributes. A business
workflow could use that prediction to prioritise review or retention
outreach. A prediction is a decision-support signal---not proof that a
customer will leave, nor an instruction to automatically contact or
treat a customer.

## 3. Current Scope

### Implemented or evidenced in the working increment

-   Telco Customer Churn CSV loaded with Pandas.
-   Initial dataset inspection using `head`, `info`, `describe`, `shape`
    and missing-value checks.
-   Target encoding: `Yes → 1`, `No → 0`.
-   `TotalCharges` converted to numeric, with unparseable values
    represented as missing.
-   Customer identifier excluded from the feature matrix by positional
    slicing (`df.iloc[:, 1:-1]`) in the supplied notebook.
-   Numerical preprocessing with median imputation and standardisation.
-   Categorical preprocessing with most-frequent imputation and one-hot
    encoding.
-   A scikit-learn preprocessing/model pipeline.
-   Logistic Regression trained and persisted with Joblib.
-   Day 2 exploration of Decision Tree, Random Forest and SVM.

### Not yet claimed as complete

-   A production REST API or web service.
-   Cloud deployment, authentication, monitoring or automated
    retraining.
-   A fully reproducible champion-model selection based on one fixed
    hold-out split.
-   A production-grade data validation or model monitoring system.

## 4. Technology Stack

  -----------------------------------------------------------------------
  Technology                          Role in the documented increments
  ----------------------------------- -----------------------------------
  Python                              Implementation language

  Pandas                              CSV ingestion and tabular data
                                      inspection

  scikit-learn                        Preprocessing, model pipelines,
                                      training and evaluation

  Joblib                              Persisting fitted model pipelines

  Jupyter Notebook                    Exploratory work and incremental
                                      experiments

  Git / GitHub                        Version control and project history
  -----------------------------------------------------------------------

Day 1 and Day 2 constraints specify the libraries to use for those
increments. Additional tools should be introduced only when a later
increment calls for them.

## 5. Repository Structure

``` text
RetailMindAI/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── logistic_reg.joblib
│   └── ...                    # additional model artifacts as validated
├── notebooks/
│   ├── day01.ipynb
│   └── day02.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── model_card.md
│   ├── roadmap.md
│   ├── sme_review_brief.md
│   └── experiments/
│       ├── day01_baseline.md
│       └── day02_comparison.md
├── README.md
└── requirements.txt
```

This is the recommended target structure. Keep only files that actually
exist in the repository; do not create empty artifacts merely to match
the diagram.

## 6. Machine-Learning Workflow

``` text
Telco CSV
   |
   v
Data inspection and validation
   |
   v
Feature / target separation
   |
   v
Train-test split
   |
   v
ColumnTransformer
   |-- Numerical: median imputation -> StandardScaler
   |-- Categorical: most-frequent imputation -> OneHotEncoder
   |
   v
Classifier
   |
   v
Evaluation
   |-- Accuracy
   |-- Precision
   |-- Recall (positive class = churn)
   |-- ROC-AUC (continuous score / churn probability)
   |
   v
Persist fitted pipeline with Joblib
   |
   v
Reusable prediction function (service foundation)
```

The preprocessing object must be fitted only on training data. Keeping
preprocessing and the classifier in one scikit-learn `Pipeline` helps
ensure that training-time transformations are also applied at prediction
time.

## 7. Evaluation Principles

The positive class is churn (`1`). Metrics must be interpreted in the
context of the retention use case.

-   **Accuracy:** overall fraction of correct predictions.
-   **Precision:** among customers predicted to churn, the fraction who
    actually churned.
-   **Recall:** among actual churners, the fraction identified by the
    model.
-   **ROC-AUC:** ranking/discrimination ability across thresholds,
    calculated from a continuous score such as `predict_proba(X)[:, 1]`
    for the churn class.

Recall is an important business metric because false negatives are
customers who churn but were not flagged. However, maximising recall
without considering precision can create too many false alarms and
overwhelm a retention team. Model selection should consider recall
alongside precision, ROC-AUC, operational capacity and the cost of
errors.

## 8. Experiment Status and Results

### Day 1 --- Logistic Regression baseline

A Logistic Regression pipeline was trained and saved. A ROC-AUC value of
**0.84 was reported during development**, but the exact notebook run,
split seed and full metric set must be captured before treating this as
a reproducible benchmark. See [Day 1 experiment
report](docs/experiments/day01_baseline.md).

### Day 2 --- Model comparison

Decision Tree, Random Forest and SVM are the planned candidates. A
confusion matrix was shared during development:

``` text
[[926, 109],
 [165, 209]]
```

If rows and columns follow scikit-learn's default order `[0, 1]`, this
corresponds to TN=926, FP=109, FN=165 and TP=209. The associated model
identity and exact split/run configuration have not yet been confirmed.
See [Day 2 experiment report](docs/experiments/day02_comparison.md).

Do not label a model as champion until all candidates have been
evaluated on the same fixed hold-out set and the selection rationale is
recorded.

## 9. Reproducibility and Model Governance

For comparable experiments:

1.  Use the same source dataset and documented data-cleaning steps.
2.  Use one fixed train-test split for all candidate models.
3.  Set and record `random_state` where supported.
4.  Fit imputers, encoders, scalers and resampling steps using training
    data only.
5.  Calculate ROC-AUC from continuous scores, not thresholded class
    labels.
6.  Record library versions, parameters, split configuration and
    metrics.
7.  Save each candidate separately; do not overwrite the Day 1 baseline.
8.  Record the selected model and the reason for selection.

If SMOTE is evaluated in a future experiment, apply it only to training
folds/data inside an imbalanced-learn pipeline. Never oversample the
full dataset before splitting. SMOTE is not part of the specified Day
1/Day 2 scikit-learn-only scope.

## 10. Setup

Create and activate a Python virtual environment, install the
dependencies used by the relevant increment, and launch Jupyter
Notebook.

Example commands (adjust for your operating system and environment):

``` bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install pandas scikit-learn joblib jupyter
jupyter notebook
```

Place the Kaggle Telco Customer Churn CSV at
`data/Telco-Customer-Churn.csv`. Notebook paths are relative to the
notebook's working directory; verify the working directory before
running cells.

## 11. Limitations

-   The dataset is a public telecom churn dataset and may not represent
    a particular retailer or organisation.
-   Predictions reflect patterns in historical labelled data and can be
    affected by sampling, label definitions and distribution shift.
-   Model scores do not establish causality or prove that an
    intervention will prevent churn.
-   The current project is a development increment, not a production
    deployment.
-   Business use requires validation, privacy review, monitoring, human
    oversight and an assessment of intervention costs and benefits.

## 12. Documentation Index

-   [Architecture](docs/architecture.md)
-   [Data dictionary](docs/data_dictionary.md)
-   [Methodology](docs/methodology.md)
-   [Model card](docs/model_card.md)
-   [Day 1 experiment report](docs/experiments/day01_baseline.md)
-   [Day 2 experiment report](docs/experiments/day02_comparison.md)
-   [SME review brief](docs/sme_review_brief.md)
-   [35-day roadmap](docs/roadmap.md)

------------------------------------------------------------------------

**Project positioning:** RetailMind AI is an evolving ML platform. This
repository documents what has been implemented, what has been measured
and what remains planned---so that each engineering decision can be
reviewed and reproduced.
