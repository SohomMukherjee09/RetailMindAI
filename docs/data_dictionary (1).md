# RetailMind AI --- Technical Architecture

## 1. Architecture Objective

The initial architecture separates data preparation, model training,
model artifacts and prediction usage. The intent is to create a small,
understandable ML backbone that can evolve into a broader
customer-intelligence platform.

This document describes the current logical design and the intended
extension points. It does not claim that an API, cloud deployment or
production monitoring system has been implemented.

## 2. Logical Component View

``` text
                 OFFLINE / DEVELOPMENT WORKFLOW
+-----------------------+
| Telco Customer CSV    |
+-----------+-----------+
            |
            v
+-----------------------+
| Data inspection       |
| validation / cleaning |
+-----------+-----------+
            |
            v
+-------------------------------+
| Train / test split            |
| X = customer features         |
| y = churn label (0 / 1)       |
+---------------+---------------+
                |
                v
+----------------------------------------------+
| Preprocessing: ColumnTransformer             |
|                                              |
| Numerical branch:                            |
| median imputation -> StandardScaler          |
|                                              |
| Categorical branch:                          |
| most-frequent imputation -> OneHotEncoder    |
+----------------------+-----------------------+
                       |
                       v
              +------------------+
              | Classifier       |
              | Logistic / Tree  |
              | Forest / SVM     |
              +--------+---------+
                       |
                       v
              +------------------+
              | Evaluation       |
              | metrics + review |
              +--------+---------+
                       |
                       v
              +------------------+
              | Joblib artifact  |
              | fitted pipeline  |
              +--------+---------+
                       |
                       v
                 PREDICTION USE
              +------------------+
              | predict() helper |
              | input validation |
              | model inference  |
              +------------------+
```

## 3. Component Responsibilities

  -----------------------------------------------------------------------
  Component               Responsibility          Key design
                                                  consideration
  ----------------------- ----------------------- -----------------------
  Data source             Supplies labelled       Keep raw data unchanged
                          customer records        and document its
                                                  provenance

  Data validation         Checks schema, types,   Fail clearly when
                          target values and       required columns are
                          missingness             absent

  Feature/target          Excludes identifier and Avoid target leakage
  separation              separates `Churn`       

  Preprocessing           Imputes, scales and     Fit only on training
                          encodes                 data

  Classifier              Learns churn patterns   Compare candidates on
                                                  identical data

  Evaluation              Quantifies predictive   Prioritise
                          behaviour               positive-class recall
                                                  with precision context

  Artifact storage        Stores fitted pipeline  Preserve model version
                                                  and training metadata

  Prediction helper       Loads artifact and      Enforce feature schema
                          scores new rows         and input validation
  -----------------------------------------------------------------------

## 4. Data Flow and Feature Contract

The current notebook uses `customerID` as an identifier and excludes it
from model features. The target column `Churn` is mapped to integer
labels: `No → 0`, `Yes → 1`.

The feature matrix should contain the same named columns at training and
inference time. A prediction request must not include the target column.
The persisted pipeline should own all fitted transformations so callers
do not separately re-encode or scale data.

## 5. Pipeline Design

A `ColumnTransformer` applies different transformations to different
column groups:

-   Numerical columns: median imputation, then standardisation.
-   Categorical columns: most-frequent imputation, then one-hot
    encoding.

A scikit-learn `Pipeline` chains the transformer and estimator. This
design reduces training/inference mismatch and makes the complete fitted
workflow persistable as one artifact.

For categorical features, `OneHotEncoder(handle_unknown="ignore")` is
recommended so an unseen category does not cause inference to fail. The
encoder's output is sparse by default; model compatibility and memory
behaviour should be checked for each candidate.

## 6. Training and Evaluation Boundary

The split must happen before fitting preprocessing. The pipeline is
fitted on `X_train, y_train`; the test set is used only for evaluation.
If resampling is evaluated, it must occur inside the training pipeline
after the split and only on training data.

All candidate models should use the same fixed hold-out rows. A fair
comparison also requires the same positive-class definition and metric
calculation.

## 7. Model Artifact and Inference Contract

Recommended artifact naming:

``` text
models/
├── logistic_reg_day01.joblib
├── decision_tree_day02.joblib
├── random_forest_day02.joblib
├── svm_day02.joblib
└── champion_model.joblib
```

Artifact names should identify the model and increment. The champion
artifact should only be updated after a documented selection decision.

A prediction helper should: 1. Load the selected artifact. 2. Validate
required feature names and input types. 3. Call `predict()` for the
class label. 4. Call `predict_proba()` only if the estimator supports it
and a probability is required. 5. Return a documented result containing
the predicted class and, when available, churn score. 6. Avoid silently
accepting malformed or incomplete records.

SVM probability estimates require `probability=True` during fitting.
Probability calibration and threshold selection may need separate
evaluation if scores are used for business decisions.

## 8. Security, Privacy and Operational Considerations

Before real customer data is used: - Confirm lawful basis,
consent/notice requirements and internal data-handling policy. -
Minimise personal data; customer identifiers should not be used as
predictive features without a justified purpose. - Restrict access to
raw data and model artifacts. - Do not log sensitive customer attributes
unnecessarily. - Validate incoming payloads and protect any future
API. - Monitor data drift, prediction distributions and model
performance when labels become available. - Require human review for
consequential retention actions.

## 9. Future Architecture Extensions

Potential future additions, subject to the project roadmap: - A REST
inference layer (for example, FastAPI). - Request validation and
versioned prediction schemas. - Experiment and artifact metadata
tracking. - Automated tests and CI. - Model monitoring and drift
detection. - Scheduled retraining and controlled model promotion. -
Business-facing dashboards and retention workflow integration.

These are extension points, not current capabilities.
