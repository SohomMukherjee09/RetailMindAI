# RetailMind AI --- Machine-Learning Methodology

## 1. Problem Formulation

The first use case is supervised binary classification. Given customer
attributes, the model estimates whether the target label is churn (`1`)
or no churn (`0`).

The model learns statistical associations from historical labelled
examples. It does not establish why a customer churned and does not
prove that a retention action will prevent churn.

## 2. Data Preparation

1.  Load the CSV with Pandas.
2.  Inspect shape, column names, dtypes, descriptive statistics, missing
    values and target balance.
3.  Convert `TotalCharges` to numeric with invalid/blank values coerced
    to missing.
4.  Map the target: `Yes → 1`, `No → 0`.
5.  Exclude `customerID` from predictive features.
6.  Separate features (`X`) and target (`y`).
7.  Split into training and hold-out test partitions before fitting
    transformations.

A stratified split is appropriate for preserving the approximate class
proportions:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

The seed and test size must be recorded and reused across candidate
models.

## 3. Preprocessing

### Numerical branch

-   `SimpleImputer(strategy="median")`
-   `StandardScaler()`

Median imputation is less sensitive to extreme values than mean
imputation. Standardisation is particularly important for
scale-sensitive estimators such as Logistic Regression and SVM.

### Categorical branch

-   `SimpleImputer(strategy="most_frequent")`
-   `OneHotEncoder(handle_unknown="ignore")`

One-hot encoding represents categories as indicator columns without
imposing an artificial numeric ordering.

### Combined transformer

`ColumnTransformer` applies the numerical and categorical branches to
their respective columns. The resulting transformer is placed before the
classifier in a single scikit-learn `Pipeline`.

## 4. Candidate Algorithms

### Day 1: Logistic Regression

A linear classification baseline. It provides a relatively simple
benchmark and can produce class probabilities. It is useful as a
reference point, not an assumed final winner.

### Day 2: Decision Tree

A tree-based classifier that recursively partitions the feature space.
It can capture non-linear relationships and interactions, but an
unconstrained tree can overfit. Important controls include `max_depth`,
`min_samples_split`, `min_samples_leaf` and `criterion`.

### Day 2: Random Forest

An ensemble of decision trees trained using bootstrap samples and
feature subsampling. Aggregating trees can reduce the variance of an
individual tree. Important controls include `n_estimators`, `max_depth`,
`min_samples_leaf`, `max_features` and `random_state`.

### Day 2: Support Vector Machine

A margin-based classifier. An RBF kernel can model non-linear decision
boundaries. SVM is sensitive to feature scale, so the numerical scaling
branch is important. Key controls include `C`, `kernel` and `gamma`. For
probability output with `SVC`, set `probability=True` before fitting;
this adds computational cost.

## 5. Evaluation Metrics

Use the same held-out test rows for all candidates.

Let: - TP = churners correctly predicted as churn. - TN = non-churners
correctly predicted as non-churn. - FP = non-churners incorrectly
flagged as churn. - FN = churners incorrectly predicted as non-churn.

  --------------------------------------------------------------------------------------------------
  Metric                  Formula                                           Interpretation
  ----------------------- ------------------------------------------------- ------------------------
  Accuracy                `(TP + TN) / (TP + TN + FP + FN)`                 Overall correctness

  Precision               `TP / (TP + FP)`                                  Reliability of churn
                                                                            alerts

  Recall                  `TP / (TP + FN)`                                  Fraction of actual
                                                                            churners detected

  F1                      `2 × precision × recall / (precision + recall)`   Harmonic balance of
                                                                            precision and recall

  Specificity             `TN / (TN + FP)`                                  Fraction of non-churners
                                                                            correctly left unflagged

  ROC-AUC                 Area under ROC curve                              Ranking/discrimination
                                                                            across thresholds
  --------------------------------------------------------------------------------------------------

For ROC-AUC, use continuous scores for the positive class, for example:

``` python
churn_scores = pipeline.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, churn_scores)
```

For an estimator without `predict_proba`, use its continuous
`decision_function` score where supported. Do not use hard 0/1
predictions as the preferred ROC-AUC input.

## 6. Class Imbalance and SMOTE

The target distribution should be measured, not assumed. Accuracy alone
can hide poor detection of the minority churn class.

SMOTE generates synthetic minority-class examples. It may improve
minority-class recall in some settings, but it does not guarantee better
accuracy, recall or generalisation. It can also increase false
positives.

If evaluated: - Split before resampling. - Apply SMOTE to training data
only. - Place it inside an imbalanced-learn pipeline so it is applied
during fitting, not to the hold-out set. - Compare precision, recall,
F1, ROC-AUC and confusion matrices. - Document the change in error
trade-offs.

SMOTE is an optional experiment, not a requirement for the Day 1/Day 2
scope, which specifies Pandas/scikit-learn and scikit-learn
respectively.

## 7. Model Selection Policy

The primary business metric is recall for churn class `1`, because false
negatives represent churners not identified by the model. But recall
must not be optimised in isolation.

Champion selection should: 1. Use a shared fixed hold-out set. 2.
Compare recall, precision, F1, ROC-AUC and confusion matrix. 3. Consider
the retention team's capacity to investigate alerts. 4. Consider the
relative costs of false negatives and false positives. 5. Record the
selected model, parameters, artifact and reason. 6. Preserve the Day 1
baseline for future comparison.

If model selection or extensive tuning is performed on the hold-out test
set, the reported test performance becomes optimistic. Prefer
cross-validation on the training set for tuning, then evaluate the
selected configuration once on the hold-out set.

## 8. Persistence and Inference

Persist the complete fitted pipeline---not just the estimator---using
Joblib. This preserves the fitted imputers, encoder, scaler and
classifier together.

Only load trusted Joblib artifacts. Pickle-based formats can execute
code during deserialisation. Record the Python and scikit-learn versions
used to train the artifact.

## 9. Reproducibility Record

For every experiment, record: - Dataset source and file/version. -
Target mapping and feature exclusions. - Train/test split and random
seed. - Preprocessing steps. - Estimator and hyperparameters. - Metric
values and confusion matrix. - Package versions. - Artifact filename and
creation date. - Interpretation and limitations.
