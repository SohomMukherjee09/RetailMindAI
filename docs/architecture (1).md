# Experiment Report --- Day 2: Decision Tree, Random Forest and SVM

## 1. Experiment Objective

Compare three classifiers against the Logistic Regression baseline:

-   Decision Tree
-   Random Forest
-   Support Vector Machine (SVM)

The business emphasis is identifying customers who churn (`Churn = 1`).
Recall is therefore a primary metric, while precision, F1, ROC-AUC and
the confusion matrix provide the necessary context.

**Status:** In progress. A champion model has not yet been confirmed.

## 2. Fair-Comparison Protocol

All candidates must use: - The same source CSV and cleaning rules. - The
same feature columns and target encoding. - The same preprocessing
design. - The same fixed train/test rows. - The same positive-class
definition (`1 = churn`). - The same evaluation functions.

Recommended split configuration:

``` python
train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

The Day 1 Logistic Regression baseline should also be re-trained and
evaluated on this exact split before the comparison table is finalised.

## 3. Candidate Pipeline Design

Each candidate should be a separate pipeline:

``` text
Shared preprocessing -> DecisionTreeClassifier
Shared preprocessing -> RandomForestClassifier
Shared preprocessing -> SVC
```

Suggested starting estimators (not tuned final configurations):

``` python
DecisionTreeClassifier(random_state=42)

RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)
```

Record the exact parameters actually used. `probability=True` is
required if ROC-AUC is calculated from SVC probabilities; alternatively
use `decision_function` scores.

## 4. Development Evidence Shared So Far

A confusion matrix was shared:

``` text
[[926, 109],
 [165, 209]]
```

Assuming the matrix uses scikit-learn's default label order `[0, 1]` and
the positive class is churn (`1`:

               Predicted 0   Predicted 1
  ---------- ------------- -------------
  Actual 0        TN = 926      FP = 109
  Actual 1        FN = 165      TP = 209

Derived metrics:

  Metric                Calculation              Value
  --------------------- --------------------- --------
  Accuracy              (926 + 209) / 1409      80.55%
  Precision (class 1)   209 / (209 + 109)       65.72%
  Recall (class 1)      209 / (209 + 165)       55.88%
  Specificity           926 / (926 + 109)       89.47%
  F1 (class 1)          2 × P × R / (P + R)     60.39%

**Provenance warning:** The development conversation did not establish
unambiguously which model, split seed or experiment configuration
produced this matrix. These calculations are mathematically derived from
the supplied matrix, but must not be attributed to Decision Tree, Random
Forest or SVM until the run is confirmed.

## 5. Class Imbalance and SMOTE Decision

The dataset has a class imbalance that should be quantified from the
actual CSV using `value_counts(normalize=True)`.

SMOTE was experimented with and then removed after accuracy decreased.
This is not, by itself, sufficient evidence that SMOTE is worse: the
effect on churn recall, precision, F1, ROC-AUC and false-positive count
must also be checked.

For the current Day 2 comparison, a clean no-SMOTE baseline is
reasonable. If SMOTE is revisited: - Apply it only to training data. -
Use an imbalanced-learn pipeline with preprocessing and resampling
correctly ordered. - Never resample the full dataset before train/test
split. - Compare all relevant metrics on the untouched hold-out set.

## 6. Results Table --- Complete After Reproducible Runs

  --------------------------------------------------------------------------------
  Model          Accuracy   Precision Recall (1)     F1 (1)    ROC-AUC Status
                                  (1)                                  
  ------------ ---------- ----------- ---------- ---------- ---------- -----------
  Logistic            TBD         TBD        TBD        TBD        TBD Day 1
  Regression                                                           baseline;
                                                                       rerun on
                                                                       shared
                                                                       split

  Decision            TBD         TBD        TBD        TBD        TBD Pending
  Tree                                                                 

  Random              TBD         TBD        TBD        TBD        TBD Pending
  Forest                                                               

  SVM                 TBD         TBD        TBD        TBD        TBD Pending
  --------------------------------------------------------------------------------

## 7. Champion Selection

**Champion:** Not selected yet.

Select only after all candidates are evaluated on the same hold-out set.
Prioritise churn recall, then review precision, F1, ROC-AUC and
operational alert volume. A model that flags nearly every customer may
achieve high recall but may not be useful to a retention team.

Document: - Selected model and parameters. - Metric values. - Confusion
matrix. - Trade-off versus baseline. - Why the model is operationally
preferable. - Artifact path and run metadata.

## 8. Next Actions

1.  Fix `random_state=42` and use one shared split.
2.  Re-run Logistic Regression on that split.
3.  Train Decision Tree, Random Forest and SVM.
4.  Generate a single comparison table.
5.  Inspect confusion matrices and churn-class recall.
6.  Select and save the champion without overwriting the baseline.
7.  Update the README and model card with verified results.
