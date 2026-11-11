# Day 2 Experiment Report — Model Comparison

## 1. Objective

Compare Decision Tree, Random Forest and Support Vector Machine (SVM) against the Day 1 Logistic Regression baseline for the Telco Customer Churn classification task. The positive class is **1 (Churn)**.

## 2. Evaluation setup

- Dataset: Telco Customer Churn
- Hold-out test size: 1,409 rows
- Test support: 1,035 non-churn examples (class 0), 374 churn examples (class 1)
- Metrics below are transcribed from the comparison results supplied from the Day 2 notebook.

For a controlled comparison, all models must use the same `xtest` and `ytest`, with the same feature preparation and target encoding. Record the exact split parameters, model hyperparameters and package versions in the notebook. Confirm that Logistic Regression was refitted on the Day 2 `xtrain`/`ytrain`; evaluating an older saved baseline on the hold-out set is not equivalent to retraining every candidate under the same protocol.

## 3. Results

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.737402 | 0.503497 | **0.770053** | 0.608879 | 0.828943 |
| Decision Tree | 0.742370 | 0.509804 | 0.764706 | **0.611765** | 0.816925 |
| SVM | 0.760114 | 0.537344 | 0.692513 | 0.605140 | 0.809807 |
| Logistic Regression | **0.805536** | **0.657233** | 0.558824 | 0.604046 | **0.841861** |

Machine-readable results: [`model_comparison_day02.csv`](../../models/model_comparison_day02.csv).

## 4. Interpretation

- **Random Forest** has the highest churn recall (0.770053), identifying the largest share of actual churners in this evaluation. Its precision is 0.503497, so a substantial fraction of flagged customers are false positives.
- **Decision Tree** has the highest churn F1-score (0.611765), narrowly ahead of Random Forest (0.608879). Its recall is 0.764706 and precision is 0.509804.
- **Logistic Regression** has the highest accuracy (0.805536), precision (0.657233) and ROC-AUC (0.841861), but lower churn recall (0.558824).
- **SVM** has accuracy 0.760114, precision 0.537344, recall 0.692513, F1 0.605140 and ROC-AUC 0.809807.

Accuracy alone is not an adequate selection criterion for this imbalanced classification task. A model may achieve high accuracy while missing many churners.

## 5. Champion decision

**Provisional champion: Decision Tree, under an F1-oriented selection policy.**

The selection rule is to maximise churn-class F1-score, with recall and ROC-AUC as tie-breakers. Decision Tree leads on F1 (0.611765). Its lead over Random Forest is small (0.002886); this does not establish statistical significance or universal superiority.

The business objective can change the choice:
- If avoiding missed churners is the priority, Random Forest is the recall leader.
- If limiting false-positive outreach is more important, Logistic Regression is the precision leader.
- If balancing precision and recall is the stated objective, Decision Tree is the current F1 leader.

The final operational choice requires business cost analysis, acceptable outreach capacity, threshold assessment and stability validation.

## 6. Confusion matrices and threshold policy

The supplied comparison table does not contain confusion matrices. Add each candidate's confusion matrix to the notebook/report to inspect true negatives, false positives, false negatives and true positives.

Choose any classification threshold using a validation set or cross-validation—not the final hold-out test set. Reserve the hold-out set for final evaluation.

## 7. Limitations

- Results represent one reported hold-out evaluation; no confidence intervals or repeated cross-validation results are included.
- The table does not report training time, inference latency, model size or a formal interpretability assessment.
- Precision and recall depend on the classification threshold.
- Results on this telecom dataset do not establish performance on a different retailer or organisation.
- Predictive performance does not prove that a retention intervention will prevent churn.

## 8. Day 2 outcome and next steps

Day 2 establishes a four-model comparison and a documented, policy-dependent provisional champion.

1. Verify that all candidates use the same split and preprocessing; retrain Logistic Regression on the same training data if required.
2. Add confusion matrices and verify ROC-AUC uses continuous scores (`predict_proba` or `decision_function`).
3. Record exact hyperparameters, split settings and environment versions.
4. Use cross-validation and validation-based threshold tuning to assess stability.
5. Save each fitted candidate separately and save the selected champion pipeline only after the selection policy is confirmed.
