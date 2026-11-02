# Day 01 — RetailMind: Customer Churn Prediction

## Experiment: Recall-Scored Grid Search

### Evaluation Metrics

| Metric          |          Result |
| --------------- | --------------: |
| Accuracy        | 0.7882 (78.82%) |
| Churn Precision | 0.6741 (67.41%) |
| Churn Recall    | 0.3897 (38.97%) |
| Churn F1-score  | 0.4936 (49.36%) |

### Observations

- Accuracy improved slightly compared with the baseline of approximately 78%.
- Churn precision increased from 0.60 to 0.6741, meaning a larger proportion of predicted churners actually churned.
- Churn recall decreased from 0.52 to 0.3897, meaning the model missed more customers who actually churned.
- The churn F1-score decreased from 0.56 to 0.4936, indicating a weaker precision–recall balance than the baseline.

### Key Learning

Using `GridSearchCV(scoring="recall")` selects hyperparameters based on mean recall across the cross-validation folds. It does not guarantee higher recall on the held-out test set.

Accuracy alone is not sufficient for evaluating a churn prediction model. Precision, recall, F1-score, and ROC-AUC should also be considered.


