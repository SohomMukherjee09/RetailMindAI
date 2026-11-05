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

# Day 02 — RetailMind: Customer Churn Prediction

## Objective
Train and compare three machine learning models to predict customer churn, prioritizing **recall for the churn class (`Yes`)** to identify at-risk customers.

## Models Compared
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

All models use the same stratified train-test split (`random_state=67`) and 5-fold stratified cross-validation (`random_state=42`), ensuring a fair comparison.

## Results

| Model | CV Recall (Yes) | Hold-out Recall | Precision | F1-score | Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **SVM** | **0.5064** | 0.4989 | **0.6436** | **0.5621** | **0.7939** | 0.7973 |
| Decision Tree | 0.4993 | **0.5161** | 0.4820 | 0.4984 | 0.7246 | 0.6581 |
| Random Forest | 0.4857 | 0.4839 | 0.6209 | 0.5439 | 0.7848 | **0.8087** |

## Champion Model: SVM

SVM was selected because it achieved the **highest mean cross-validation recall for churn (`0.5064`)**, making it the best choice under our predefined selection criterion.

Although Decision Tree achieved slightly higher hold-out recall, SVM provided a better balance of recall, precision, F1-score, and accuracy. Random Forest achieved the highest ROC-AUC.

## Saved Artifacts
- `models/churn_pipeline_champion.joblib` — selected SVM pipeline
- Day 01 model artifact — preserved separately

## Day 02 Outcome
Successfully compared three models on a shared hold-out set, selected SVM using cross-validation churn recall, and documented the decision for reproducibility and auditing.

**Next:** Improve churn detection by tuning classification thresholds and analyzing false negatives.

# Day 04 — Feature Engineering & Ensembles

## Phase-1 Deliverable

Engineered four domain features: tenure bucket, charge ratio, month-to-month contract flag, and online-security flag. Each candidate uses a leakage-safe preprocessing pipeline. `SelectKBest(f_classif, k=30)` is fitted inside cross-validation folds.

| Model | CV Recall | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Day 3 SVM (recorded) | **0.8355** | **0.8209** | 0.5876 | 0.8093 |
| XGBoost + SelectKBest | 0.7940 | 0.7995 | 0.6242 | **0.8405** |
| Random Forest + SelectKBest | 0.7492 | 0.7380 | 0.6216 | 0.8329 |
| Random Forest | 0.7378 | 0.7460 | **0.6277** | 0.8394 |

## Day 4 Outcome

XGBoost with SelectKBest is the strongest new Day 4 ensemble, improving test ROC-AUC and F1 over the Day 3 SVM. The Day 3 SVM remains the recall leader, so it should be retained when recall is the sole model-selection metric. The fitted Day 4 XGBoost pipeline, including preprocessing and feature selection, is saved as `models/churn_pipeline_day4.joblib`.



# Day 05 — Neural Networks & ANN (Keras)

## Deliverable

`src/day05.ipynb` contains a Keras MNIST warm-up and a churn ANN using normalized inputs, two ReLU hidden layers, a sigmoid output, dropout, and early stopping on validation loss. Run the notebook from the repository root after installing TensorFlow in the project environment. It saves `models/churn_ann_day5.keras` and writes `data/reports/day5_model_comparison.csv`.

## Benchmark status and interpretation

The historical Phase 1 champion is the Day 3 SVM, selected for cross-validation churn recall of **0.8355** and recorded test recall of **0.8209**. The available Day 4 report used a separate engineered feature set and lists XGBoost + SelectKBest at test recall **0.7995**, F1 **0.6242**, and ROC-AUC **0.8405**. Those historical figures are context, not a head-to-head result: they do not share a verified test feature schema.

The Day 5 notebook evaluates the ANN and the persisted Day 4 pipeline on the same Day 4 feature rows and split, then saves their metrics. The Day 3 SVM model artifact is not present in this repository, so its historical score cannot honestly be presented as a fresh same-row benchmark. TensorFlow is also not installed in the checked-in `.venv`; run the notebook in a TensorFlow-enabled environment to produce actual ANN metrics. No ANN outcome is claimed until that run is completed.

## When to use which

For this modest-sized structured churn dataset, start with the classical champion: it is cheaper to train, easier to inspect, and has a recorded recall advantage. Use the ANN only if its same-split results improve the business-priority metric enough to justify extra training and serving complexity. Neural networks become a stronger fit when RetailMind has substantially more rows, unstructured inputs such as product images/text, or shared representations across several tasks.

**Next:** Day 6 moves to CNN-based product-photo categorization.
