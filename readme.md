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

# RetailMind AI — Day 3: Multi-Model Comparison, Tuning & Bias-Variance

## Project Overview

RetailMind AI is an end-to-end retail AI platform developed incrementally. In Day 3, we improve the customer churn prediction module by comparing multiple machine learning models, tuning hyperparameters, evaluating generalization, and selecting the best-performing model using cross-validation.

## Objective

- Compare multiple classification algorithms.
- Perform hyperparameter tuning using `GridSearchCV`.
- Evaluate models using stratified cross-validation.
- Diagnose potential bias and variance using training and validation scores.
- Analyze learning curves to understand model generalization.
- Save the selected champion model for future iterations.

## Dataset

**Telco Customer Churn**

Dataset file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Target column: `Churn`

- `0` — Customer did not churn.
- `1` — Customer churned.

The dataset is cleaned by removing duplicate rows, dropping the `customerID` identifier, and converting `TotalCharges` to numeric values. Missing numerical and categorical values are handled through imputation pipelines.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Machine Learning Models

The following models are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)

## Methodology

### 1. Data Preparation
- Load and clean the dataset.
- Separate features and target.
- Split the data into 80% training and 20% testing sets.
- Use stratified splitting to preserve the target class distribution.

### 2. Preprocessing
- Apply median imputation and standard scaling to numerical features.
- Apply most-frequent imputation and one-hot encoding to categorical features.
- Use Scikit-learn pipelines to keep preprocessing inside cross-validation and reduce data leakage.

### 3. Hyperparameter Tuning
Use `GridSearchCV` to evaluate candidate hyperparameter combinations.

- Cross-validation: 5-fold `StratifiedKFold`
- Primary selection metric: Churn recall
- Random state: 42

The model with the highest mean cross-validation recall is selected as the champion.

### 4. Model Evaluation
Evaluate the tuned models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

A classification report and confusion matrix are generated for the selected champion model.

### 5. Bias-Variance Analysis
Compare training and cross-validation recall to identify potential generalization problems.

- **High bias:** Training and validation performance are both low.
- **High variance:** Training performance is substantially higher than validation performance.
- **Good generalization:** Training and validation scores are relatively close.

These patterns are diagnostic indicators rather than definitive proof of bias or variance.

### 6. Learning Curve
Generate a learning curve to compare training recall and cross-validation recall at different training-set sizes.

The curve helps assess whether additional training data or changes to model complexity may improve generalization.

## Model Selection

The champion model is selected based on the highest mean cross-validation recall.

Recall is emphasized because identifying customers who are likely to churn is important for customer retention.

The held-out test set is used for final evaluation, not for choosing the champion.

Actual model scores and the selected algorithm depend on the results obtained when the notebook is executed.

## Project Structure

```text
RetailMindAI/
├── models/
│   └── churn_v3.joblib
├── reports/
│   ├── model_comparison.csv
│   └── churn_v3_report.json
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── day3_model_comparison.ipynb
```

## How to Run

1. Clone or open the RetailMindAI repository.
2. Place the Telco Customer Churn CSV in the project directory.
3. Create and activate a Python virtual environment.
4. Install the required dependencies:

```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib ipykernel
```

5. Open the Day 3 notebook in VS Code or Jupyter.
6. Select the project's Python environment as the notebook kernel.
7. Run the notebook cells.

## Generated Artifacts

After successful execution:

- `models/churn_v3.joblib` — Saved champion model, including its preprocessing pipeline.
- `reports/model_comparison.csv` — Comparison of candidate model metrics.
- `reports/churn_v3_report.json` — Champion information, hyperparameters, and test metrics.

## Future Improvements

The next iteration will focus on:

- Feature engineering
- Principal Component Analysis (PCA)
- Boosting and bagging ensembles
- Further improvements to churn prediction performance

## Day 3 Outcome

The Day 3 increment establishes a systematic model selection and validation workflow for RetailMind AI. It produces a tuned churn prediction model, evaluates its generalization, and records experimental results for future development.
