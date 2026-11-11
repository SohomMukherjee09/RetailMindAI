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


# Day 09 — LSTM-Based Demand Forecasting
### RetailMind AI | Time-Series Forecasting Module

## 1. Project Overview

The objective of Day 9 was to develop a time-series forecasting module using **Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM)** networks to predict future product demand across multiple retail stores.

Retail businesses need reliable demand forecasts to make inventory decisions. Underestimating demand can lead to stockouts and lost revenue, while overestimating demand can result in excess inventory and increased holding costs.

This increment introduces a deep-learning-based forecasting component into RetailMind AI, using historical sales patterns to predict the next sales observation for individual store-item combinations.

**Business Objective:**
- Forecast product demand using historical sales.
- Learn temporal dependencies from sequential sales data.
- Evaluate forecasting performance using regression metrics.
- Generate predictions for unseen test observations.
- Save the trained model for integration into the RetailMind inventory-planning module.

---

## 2. Dataset

**Dataset:** Store Item Demand Forecasting Challenge (Kaggle)

The dataset contains historical daily sales records for multiple stores and products.

### Dataset Features

| Feature | Description |
|---|---|
| date | Daily sales transaction date |
| store | Unique store identifier |
| item | Unique product identifier |
| sales | Number of units sold |

### Dataset Characteristics

- Training records: 913,000
- Number of stores: 10
- Number of unique products: 50
- Total store-item combinations: 500
- Granularity: Daily sales
- Target variable: `sales`

### Initial Data Inspection

Performed exploratory data analysis to understand the dataset before constructing time-series sequences.

Operations included:
- Dataset dimensions and structural inspection.
- Data type verification.
- Missing-value checks.
- Duplicate-record checks.
- Minimum and maximum sales inspection.
- Date-range inspection.
- Unique store and product identification.

The dataset was inspected for missing values and duplicate records before model preparation.

---

## 3. Technical Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations and sequence generation |
| TensorFlow | Deep learning framework |
| Keras | LSTM model construction and training |
| Matplotlib | Forecast visualization |
| Scikit-learn | MAE and MSE evaluation metrics |

---

## 4. Methodology & Implementation

The forecasting pipeline was implemented through the following stages.

### Step 1 — Data Preparation

Loaded the training and test datasets using Pandas.

The historical sales data was organized by individual store-item combinations.

Each combination represents an independent daily sales time series.

This ensures that the model learns product-specific and store-specific historical sales patterns rather than treating all sales observations as one continuous sequence.

### Step 2 — Sales Normalization

Applied Min-Max normalization to the sales target.

Formula:

\[
X_{scaled}=\frac{X-X_{min}}{X_{max}-X_{min}}
\]

Normalization helps stabilize the numerical scale of the input data and supports more effective neural network optimization.

The original sales scale was restored after prediction using the inverse transformation.

### Step 3 — Sliding Window Sequence Generation

Implemented a sliding-window approach to convert historical time-series data into supervised learning samples.

**Configuration:**
- Look-back window: 30 days
- Prediction horizon: Next observation
- Input features: Historical sales
- Target: Sales on the following day

For example:

| Input Sequence | Target |
|---|---|
| Day 1–30 | Day 31 |
| Day 2–31 | Day 32 |
| Day 3–32 | Day 33 |

Used NumPy's `sliding_window_view()` to efficiently generate overlapping sequences.

This allowed the historical time series to be transformed into input-output pairs suitable for LSTM training.

### Step 4 — Temporal Train-Validation Split

Instead of randomly splitting the time-series observations, the most recent 30 days from each store-item combination were reserved for validation.

- Training: Historical observations excluding the last 30 days.
- Validation: Last 30 days, with the preceding 30 days retained as input context.

This preserves chronological ordering and provides a more realistic evaluation of forecasting performance.

### Step 5 — LSTM Model Architecture

Implemented a Sequential neural network using TensorFlow/Keras.

**Model architecture:**

```python
model = Sequential([
    tf.keras.Input(shape=(30, 1)),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
```

**Architecture breakdown:**

| Layer | Configuration | Purpose |
|---|---|---|
| Input | (30, 1) | Accepts 30 historical sales observations |
| LSTM | 32 units | Learns temporal dependencies |
| Dropout | 0.2 | Helps reduce overfitting |
| Dense | 1 neuron | Produces next-step sales prediction |

### Step 6 — Model Compilation

The model was compiled using:

- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Evaluation metric: Mean Absolute Error (MAE)

```python
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

### Step 7 — Model Training

Training configuration:

| Parameter | Value |
|---|---|
| Epochs | 15 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss | MSE |
| Validation data | Chronological holdout |
| LSTM units | 32 |
| Dropout | 0.2 |

The model was trained using the generated training sequences and evaluated against the validation sequences after each epoch.

### Step 8 — Model Evaluation

Predictions were generated on the validation dataset.

The normalized predictions and actual values were converted back to their original sales scale before calculating the evaluation metrics.

Two regression metrics were used.

**Mean Absolute Error (MAE)**

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

MAE measures the average absolute difference between actual and predicted sales.

**Root Mean Squared Error (RMSE)**

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

RMSE penalizes larger prediction errors more heavily than MAE.

### Step 9 — Prediction Visualization

Used Matplotlib to visualize actual versus predicted sales over the first 200 validation observations.

This provides a visual sanity check of how closely the LSTM follows the observed sales patterns.

### Step 10 — Test Prediction and Model Export

Generated predictions for the Kaggle test dataset using the most recent 30 historical observations for each store-item combination.

The normalized predictions were converted back into the original sales scale, and negative predictions were clipped to zero.

The resulting predictions were exported as:

`submission_lstm.csv`

The trained Keras model was saved as:

`demand_lstm.keras`

---

## 5. Model Performance

### Validation Results

| Evaluation Metric | Result |
|---|---:|
| MAE | **6.7457** |
| RMSE | **8.9832** |

### Training Observations

The model was trained for 15 epochs.

| Epoch | Training MAE | Validation MAE |
|---|---:|---:|
| 1 | 0.0333 | 0.0330 |
| 5 | 0.0287 | 0.0288 |
| 10 | 0.0284 | 0.0284 |
| 13 | 0.0283 | 0.0277 |
| 15 | 0.0282 | 0.0292 |

*The epoch-level MAE values are calculated on normalized sales data, whereas the final evaluation metrics are calculated after inverse transformation.*

### Interpretation

The final validation MAE of approximately 6.75 indicates that predictions differ from actual sales by around 6.75 units on average.

The RMSE of approximately 8.98 indicates that larger forecasting errors also contribute to the overall prediction error.

The training and validation metrics remained relatively close throughout training, suggesting that the model did not exhibit a substantial training-validation performance gap.

These results establish an initial LSTM forecasting baseline for RetailMind AI. Further comparisons against statistical and machine-learning baselines are necessary before considering the model suitable for production inventory decisions.

---

## 6. Engineering Challenges & Resolutions

### Challenge 1 — Converting Time Series into Supervised Learning Data

**Problem:**

LSTM networks require sequential input tensors, whereas the original dataset contains individual daily sales records.

**Resolution:**

Implemented a sliding-window sequence generator using NumPy's `sliding_window_view()`.

This converted historical sales into fixed-length 30-day sequences with the following observation as the target.

### Challenge 2 — Preventing Temporal Data Leakage

**Problem:**

Random train-validation splitting can introduce future observations into training data, resulting in unrealistic forecasting evaluation.

**Resolution:**

Used chronological splitting within each store-item group, reserving the latest 30 days for validation.

The validation inputs were constructed using the preceding 30 days as historical context.

### Challenge 3 — Handling Multiple Store-Item Time Series

**Problem:**

The dataset contains 500 independent store-item combinations. Combining them without grouping could create artificial relationships between unrelated sales observations.

**Resolution:**

Grouped the dataset by `store` and `item` and generated sequences independently for every combination.

### Challenge 4 — TensorFlow GPU Compatibility on Windows

**Problem:**

During training, TensorFlow reported that native Windows GPU support was unavailable for TensorFlow versions 2.11 and above.

Even though the development machine contained an NVIDIA GPU, the installed TensorFlow environment could not utilize it through the standard native Windows configuration.

**Resolution:**

The model was successfully trained using CPU execution.

This allowed development and experimentation to continue without blocking the forecasting implementation.

### Challenge 5 — Multi-Day Test Forecasting

**Problem:**

The Kaggle test dataset requires predictions across multiple future dates, whereas the implemented LSTM architecture predicts one next-step observation.

**Resolution:**

Implemented a simple baseline inference strategy that predicts the next sales value for each store-item combination using its latest 30 historical observations.

The resulting prediction is assigned to the corresponding test records.

**Limitation:** This is not a recursive multi-step forecasting strategy. The same forecast is assigned to every future date of a given store-item combination. It should be improved before production use.

---

## 7. Project Deliverables

| Artifact | Description |
|---|---|
| `demand_lstm.keras` | Trained LSTM model |
| `submission_lstm.csv` | Generated Kaggle test predictions |
| Day 9 notebook | Complete implementation and evaluation |
| README.md | Technical documentation |

---

## 8. Current Limitations & Future Improvements

Although the initial LSTM pipeline is implemented, several improvements can strengthen forecasting accuracy and production readiness.

**Planned improvements:**

1. **Recursive Multi-Step Forecasting:** Generate sequential predictions for every future date rather than repeating a single prediction.

2. **Feature Engineering:** Introduce day-of-week, month, seasonal and holiday-related features.

3. **Scaling Improvements:** Fit normalization parameters only on the training partition to avoid validation-data leakage.

4. **Hyperparameter Optimization:** Experiment with LSTM units, look-back windows, batch sizes and learning rates.

5. **Baseline Comparison:** Compare LSTM against persistence, moving-average and traditional forecasting models.

6. **Advanced Architectures:** Experiment with stacked LSTM, GRU and other sequence-modeling architectures.

7. **Inventory Integration:** Use predicted demand as an input to inventory replenishment and stock-level planning.

---

## 9. Business Relevance — RetailMind AI

This module establishes the forecasting foundation for the inventory-planning component of RetailMind AI.

The intended downstream workflow is:

```text
Historical Sales Data
        |
        v
Store-Item Time Series
        |
        v
Data Normalization
        |
        v
30-Day Sliding Windows
        |
        v
LSTM Demand Forecasting
        |
        v
Predicted Product Demand
        |
        v
Inventory Planning
        |
        v
Replenishment Decisions
```

The long-term objective is to transform historical retail transactions into actionable demand signals that can support inventory optimization.

---

## 10. Key Learnings

Through this increment, the following concepts were implemented and explored:

- Fundamentals of RNN and LSTM architectures.
- Sequential data preparation for deep learning.
- Sliding-window time-series transformations.
- Min-Max normalization and inverse scaling.
- Chronological validation strategies.
- LSTM model construction using Keras.
- Training with Adam optimization and MSE loss.
- Regression evaluation using MAE and RMSE.
- Model serialization using the Keras `.keras` format.
- Generating predictions for unseen time-series observations.

---

## 11. Conclusion

Day 9 successfully introduced a deep-learning-based time-series forecasting module into RetailMind AI.

An LSTM model was trained on historical sales sequences from multiple stores and products, achieving a validation MAE of **6.7457** and RMSE of **8.9832** on the original sales scale.

The implementation covers the complete initial forecasting workflow, from historical data preparation and sequence generation to model training, evaluation, visualization and test prediction generation.

The saved model provides a foundation for subsequent inventory-planning functionality.

**Status:** Initial LSTM forecasting pipeline implemented. Multi-step forecasting and further model optimization remain future improvements.

---

*RetailMind AI — An incremental end-to-end retail intelligence platform.*