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

- Training records: 912,000
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
| 12 | 0.0283 | 0.0277 |
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
# RetailMind AI — Day 10: Seq2Seq & Attention Forecasting

## Overview

Day 10 upgrades the RetailMind AI demand forecasting module from the Day 9 single-step LSTM to a **multi-horizon sequence-to-sequence (Seq2Seq) model with temporal attention**.

The model reads the previous 30 days of sales and predicts the next 14 days in one forward pass. An attention layer allows each decoder forecast step to weight the encoder's historical hidden states differently.

The implementation uses TensorFlow / Keras for neural-network modeling. Pandas and NumPy are used for data preparation and metric calculations.

## Dataset and continuity with Day 9

This increment carries forward the dataset actually used in the Day 9 implementation:

- **Dataset:** Store Item Demand Forecasting
- **Training file:** `data/store_item_demand/train.csv`
- **Test file:** `data/store_item_demand/test.csv`
- **Series identifiers:** `store`, `item`
- **Target:** daily `sales`
- **History window:** 30 days
- **Forecast horizon:** 14 days
- **Evaluation holdout:** final 30 days per series; the first 14 days are used for the comparable forecast evaluation.

The notebook also includes a basic Rossmann input branch (`Store`, `Date`, `Sales`) if that dataset is used instead. The primary configuration follows the Store Item Demand data from Day 9.

## Architecture

### Encoder

An LSTM encoder reads the 30-day historical sequence and returns:
- The full sequence of hidden states.
- The final hidden state.
- The final cell state.

### Decoder

The encoder's final hidden state is repeated across the 14 forecast positions. A decoder LSTM initialized with the encoder states produces a sequence of decoder outputs.

### Temporal attention

Keras `Attention` compares decoder outputs with all encoder hidden states. For each future time step, it generates a context vector that summarizes the relevant parts of the 30-day history.

The context vector is concatenated with the decoder output and passed through a time-distributed Dense layer to produce one sales prediction per forecast day.

**Input shape:** `(batch, 30, 1)`  
**Output shape:** `(batch, 14, 1)`

## Day 9 baseline comparison

A Day 9-style single-step LSTM is trained in the notebook using:
- 30-day look-back
- LSTM with 32 units
- Dropout of 0.2
- Dense output layer
- Adam optimizer and MSE loss

At evaluation time, the single-step model is run recursively: each predicted day is appended to the input window to forecast the next day. Both models are evaluated on the same held-out 14-day periods.

## Evaluation

The notebook calculates:
- **MAE:** mean absolute error in sales units.
- **RMSE:** root mean squared error in sales units.
- **MAPE:** percentage error over non-zero actual values.
- **Horizon-wise MAE:** separate error for each day ahead.

It also plots:
- Training and validation loss.
- Actual versus predicted demand over a 14-day example.
- Attention weights across the 30 historical input days and 14 forecast steps.

No metric values are hard-coded in this README. Run the notebook and fill in the table below with the generated results.

| Model | MAE | RMSE | MAPE (non-zero actuals) |
|---|---:|---:|---:|
| Day 9 single-step LSTM (recursive) | _Run notebook_ | _Run notebook_ | _Run notebook_ |
| Day 10 Seq2Seq + Attention | _Run notebook_ | _Run notebook_ | _Run notebook_ |

The model should be considered an improvement only if the measured errors support that conclusion. Compare horizon-wise performance as well as the overall metrics.

## Attention analysis

The attention visualization is a matrix with:
- Rows representing forecast days 1–14.
- Columns representing historical days in the 30-day encoder window.
- Cell values representing the attention weight assigned to each historical position.

The visualization helps inspect which parts of the recent history the decoder uses at different forecast horizons. Attention weights are descriptive model diagnostics; they should not be interpreted as proof of causation.

## Multi-week inference and artifacts

The trained Seq2Seq model predicts future test dates in consecutive 14-day blocks. For each block, predictions are appended to the history before the next block is generated. The final block is trimmed to match the number of requested test dates.

Generated artifacts:

```text
models/
├── demand_seq2seq_attention.keras
└── demand_lstm_day09_baseline.keras

data/reports/
├── day10_model_comparison.csv
├── day10_horizon_metrics.csv
└── day10_per_series_metrics.csv

submission_seq2seq_attention.csv
```

The saved attention model is also accompanied in the notebook by an attention-visualization model built from the trained network.

## Challenges and design considerations

1. **Multi-step error accumulation:** recursive one-step forecasts can accumulate errors as predictions are fed back into the next input window.
2. **Sequence-to-sequence learning:** the decoder must learn the full future sequence rather than a single next-day value.
3. **Attention dimensions:** attention aligns each decoder time step with the encoder's complete history; the attention matrix therefore has 14 forecast rows and 30 history columns.
4. **Scaling:** one global min-max scaler is fitted on the pre-holdout training values and reused for all series and predictions.
5. **Compute and memory:** sliding windows can create a large training set. A seven-day stride and a configurable maximum number of windows keep the lab run manageable.
6. **Limited features:** this version uses sales history only. Calendar effects, promotions, holidays, prices, and store metadata are not included.
7. **Evaluation scope:** the comparable evaluation horizon is 14 days, even though the holdout contains 30 days. The remaining holdout days are not included in the headline metrics.

## How to run

1. Place the dataset in `data/store_item_demand/` with `train.csv` and `test.csv`.
2. Open `src/day10_seq2seq_attention_forecasting.ipynb` (or the downloaded notebook location) in Jupyter or VS Code.
3. Activate the TensorFlow environment used for the RetailMind forecasting work.
4. Run all cells in order.
5. Review the overall and horizon-wise metrics, plus the attention heatmap.
6. Copy the actual metric values into the evaluation table above.
7. Inspect the generated Kaggle submission before uploading it.

If the notebook is running slowly, reduce `MAX_TRAIN_WINDOWS`, `EPOCHS`, or `BATCH_SIZE`. Keep the evaluation split and forecast horizon unchanged when comparing model variants.

## Suggested Git commit

```bash
git add src/day10_seq2seq_attention_forecasting.ipynb README_Day10.md
git commit -m "Day 10: Add Seq2Seq attention demand forecasting"
git push
```

Do not commit the raw Kaggle dataset. Follow the repository's existing `.gitignore` rules for data files and generated model weights.

## Conclusion

Day 10 introduces multi-horizon demand forecasting to RetailMind AI. An LSTM encoder–decoder with temporal attention predicts two weeks of demand from a 30-day history, while a recursively evaluated Day 9-style LSTM provides the baseline. The experiment reports overall and horizon-specific errors, visualizes attention weights, saves the Keras model, and generates multi-week test forecasts for procurement planning.
# RetailMind AI — Day 12: Transformers & BERT

## Overview

This increment upgrades RetailMind AI's text-intelligence capability using a pretrained **BERT (`bert-base-uncased`)** model from Hugging Face Transformers.

The notebook covers two NLP tasks:

1. **Amazon review sentiment classification** — fine-tune BERT to classify reviews as positive or negative.
2. **Support-ticket routing** — derive a small set of ticket categories from review text, then fine-tune the same BERT backbone for multi-class routing.

It also builds TF-IDF + Logistic Regression baselines so the transformer can be evaluated against a lighter classical approach.

## Dataset

- **Source:** Hugging Face `fancyzhx/amazon_polarity` (Amazon Polarity).
- **Fields used:** `title`, `content`, and `label`.
- **Target:** `0 = negative`, `1 = positive`.
- **Sampling:** the notebook uses a reproducible subset (12,000 training examples and 3,000 test examples by default) to keep the exercise manageable in a 60-minute lab.
- **Text preparation:** title and review content are concatenated into one text field.

The full Amazon Polarity dataset is substantially larger. Increase the sample sizes only when compute resources allow.

## Implementation

### 1. Classical sentiment baseline

A scikit-learn pipeline applies:
- TF-IDF vectorization (unigrams and bigrams)
- Logistic Regression

This establishes a reference score on the same sampled train/test data.

### 2. BERT sentiment classifier

- Load `bert-base-uncased` tokenizer.
- Tokenize review text with truncation and a maximum sequence length of 128.
- Convert token IDs and attention masks into PyTorch datasets.
- Fine-tune `AutoModelForSequenceClassification` with two output classes.
- Evaluate accuracy, precision, recall, F1-score, and the confusion matrix.

### 3. Support-ticket routing

Amazon Polarity does not contain support-ticket categories. To demonstrate the routing workflow, the notebook creates **weak labels** using keyword rules for four categories:

| Category | Example matching terms |
|---|---|
| `delivery` | shipping, delivery, arrived, late, tracking |
| `billing` | price, charged, payment, refund, cost |
| `product_quality` | broken, defective, damaged, faulty, quality |
| `account_support` | account, login, password, customer service |

Rows that do not match any category are excluded. A TF-IDF baseline and a multi-class BERT model are trained on the resulting data.

**Data-quality warning:** these are heuristic labels, not verified support-ticket annotations. Their evaluation scores measure agreement with the keyword-derived labels and should not be presented as real-world ticket-routing accuracy. A production system should use actual support tickets with human-reviewed labels.

### 4. Model persistence and pipeline integration

The notebook saves model/tokenizer artifacts under:

```text
artifacts/day12/
├── bert_sentiment/
└── bert_ticket_router/
```

It also exports:

```text
artifacts/day12/day12_model_comparison.csv
```

The `predict_sentiment()` function demonstrates replacing the previous classical sentiment predictor with BERT inference. The ticket model is exposed through `predict_ticket_category()`.

For a deployed service, load each model once at application startup rather than reloading it for every request. Keep the tokenizer, model, label mapping, and preprocessing logic versioned together.

## Evaluation

The notebook reports:
- Accuracy for TF-IDF and BERT sentiment models.
- Per-class precision, recall, and F1 for sentiment.
- Confusion matrix for sentiment.
- Accuracy and per-class precision, recall, and F1 for ticket routing.
- A CSV comparison of both models across both tasks.

**Results are generated when the notebook is executed.** No metric values are hard-coded in this README because they depend on the sampled data, package versions, hardware, and training run. Add your actual results below after execution.

| Task | Model | Accuracy |
|---|---|---:|
| Review sentiment | TF-IDF + Logistic Regression | _Fill after run_ |
| Review sentiment | Fine-tuned BERT | _Fill after run_ |
| Ticket routing | TF-IDF + Logistic Regression | _Fill after run_ |
| Ticket routing | Fine-tuned BERT | _Fill after run_ |

Compare the measured BERT lift with the added training/inference cost. BERT is justified only if the quality improvement is meaningful for the product's requirements.

## Challenges and considerations

1. **Dataset size:** the complete Amazon Polarity corpus is large, so a subset is used for a time-boxed lab.
2. **Compute requirements:** fine-tuning BERT is much heavier than fitting TF-IDF. A CUDA GPU is recommended; CPU execution may take significantly longer.
3. **Sequence length:** limiting inputs to 128 tokens controls memory and training time, but truncates some long reviews.
4. **Synthetic ticket labels:** review text is not the same as real support-ticket text. Keyword-derived labels are suitable for demonstrating the workflow, not production validation.
5. **Class imbalance:** category counts and per-class metrics should be checked. Accuracy alone can hide poor performance on minority classes.
6. **Deployment cost:** model size, latency, memory, and throughput should be measured before replacing a lightweight model in a production service.

## How to run

1. Open `day12_transformers_bert_text_intelligence.ipynb` in Jupyter or VS Code.
2. Use a Python environment with PyTorch installed.
3. Install the required packages if needed:

```bash
pip install -U transformers datasets accelerate scikit-learn pandas numpy
```

4. Run the notebook from top to bottom.
5. Review the metrics and weak-label examples.
6. Copy the actual metric values into the evaluation table above.
7. Integrate the saved sentiment model into the existing RetailMind sentiment module.

## Suggested Git commit

```bash
git add day12_transformers_bert_text_intelligence.ipynb README_Day12.md
git commit -m "Add Day 12 BERT sentiment and ticket routing"
git push
```

## Conclusion

Day 12 adds a transformer-based text classification path to RetailMind AI. BERT is fine-tuned for review sentiment and reused as the backbone for a support-ticket routing demonstration. The notebook benchmarks both tasks against TF-IDF and persists the trained artifacts for integration. The final model choice should be based on measured quality gains, verified ticket labels, and deployment cost—not accuracy alone.
# Day 13 — NLP & Classical Text Processing
## RetailMind AI Increment

## Scenario

RetailMind AI is an end-to-end retail AI platform built incrementally, one module per day.

Each daily increment adds a new AI/ML capability to the same RetailMind codebase.

For **Day 13**, the goal is to add a **review analytics and sentiment analysis module** using classical Natural Language Processing techniques.

Customer reviews contain valuable information about:

- Customer satisfaction
- Product quality
- Product issues
- Delivery experience
- Common complaints
- Positive product features
- Overall customer sentiment

This increment converts raw customer review text into useful sentiment signals.

---

# Objective

Build a classical NLP pipeline that processes Amazon product reviews and predicts whether a review has **positive or negative sentiment**.

The complete workflow is:

```text
Amazon Reviews
      ↓
Text Cleaning
      ↓
Lowercasing
      ↓
Noise Removal
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
TF-IDF Vectorization
      ↓
Classical ML Classifier
      ↓
Sentiment Prediction
      ↓
Model Evaluation
      ↓
Informative Terms
      ↓
Persist Model
```

This model acts as the **classical sentiment baseline** for RetailMind.

Later increments can replace or compare this model with more advanced NLP models such as BERT.

---

# Dataset

The recommended dataset is:

## Amazon Product Reviews

Dataset:

```text
amazon_polarity
```

The dataset contains Amazon customer reviews labelled by sentiment.

A Kaggle Amazon reviews dataset may also be used.

Example:

```text
Review:
"This product is excellent and works perfectly."

Sentiment:
Positive
```

Another example:

```text
Review:
"Very poor quality. It stopped working after two days."

Sentiment:
Negative
```

---

# Dataset Features

Depending on the version of the dataset, columns may include:

```text
title
content
label
```

For this project, the main inputs are:

```text
content → customer review
label   → sentiment
```

The notebook renames these internally as:

```text
review
sentiment
```

If the dataset uses different column names, update:

```python
TEXT_COLUMN = "content"
LABEL_COLUMN = "label"
```

---

# Libraries Used

The Day 13 increment focuses on:

```text
NLTK
scikit-learn
```

The notebook also uses:

```text
pandas
re
pickle
```

for dataset handling, text cleaning, and model persistence.

---

# 1. Text Cleaning

Raw customer reviews contain unnecessary information such as:

```text
URLs
HTML tags
punctuation
numbers
special characters
uppercase/lowercase variations
```

Example raw review:

```text
"This Product is AMAZING!!! I loved it :) https://example.com"
```

After cleaning:

```text
product amazing loved
```

The cleaning pipeline performs:

```text
Raw Review
    ↓
Convert to lowercase
    ↓
Remove URLs
    ↓
Remove HTML
    ↓
Remove numbers
    ↓
Remove punctuation
    ↓
Tokenize words
    ↓
Remove stopwords
    ↓
Clean Review
```

---

# 2. Lowercasing

All text is converted to lowercase.

Example:

```text
"This PRODUCT Is Amazing"
```

becomes:

```text
"this product is amazing"
```

This prevents words such as:

```text
Product
PRODUCT
product
```

from being treated as different words.

---

# 3. Noise Removal

Regular expressions are used to remove unwanted characters.

For example:

```text
"This product is AMAZING!!! 100% recommended."
```

may become:

```text
this product is amazing recommended
```

This reduces unnecessary variation in the text.

---

# 4. Tokenization

Tokenization splits a sentence into individual words or tokens.

Example:

```text
"This product is really good"
```

becomes:

```text
["This", "product", "is", "really", "good"]
```

NLTK tokenization is used for this step.

---

# 5. Stopword Removal

Stopwords are common words that often provide little sentiment information.

Examples:

```text
the
is
a
an
and
of
to
in
```

Example:

```text
"this product is really good"
```

After stopword removal:

```text
"product really good"
```

This allows the model to focus more on meaningful terms.

---

# 6. TF-IDF Vectorization

Machine-learning models cannot directly understand text.

The cleaned reviews therefore need to be converted into numerical features.

This project uses:

```text
TF-IDF
```

TF-IDF stands for:

```text
Term Frequency - Inverse Document Frequency
```

It gives more importance to useful terms while reducing the importance of extremely common words.

Example:

```text
Review 1:
"excellent product"

Review 2:
"poor product"
```

TF-IDF converts these reviews into numerical feature vectors that can be provided to the machine-learning model.

---

# 7. N-Grams

The TF-IDF vectorizer uses:

```python
ngram_range=(1, 2)
```

This means the model can learn:

### Unigrams

Single words:

```text
excellent
broken
quality
waste
```

### Bigrams

Two-word combinations:

```text
very good
poor quality
highly recommend
waste money
```

Bigrams can provide additional sentiment information compared with individual words alone.

---

# 8. Train/Test Split

The dataset is divided into:

```text
Training Data
Testing Data
```

Example split:

```text
80% → Training
20% → Testing
```

The training data is used to learn the sentiment model.

The testing data is kept separate and used to evaluate how well the trained model performs on unseen reviews.

A fixed:

```python
random_state=42
```

is used to make the split reproducible.

Stratification is also used to preserve the sentiment class distribution.

---

# 9. Sentiment Classification Model

The classical baseline classifier used in this project is:

```text
Logistic Regression
```

The complete machine-learning pipeline is:

```text
Clean Review
      ↓
TF-IDF
      ↓
Logistic Regression
      ↓
Positive / Negative
```

Logistic Regression works well as a baseline for sparse TF-IDF text features.

It also provides model coefficients that can be inspected to understand which terms contribute strongly to positive or negative predictions.

---

# 10. Scikit-Learn Pipeline

TF-IDF and Logistic Regression are combined using a scikit-learn Pipeline.

Conceptually:

```text
Review
   ↓
TfidfVectorizer
   ↓
LogisticRegression
   ↓
Prediction
```

Using a Pipeline keeps preprocessing and prediction together.

It also helps prevent incorrect preprocessing of test data.

The TF-IDF vectorizer is fitted using training data.

The test data is only transformed using the vocabulary learned from the training data.

---

# 11. Model Training

The model learns relationships between TF-IDF features and sentiment labels.

For example, the model may learn that words such as:

```text
excellent
perfect
amazing
love
great
```

are more strongly associated with positive sentiment.

Words such as:

```text
broken
waste
poor
worst
disappointed
```

may be more strongly associated with negative sentiment.

The actual terms depend on the training dataset.

---

# 12. Model Evaluation

The trained sentiment model is evaluated using multiple classification metrics.

The project calculates:

```text
Accuracy
Precision
Recall
F1 Score
Classification Report
Confusion Matrix
```

---

# 13. Accuracy

Accuracy measures the proportion of predictions that were correct.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

Accuracy is useful for an overall view of model performance.

---

# 14. Precision

Precision measures:

```text
Out of the reviews predicted as a class,
how many predictions were correct?
```

For the positive class:

```text
Precision = TP / (TP + FP)
```

High precision means the model produces relatively few false-positive predictions.

---

# 15. Recall

Recall measures:

```text
Out of the reviews that actually belong to a class,
how many did the model correctly identify?
```

For the positive class:

```text
Recall = TP / (TP + FN)
```

High recall means the model misses relatively few positive reviews.

---

# 16. F1 Score

F1 combines Precision and Recall.

Formula:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is useful when both precision and recall matter.

---

# 17. Confusion Matrix

The confusion matrix shows:

```text
True Negative
False Positive
False Negative
True Positive
```

It provides more detail than accuracy alone.

It helps identify whether the model is making more errors on positive or negative sentiment.

---

# 18. Informative Terms

One important goal of the RetailMind review analytics module is not just predicting sentiment.

It should also help understand **why customers are happy or unhappy**.

Logistic Regression assigns coefficients to TF-IDF features.

Terms with large positive coefficients are strongly associated with positive sentiment.

Terms with large negative coefficients are strongly associated with negative sentiment.

Example positive terms might include:

```text
excellent
perfect
amazing
great
love
highly recommend
```

Example negative terms might include:

```text
waste
broken
poor
disappointed
worst
poor quality
```

These terms can provide useful product-level insights.

However, model coefficients show **association rather than causation**. Important findings should be checked against the original customer reviews.

---

# 19. Predicting New Reviews

After training, RetailMind can classify previously unseen reviews.

Example:

```text
Input:
"This product is excellent and works perfectly"

Prediction:
Positive
```

Another example:

```text
Input:
"Poor quality and completely useless"

Prediction:
Negative
```

This capability can later be integrated into a RetailMind API or dashboard.

---

# 20. Model Persistence

Retraining the sentiment model every time RetailMind starts would be inefficient.

Therefore, the trained pipeline is persisted.

The saved artifact is:

```text
review_sentiment_model.pkl
```

The persisted object contains:

```text
TF-IDF Vectorizer
+
Logistic Regression Model
```

The saved pipeline can later be loaded and used to predict new reviews without retraining.

---

# 21. RetailMind Integration

The Day 13 increment adds a new review analytics capability to RetailMind.

Conceptually:

```text
RetailMind AI
    |
    |-- Previous AI/ML modules
    |
    |-- Review Analytics
            |
            |-- Text Cleaning
            |-- Tokenization
            |-- Stopword Removal
            |-- TF-IDF
            |-- Sentiment Classifier
            |-- Product Insights
            |-- Persisted Model
```

This module remains part of the RetailMind codebase and can be reused by future increments.

---

# 22. Future BERT Upgrade

Day 13 establishes the **classical NLP baseline**.

A later increment can replace:

```text
TF-IDF
+
Logistic Regression
```

with a transformer model such as:

```text
BERT
```

The classical model remains useful because it provides:

- A performance baseline
- Faster training
- Lower computational requirements
- Easier interpretation
- Informative feature coefficients

The future transformer model can then be compared against this baseline.

---

# Project Structure

A simple project structure is:

```text
RetailMind/
│
├── amazon_reviews.csv
│
├── day13_nlp_classical_text_processing.ipynb
│
├── review_sentiment_model.pkl
│
└── README.md
```

As RetailMind grows, the model and review-processing functions can be moved into dedicated modules.

---

# Day 13 Workflow

```text
1. Load Amazon reviews
        ↓
2. Inspect dataset
        ↓
3. Handle missing reviews
        ↓
4. Lowercase text
        ↓
5. Remove noise
        ↓
6. Tokenize
        ↓
7. Remove stopwords
        ↓
8. Create cleaned reviews
        ↓
9. Train/Test split
        ↓
10. TF-IDF vectorization
        ↓
11. Train Logistic Regression
        ↓
12. Predict test sentiment
        ↓
13. Evaluate model
        ↓
14. Extract informative terms
        ↓
15. Predict new reviews
        ↓
16. Save trained pipeline
        ↓
17. Reload and test model
```

---

# Day 13 Deliverables

The completed increment contains:

- Amazon review dataset loading
- Text cleaning
- Lowercasing
- URL removal
- HTML removal
- Punctuation and noise removal
- NLTK tokenization
- NLTK stopword removal
- TF-IDF feature extraction
- Unigrams
- Bigrams
- Train/test splitting
- Logistic Regression sentiment classifier
- Accuracy
- Precision
- Recall
- F1 score
- Classification report
- Confusion matrix
- Informative positive terms
- Informative negative terms
- New-review sentiment prediction
- Saved review sentiment model
- Reloadable review-processing pipeline

---

# Conclusion

In Day 13 of RetailMind AI, a complete classical NLP sentiment-analysis baseline was added to the platform.

Raw Amazon customer reviews are transformed from unstructured text into numerical TF-IDF features and classified using Logistic Regression.

The module not only predicts whether reviews are positive or negative but also exposes informative terms that can help identify customer satisfaction signals and recurring product issues.

The trained pipeline is persisted so other RetailMind components can reuse it.

This classical NLP baseline also prepares RetailMind for a future upgrade to transformer-based sentiment analysis using BERT.
