# RetailMind AI --- Dataset and Data Dictionary

## 1. Dataset

**Dataset:** Telco Customer Churn\
**Source:** Kaggle dataset `blastchar/telco-customer-churn`\
**File expected by the project:** `data/Telco-Customer-Churn.csv`

The dataset contains customer account, service and billing attributes
with a churn label. It is used here as a public benchmark dataset for
developing the first churn-classification workflow.

Source link:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 2. Target Definition

  Field     Meaning
  --------- ------------------------------
  `Churn`   Whether the customer churned
  `No`      Encoded as `0`
  `Yes`     Encoded as `1`

In this project, churn (`1`) is the positive class. Precision, recall,
F1 and ROC-AUC must consistently treat class `1` as the positive class.

## 3. Common Dataset Fields

The source dataset commonly includes the following fields. Confirm the
actual CSV schema with `df.columns` before relying on this table.

  --------------------------------------------------------------------------
  Field                Typical type      Description       Modeling
                                                           treatment
  -------------------- ----------------- ----------------- -----------------
  `customerID`         Identifier / text Customer          Exclude from
                                         reference         model features
                                         identifier        

  `gender`             Categorical       Recorded customer One-hot encode
                                         gender category   

  `SeniorCitizen`      Binary numeric    Indicator for     Numeric feature,
                                         senior-citizen    subject to schema
                                         status            check

  `Partner`            Categorical       Whether the       One-hot encode
                                         customer has a    
                                         partner           

  `Dependents`         Categorical       Whether the       One-hot encode
                                         customer has      
                                         dependents        

  `tenure`             Numeric           Customer tenure   Impute and scale
                                         in months         

  `PhoneService`       Categorical       Whether phone     One-hot encode
                                         service is        
                                         subscribed        

  `MultipleLines`      Categorical       Multiple-lines    One-hot encode
                                         service status    

  `InternetService`    Categorical       Internet service  One-hot encode
                                         type              

  `OnlineSecurity`     Categorical       Online security   One-hot encode
                                         service status    

  `OnlineBackup`       Categorical       Online backup     One-hot encode
                                         service status    

  `DeviceProtection`   Categorical       Device protection One-hot encode
                                         service status    

  `TechSupport`        Categorical       Tech support      One-hot encode
                                         service status    

  `StreamingTV`        Categorical       Streaming TV      One-hot encode
                                         service status    

  `StreamingMovies`    Categorical       Streaming movies  One-hot encode
                                         service status    

  `Contract`           Categorical       Contract term     One-hot encode

  `PaperlessBilling`   Categorical       Paperless billing One-hot encode
                                         indicator         

  `PaymentMethod`      Categorical       Payment method    One-hot encode

  `MonthlyCharges`     Numeric           Monthly customer  Impute and scale
                                         charges           

  `TotalCharges`       Numeric after     Total charges     Convert to
                       conversion        accumulated       numeric; impute
                                                           and scale

  `Churn`              Target            Churn label       Map `No → 0`,
                                                           `Yes → 1`;
                                                           exclude from X
  --------------------------------------------------------------------------

## 4. Data Quality Notes

### `TotalCharges`

In the commonly used CSV, `TotalCharges` may be stored as text because
some rows contain blank values. Convert it explicitly:

``` python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Unparseable values become missing values (`NaN`). The numerical
pipeline's median imputer can then handle them, provided the column is
included in the numerical feature list.

### Missing values

A missing-value report should be generated after type conversion,
because conversion may reveal missing values that were previously
represented as blank strings.

### Categorical values

Inspect unique values and category spelling. Values such as
`"No internet service"` are categories, not necessarily missing values.
Do not replace them with null without a domain reason.

### Class balance

Record `df["Churn"].value_counts()` and its normalised proportions for
the exact dataset version used. Do not assume a class distribution
without measuring it.

## 5. Feature and Target Separation

Recommended explicit form:

``` python
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]
```

The notebook's positional form `df.iloc[:, 1:-1]` also excludes the
first column and final target column if the source column order is
unchanged. Explicit column names are more robust to schema reordering.

## 6. Data Governance

The public benchmark is not a substitute for an organisation's own
validated customer data. Before applying the workflow to real customer
records, confirm permitted use, retention, access controls, privacy
obligations and whether any fields require exclusion or special
handling.
