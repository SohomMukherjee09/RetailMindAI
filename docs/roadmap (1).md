# Model Card --- RetailMind AI Churn Classifier

## Model Identity

  -----------------------------------------------------------------------
  Field                               Current record
  ----------------------------------- -----------------------------------
  Project                             RetailMind AI

  Use case                            Customer churn classification

  Model family                        Logistic Regression baseline; tree
                                      and SVM candidates under evaluation

  Target                              `Churn` (`Yes → 1`, `No → 0`)

  Current champion                    Not yet confirmed

  Artifact                            Day 1 notebook reports
                                      `models/logistic_reg.joblib`
  -----------------------------------------------------------------------

## Intended Use

The model is intended as a development-stage decision-support component
for prioritising customers who may be at risk of churn. A future
retention workflow could use a risk score to decide which accounts merit
human review.

## Out-of-Scope Use

-   Treating a prediction as certainty that a customer will churn.
-   Automatically denying, pricing, restricting or changing service
    based solely on a prediction.
-   Using the public benchmark as evidence of performance on a different
    organisation's customer base.
-   Inferring causal reasons for churn from model associations.
-   Making consequential customer decisions without appropriate
    validation and human oversight.

## Training Data

The initial model uses the public Kaggle Telco Customer Churn dataset.
The exact dataset version, row count, feature schema and split
configuration should be captured in each experiment report.

## Features and Processing

-   Customer identifier excluded from model features.
-   Target mapped to integer classes.
-   `TotalCharges` explicitly converted to numeric.
-   Numerical features: median imputation and standardisation.
-   Categorical features: most-frequent imputation and one-hot encoding.
-   Transformations and classifier packaged in a scikit-learn pipeline.

## Performance

A ROC-AUC value of 0.84 was reported during Day 1 development, but it
requires reproduction on a fixed, documented split before being treated
as an official benchmark. A confusion matrix was also shared during Day
2 development; its model/run provenance remains unconfirmed. See the
experiment reports.

## Limitations and Risks

-   Dataset domain and population may not match a real retail or telecom
    deployment.
-   Historical patterns can change over time.
-   Class imbalance can make accuracy misleading.
-   False negatives may represent missed retention opportunities; false
    positives may waste outreach capacity.
-   Probability estimates may not be calibrated.
-   Feature associations are not causal explanations.
-   Data quality and schema drift can degrade predictions.
-   No production monitoring, API or deployment is claimed at this
    stage.

## Human Oversight

A future business workflow should present predictions as prioritisation
signals. Retention staff should consider context, customer preferences
and policy before acting. The model should be periodically evaluated
against fresh labelled outcomes.

## Maintenance

Before promoting any model: 1. Validate data schema and quality. 2.
Reproduce evaluation on a fixed test set. 3. Compare against the
existing champion and baseline. 4. Review errors and subgroup
performance where appropriate and lawful. 5. Record model version,
parameters, dependencies and artifact checksum. 6. Define monitoring and
rollback procedures.
