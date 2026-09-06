# Model Card: Churn Prediction Model

## Overview
This model predicts whether a telecom customer is likely to churn (cancel their service), based on the Telco Customer Churn dataset. It's an XGBoost classifier trained on customer demographics, account details, and service usage patterns.

## Intended Use
- Flagging customers at high risk of churning, so a retention team can proactively reach out.
- **Not intended** for automated decisions (e.g. automatically approving discounts) without human review.
- Trained specifically on telecom industry data; not validated for other industries without retraining.

## Performance
On a held-out 20% test set:

| Metric | Class 0 (Stable) | Class 1 (Churn) |
|---|---|---|
| Precision | 0.89 | 0.49 |
| Recall | 0.72 | 0.75 |
| F1-score | 0.79 | 0.59 |

Overall accuracy: ~73%.

## Why Recall Was Prioritized Over Precision
For churn prediction, missing a customer who's about to leave (false negative) is usually costlier than a false alarm (false positive) — a false alarm just means an unnecessary retention offer, while a missed churner is a lost customer. The model is tuned to favor catching more true churners, at the cost of more false alarms.

## Limitations
- Trained on a single, static dataset snapshot; customer behavior changes over time, so the model should be retrained periodically on fresh data.
- ~49% precision on the churn class means roughly half of "high risk" flags will be false alarms — this model should support human decision-making, not replace it.
- Not validated on customer populations outside this dataset's demographic and geographic scope.

## When to Get a Human Involved
Any high-stakes action (contract changes, service cancellations, large discounts) based on this model's output should be reviewed by a person before being acted on.