# Amazon Delivery Speed Prediction

## Problem
Predicting whether a delivery will be fast or slow based on agent, weather, traffic, and order information. This is a binary classification problem.

## Data
The dataset consisted of 43,739 rows and 16 columns. Several feature engineering steps were applied — datetime columns were parsed to extract `day_of_week`, `is_weekend`, `month`, and `order_hour_block`. A binary target `is_fast_delivery` was created from the raw `Delivery_Time` column.

## Approach
A full production pipeline was built using sklearn's `ColumnTransformer` with three parallel branches:
- **Numeric columns** — `SimpleImputer` + `StandardScaler`
- **Low cardinality categoricals** — `OneHotEncoder`
- **High cardinality categoricals** — `TargetEncoder` (applied to `Category` only)

Hyperparameter tuning was performed using `HalvingRandomSearchCV`, tuning both model parameters and encoder smoothing simultaneously.

## Results
| | Score |
|---|---|
| Train | 0.8176 |
| Test | 0.8078 |

Best parameters: `learning_rate=0.1`, `max_depth=7`, `reg_lambda=3`, `reg_alpha=2`, `smooth=50.0`

## Explainability
SHAP analysis revealed that `Agent_Rating` was the strongest predictor of fast delivery. Product `Category` showed a strong consistent negative impact for certain categories. Traffic conditions and weather had moderate impact.

## How To Use
```python
import joblib
model = joblib.load('best_model.pkl')
predictions = model.predict(X_new)
```

---

Two corrections from your draft:
- You said "XGB Regressor" — it's a Classifier, not Regressor
- "pivot role" should be "pivotal role"

Otherwise solid foundation for a first README.