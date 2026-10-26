# Customer Churn Prediction with Logistic Regression

Predict which telecom customers will cancel, and explain *why*, using an interpretable logistic regression model.

## Setup
1. Download **Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) from Kaggle.
2. Save it as `data/telco_churn.csv`.
3. `pip install -r requirements.txt`
4. `python churn_logreg.py` (training script + saved plots)
5. `streamlit run app.py` (interactive app)

Outputs: `outputs/evaluation.png`, `outputs/coefficients.png`, `outputs/feature_effects.csv`.

## What the pipeline does
- Cleans data (blank `TotalCharges`, target encoding)
- Stratified train/test split (churn is imbalanced, ~27%)
- `ColumnTransformer` + `Pipeline` so scaling/encoding is fit on training data only (no leakage)
- Compares against a majority-class baseline
- Tunes regularization (L1/L2, C) with 5-fold stratified CV, optimizing ROC-AUC
- Evaluates with ROC-AUC, precision/recall/F1, confusion matrix
- Tunes the decision threshold for the business goal
- Interprets coefficients as odds ratios

## Results (fill in after you run it)
- Baseline accuracy: __
- Test ROC-AUC: __
- Churn recall / precision at tuned threshold: __ / __
- Top drivers: __

## Streamlit app (`app.py`)
- **Predict a customer:** enter attributes, get churn probability and a per-feature "why this score" chart
- **Model performance:** ROC-AUC, recall/precision, confusion matrix, all reacting to the threshold slider
- **What drives churn:** coefficients and odds ratios
- **Business impact:** set offer cost, customer value and success rate to find the profit-maximizing threshold
