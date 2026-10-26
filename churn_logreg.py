"""
Customer Churn Prediction with Logistic Regression
Dataset: Telco Customer Churn (IBM sample data, available on Kaggle)
Put the CSV at data/telco_churn.csv, then run: python churn_logreg.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             RocCurveDisplay, ConfusionMatrixDisplay,
                             precision_recall_curve, f1_score)

DATA_PATH = "data/telco_churn.csv"
SEED = 42

# ---------- 1. Load & clean ----------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])
# TotalCharges has blank strings for brand-new customers (tenure = 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(f"Rows: {len(df)} | Churn rate: {df['Churn'].mean():.1%}")

X, y = df.drop(columns="Churn"), df["Churn"]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------- 2. Split (stratified because classes are imbalanced) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---------- 3. Pipeline (preprocessing inside => no data leakage) ----------
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])
pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

# Baseline: always predict majority class. Your model must beat this.
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"Baseline accuracy: {base.score(X_test, y_test):.3f}")

# ---------- 4. Tune regularization strength C with cross-validation ----------
grid = GridSearchCV(
    pipe,
    param_grid={"clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ["l1", "l2"],
                "clf__solver": ["liblinear"]},
    scoring="roc_auc",
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    n_jobs=-1,
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print(f"Best params: {grid.best_params_} | CV AUC: {grid.best_score_:.3f}")

# ---------- 5. Evaluate on held-out test set ----------
proba = best.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print("\nDefault threshold (0.5):")
print(classification_report(y_test, (proba >= 0.5).astype(int), target_names=["Stay", "Churn"]))

# Threshold tuning: pick the threshold that maximizes F1 on the churn class
prec, rec, thr = precision_recall_curve(y_test, proba)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_thr = thr[np.argmax(f1s)]
print(f"F1-optimal threshold: {best_thr:.2f} (F1 = {f1s.max():.3f})")
pred = (proba >= best_thr).astype(int)

# ---------- 6. Plots ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
RocCurveDisplay.from_predictions(y_test, proba, ax=ax[0])
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set_title("ROC curve")
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stay", "Churn"]).plot(ax=ax[1])
ax[1].set_title(f"Confusion matrix (threshold={best_thr:.2f})")
plt.tight_layout()
plt.savefig("outputs/evaluation.png", dpi=150)

# ---------- 7. Interpret: coefficients -> odds ratios ----------
names = best.named_steps["pre"].get_feature_names_out()
coefs = best.named_steps["clf"].coef_[0]
imp = (pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
         .assign(abs_coef=lambda d: d.coef.abs())
         .sort_values("abs_coef", ascending=False).drop(columns="abs_coef"))
imp.to_csv("outputs/feature_effects.csv", index=False)
print("\nTop drivers of churn (odds ratio > 1 raises churn odds):")
print(imp.head(10).round(3).to_string(index=False))

top = imp.head(12).iloc[::-1]
plt.figure(figsize=(7, 5))
plt.barh(top.feature, top.coef, color=np.where(top.coef > 0, "tab:red", "tab:green"))
plt.title("Top coefficients (red = more churn, green = less)")
plt.tight_layout()
plt.savefig("outputs/coefficients.png", dpi=150)
