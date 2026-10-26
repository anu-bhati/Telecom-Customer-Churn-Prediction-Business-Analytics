"""
Streamlit app: Customer Churn Predictor (Logistic Regression)
Run: streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path("data/telco_churn.csv")
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------- data + model
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["customerID"], errors="ignore").copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_resource(show_spinner="Training model (one-time)...")
def train(df: pd.DataFrame):
    X, y = df.drop(columns="Churn"), df["Churn"]
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    grid = GridSearchCV(
        pipe,
        {"clf__C": [0.01, 0.1, 1, 10], "clf__solver": ["liblinear"]},
        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1,
    ).fit(X_tr, y_tr)

    model = grid.best_estimator_
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).score(X_te, y_te)
    return {
        "model": model, "X_test": X_te, "y_test": y_te,
        "proba": model.predict_proba(X_te)[:, 1],
        "cv_auc": grid.best_score_, "best_C": grid.best_params_["clf__C"],
        "baseline": baseline, "cat_cols": cat_cols, "columns": list(X.columns),
    }


def feature_effects(model) -> pd.DataFrame:
    names = model.named_steps["pre"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coef = model.named_steps["clf"].coef_[0]
    return pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})


# ---------------------------------------------------------------- load data
st.sidebar.title("📉 Churn Predictor")
st.sidebar.caption("Logistic regression on the Telco Customer Churn dataset")

if DATA_PATH.exists():
    raw = pd.read_csv(DATA_PATH)
else:
    up = st.sidebar.file_uploader("Upload telco_churn.csv", type="csv")
    if up is None:
        st.title("Customer Churn Predictor")
        st.info("Place the Kaggle *Telco Customer Churn* CSV at `data/telco_churn.csv` "
                "or upload it in the sidebar to begin.")
        st.stop()
    raw = pd.read_csv(up)

df = clean(raw)
art = train(df)
model, y_test, proba = art["model"], art["y_test"], art["proba"]

st.sidebar.markdown("### Decision threshold")
threshold = st.sidebar.slider("Flag customer as churn-risk if P(churn) ≥", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Lower = catch more churners (more false alarms). "
                   "Higher = fewer, more certain flags.")

st.title("Customer Churn Predictor")
tab_pred, tab_perf, tab_drivers, tab_biz = st.tabs(
    ["🔮 Predict a customer", "📊 Model performance", "🧠 What drives churn", "💰 Business impact"])

# ---------------------------------------------------------------- tab 1: predict
with tab_pred:
    st.subheader("Describe a customer")
    X_all = df.drop(columns="Churn")
    inputs = {}
    key_cols = [c for c in ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]
                if c in art["columns"]]
    other_cols = [c for c in art["columns"] if c not in key_cols]

    def widget(col, container):
        if col in NUM_COLS:
            lo, hi, med = float(X_all[col].min()), float(X_all[col].max()), float(X_all[col].median())
            if col == "TotalCharges":
                return None  # derived below
            return container.slider(col, lo, hi, med)
        opts = sorted(X_all[col].astype(str).unique())
        default = X_all[col].astype(str).mode()[0]
        return container.selectbox(col, opts, index=opts.index(default))

    cols = st.columns(len(key_cols))
    for c, box in zip(key_cols, cols):
        inputs[c] = widget(c, box)
    with st.expander("More attributes (defaults to the most common value)"):
        grid = st.columns(3)
        for i, c in enumerate(other_cols):
            inputs[c] = widget(c, grid[i % 3])

    # TotalCharges is derived so the form can't contradict itself
    if "TotalCharges" in art["columns"]:
        inputs["TotalCharges"] = inputs["tenure"] * inputs["MonthlyCharges"]
    # keep numeric-looking binary columns (e.g. SeniorCitizen 0/1) as numbers
    for c in art["cat_cols"]:
        if pd.api.types.is_numeric_dtype(X_all[c]):
            inputs[c] = type(X_all[c].iloc[0])(inputs[c])

    row = pd.DataFrame([inputs])[art["columns"]]
    p = float(model.predict_proba(row)[0, 1])

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Churn probability", f"{p:.1%}")
        st.progress(min(p, 1.0))
        if p >= threshold:
            st.error(f"⚠️ High risk (≥ {threshold:.0%} threshold)")
        else:
            st.success(f"✅ Low risk (< {threshold:.0%} threshold)")
        st.caption(f"Average churn rate in data: {df['Churn'].mean():.1%}")
    with c2:
        st.markdown("**Why this score?** Contribution of each feature to the log-odds "
                    "(coefficient × scaled value). Positive pushes toward churn.")
        z = model.named_steps["pre"].transform(row)[0]
        names = [n.split("__", 1)[1] for n in model.named_steps["pre"].get_feature_names_out()]
        contrib = pd.Series(z * model.named_steps["clf"].coef_[0], index=names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        st.bar_chart(top.iloc[::-1])

# ---------------------------------------------------------------- tab 2: performance
with tab_perf:
    pred = (proba >= threshold).astype(int)
    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")
    m2.metric("CV ROC-AUC", f"{art['cv_auc']:.3f}")
    m3.metric("Churn recall", f"{rep['1']['recall']:.2f}")
    m4.metric("Churn precision", f"{rep['1']['precision']:.2f}")
    m5.metric("Baseline accuracy", f"{art['baseline']:.2f}",
              help="Accuracy of always predicting 'stays'. Accuracy alone is misleading here.")
    st.caption(f"Best regularization C = {art['best_C']}. Metrics on the {len(y_test)}-row held-out "
               f"test set at threshold {threshold:.2f} (move the sidebar slider to see the trade-off).")

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title("ROC curve")
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=[0, 1]),
                               display_labels=["Stay", "Churn"]).plot(ax=ax, colorbar=False)
        ax.set_title("Confusion matrix")
        st.pyplot(fig)

# ---------------------------------------------------------------- tab 3: drivers
with tab_drivers:
    eff = feature_effects(model)
    eff["abs"] = eff["coef"].abs()
    topn = st.slider("Features to show", 5, min(25, len(eff)), 12)
    top = eff.sort_values("abs", ascending=False).head(topn).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * topn + 1))
    ax.barh(top["feature"], top["coef"], color=np.where(top["coef"] > 0, "tab:red", "tab:green"))
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title("Red raises churn odds, green lowers them")
    st.pyplot(fig)
    st.markdown("**Reading odds ratios:** `odds_ratio = exp(coef)`. For numeric features it's per "
                "1 standard deviation; for categories it's relative to the dropped reference level.")
    st.dataframe(eff.sort_values("abs", ascending=False).drop(columns="abs").round(3),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 4: business
with tab_biz:
    st.markdown("Estimate the value of a retention campaign that contacts every flagged customer.")
    b1, b2, b3 = st.columns(3)
    value = b1.number_input("Value of a retained customer ($)", 0, 10000, 500, 50)
    cost = b2.number_input("Cost per retention offer ($)", 0, 1000, 30, 5)
    success = b3.slider("Offer success rate", 0.05, 1.0, 0.30, 0.05)

    ts = np.linspace(0.05, 0.95, 91)
    y_arr = np.asarray(y_test)
    nets = []
    for t in ts:
        flag = proba >= t
        tp = int((flag & (y_arr == 1)).sum())
        nets.append(tp * success * value - int(flag.sum()) * cost)
    nets = np.array(nets)
    best_t = float(ts[nets.argmax()])

    flag = proba >= threshold
    tp = int((flag & (y_arr == 1)).sum())
    net_now = tp * success * value - int(flag.sum()) * cost
    n1, n2, n3 = st.columns(3)
    n1.metric("Customers flagged", f"{int(flag.sum())} of {len(y_arr)}")
    n2.metric("Net value at your threshold", f"${net_now:,.0f}")
    n3.metric("Profit-maximizing threshold", f"{best_t:.2f}", f"${nets.max():,.0f}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ts, nets)
    ax.axvline(threshold, ls="--", color="gray", label="your threshold")
    ax.axvline(best_t, ls=":", color="tab:green", label="best threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net value on test set ($)")
    ax.legend()
    st.pyplot(fig)
    st.caption("Assumes retained = churner who accepts the offer; costs apply to everyone contacted. "
               "Numbers are illustrative: change the assumptions to match a real business.")
