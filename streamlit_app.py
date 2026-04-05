"""
Customer Churn Prediction Dashboard
================================================
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, f1_score, classification_report, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TeleChurnAI — Churn Prediction Dashboard",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
  h1 { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.2, 0.8, 0.64, 0.01)
n_estimators = st.sidebar.slider("GB n_estimators", 50, 500, 300, 50)

# ── Load & Preprocess ────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    train = pd.read_csv("churn-bigml-80.csv")
    test  = pd.read_csv("churn-bigml-20.csv")

    def preprocess(df):
        d = df.copy()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ["International plan", "Voice mail plan"]:
            d[col] = le.fit_transform(d[col])
        d = pd.get_dummies(d, columns=["State"], drop_first=True)
        d["Churn"] = d["Churn"].astype(int)
        return d

    train_p = preprocess(train)
    test_p  = preprocess(test)
    train_p, test_p = train_p.align(test_p, join="left", axis=1, fill_value=0)
    return train_p, test_p

train_p, test_p = load_and_preprocess()
FEAT_COLS = [c for c in train_p.columns if c != "Churn"]
X_train, y_train = train_p[FEAT_COLS], train_p["Churn"]
X_test,  y_test  = test_p[FEAT_COLS],  test_p["Churn"]

# ── Train Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(n_est):
    sw = compute_sample_weight("balanced", y_train)
    gb = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=0.05, max_depth=4, random_state=42
    )
    gb.fit(X_train, y_train, sample_weight=sw)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    return gb, rf

gb, rf = train_model(n_estimators)
gb_prob = gb.predict_proba(X_test)[:, 1]
gb_pred = (gb_prob >= threshold).astype(int)
rf_prob = rf.predict_proba(X_test)[:, 1]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📡 TeleChurnAI — Customer Churn Prediction Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("F1-Score (Churn)", f"{f1_score(y_test, gb_pred):.3f}", "Gradient Boosting")
col2.metric("ROC-AUC", f"{roc_auc_score(y_test, gb_prob):.3f}", "Gradient Boosting")
col3.metric("Churn Rate", f"{y_train.mean()*100:.1f}%", "Training set")
col4.metric("Threshold", f"{threshold:.2f}", "Tunable in sidebar")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Feature Importance", "🤖 Predict Customer", "📈 ROC Curve"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classification Report — Gradient Boosting")
        report = classification_report(y_test, gb_pred, target_names=["Retained", "Churned"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))
    with c2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, gb_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Retained", "Actual: Churned"],
            columns=["Pred: Retained", "Pred: Churned"]
        )
        st.dataframe(cm_df)

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Pred: Retained", "Pred: Churned"],
            y=["Actual: Retained", "Actual: Churned"],
            colorscale="YlOrRd",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
        ))
        fig_cm.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

with tab2:
    st.subheader("Top-15 Feature Importances — Random Forest")
    fi = (
        pd.Series(rf.feature_importances_, index=FEAT_COLS)
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    fi.columns = ["Feature", "Importance"]

    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"],
        y=fi["Feature"],
        orientation="h",
        marker=dict(color=fi["Importance"], colorscale="YlOrRd"),
    ))
    fig_fi.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with tab3:
    st.subheader("🔮 Predict Churn for a Single Customer")
    st.markdown("Adjust the sliders to simulate a customer profile:")

    c1, c2, c3 = st.columns(3)
    with c1:
        day_charge  = st.slider("Total Day Charge ($)", 0.0, 60.0, 35.0, 0.5)
        svc_calls   = st.slider("Customer Service Calls", 0, 9, 1)
        intl_plan   = st.selectbox("International Plan", ["No", "Yes"])
    with c2:
        eve_charge  = st.slider("Total Eve Charge ($)", 0.0, 30.0, 15.0, 0.5)
        intl_charge = st.slider("Total Intl Charge ($)", 0.0, 10.0, 2.5, 0.1)
        vmail_plan  = st.selectbox("Voice Mail Plan", ["No", "Yes"])
    with c3:
        night_charge = st.slider("Total Night Charge ($)", 0.0, 20.0, 9.0, 0.5)
        account_len  = st.slider("Account Length (days)", 1, 243, 101)
        vmail_msgs   = st.slider("Number Vmail Messages", 0, 51, 0)

    sample = X_test.iloc[[0]].copy()
    sample["Total day charge"]       = day_charge
    sample["Customer service calls"] = svc_calls
    sample["International plan"]     = 1 if intl_plan == "Yes" else 0
    sample["Total eve charge"]       = eve_charge
    sample["Total intl charge"]      = intl_charge
    sample["Voice mail plan"]        = 1 if vmail_plan == "Yes" else 0
    sample["Total night charge"]     = night_charge
    sample["Account length"]         = account_len
    sample["Number vmail messages"]  = vmail_msgs

    prob  = gb.predict_proba(sample)[0, 1]
    churn = prob >= threshold

    label = "⚠️ HIGH CHURN RISK" if churn else "✅ LOW CHURN RISK"
    st.markdown(f"### {label}")
    st.markdown(f"**Churn Probability: `{prob*100:.1f}%`** (threshold = {threshold})")
    st.progress(float(prob))

    if churn:
        st.error("**Recommended Action:** Assign to retention campaign. Offer personalised plan discount or dedicated support agent.")
    else:
        st.success("**Customer is stable.** No immediate retention action required.")

with tab4:
    st.subheader("ROC Curves — All Models")

    fig_roc = go.Figure()
    for name, prob, color in [
        ("Random Forest", rf_prob, "#34d399"),
        ("Gradient Boosting", gb_prob, "#f59e0b"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=color, width=2),
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode="lines",
        name="Random Baseline",
        line=dict(color="#475569", dash="dash"),
    ))
    fig_roc.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(bgcolor="#0f172a"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")
st.caption("TeleChurnAI | scikit-learn | Gradient Boosting + threshold tuning | Telecom Churn Dataset (BigML)")
