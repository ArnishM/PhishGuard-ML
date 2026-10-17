import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Custom function needed to unpickle the Naive Bayes model
def shift_features(X):
    return X + 1

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PhishGuard ML", page_icon="shield", layout="wide")

# ── Minimal CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }

    .stat-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .stat-box .num  { font-size: 1.5rem; font-weight: 600; color: #f9fafb; }
    .stat-box .lbl  { font-size: 0.78rem; color: #9ca3af; margin-top: 0.15rem; }

    .verdict-bad {
        background: #1c1012; border: 1px solid #7f1d1d;
        border-radius: 8px; padding: 1.6rem; text-align: center;
    }
    .verdict-bad h3 { color: #fca5a5; margin: 0 0 0.4rem 0; }
    .verdict-bad p  { color: #d1d5db; margin: 0; font-size: 0.9rem; }

    .verdict-ok {
        background: #0c1a12; border: 1px solid #14532d;
        border-radius: 8px; padding: 1.6rem; text-align: center;
    }
    .verdict-ok h3 { color: #86efac; margin: 0 0 0.4rem 0; }
    .verdict-ok p  { color: #d1d5db; margin: 0; font-size: 0.9rem; }

    .tag { display: inline-block; font-size: 0.7rem; font-weight: 600;
           padding: 0.15rem 0.5rem; border-radius: 4px; margin-right: 0.4rem; }
    .tag-red  { background: #7f1d1d; color: #fca5a5; }
    .tag-amber { background: #78350f; color: #fde68a; }
</style>
""", unsafe_allow_html=True)


# ── Data & Models ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/phishing_websites.csv")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("models"):
        for f in os.listdir("models"):
            if f.endswith(".pkl"):
                name = f.replace(".pkl", "").replace("_", " ").title()
                models[name] = joblib.load(f"models/{f}")
    return models

df = load_data()
models = load_models()


# ── Feature metadata (human labels, no emojis) ──────────────────────────────
FEATURES = {
    "sslfinal_state":    {"label": "SSL Certificate",       "options": {-1: "Invalid or absent",   0: "Suspicious",       1: "Trusted and valid"}},
    "url_of_anchor":     {"label": "Anchor URLs",           "options": {-1: "Mostly external",     0: "Mixed",            1: "Same domain"}},
    "web_traffic":       {"label": "Web Traffic",           "options": {-1: "Very low or none",    0: "Moderate",         1: "High"}},
    "having_ip_address": {"label": "Domain vs IP Address",  "options": {-1: "Uses raw IP address", 1: "Uses domain name"}},
    "prefix_suffix":     {"label": "Dash in Domain",        "options": {-1: "Contains dash",       1: "No dash"}},
    "url_length":        {"label": "URL Length",            "options": {-1: "Very long",           0: "Moderate",         1: "Short"}},
    "having_sub_domain": {"label": "Sub-domain Count",      "options": {-1: "Multiple",            0: "One",              1: "Normal"}},
    "request_url":       {"label": "External Requests",     "options": {-1: "Mostly external",     0: "Some external",    1: "Mostly internal"}},
    "sfh":               {"label": "Form Handler (SFH)",    "options": {-1: "Suspicious or empty", 0: "Points elsewhere", 1: "Same domain"}},
    "age_of_domain":     {"label": "Domain Age",            "options": {-1: "Less than 6 months",  1: "Over 6 months"}},
}

DEMO_KEYS = list(FEATURES.keys())


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PhishGuard ML")
    st.caption("Phishing Website Detection")
    st.markdown("---")
    page = st.radio(
        "Pages",
        ["Overview", "Model Comparison", "Predict", "Risk Analysis"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("UCI Phishing Websites Dataset · Scikit-learn")

if df is None:
    st.error("Dataset not found. Run the Jupyter notebooks first.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.title("PhishGuard ML")
    st.markdown("A supervised machine-learning system that classifies websites as **phishing** or **legitimate** based on 30 pre-extracted URL and page characteristics.")
    st.markdown("---")

    total = len(df)
    phishing = int(df["Result"].sum())
    legit = total - phishing
    features = df.shape[1] - 1

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, f"{total:,}",    "Total records"),
        (c2, f"{legit:,}",    "Legitimate"),
        (c3, f"{phishing:,}", "Phishing"),
        (c4, str(features),   "Features"),
    ]:
        col.markdown(f'<div class="stat-box"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        bars = ax.bar(["Legitimate", "Phishing"], [legit, phishing],
                      color=["#3b82f6", "#ef4444"], width=0.45, edgecolor="none")
        for b, v in zip(bars, [legit, phishing]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
                    f"{v:,}", ha="center", color="#d1d5db", fontsize=11, fontweight="500")
        ax.tick_params(colors="#9ca3af")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#1f2937")
        ax.spines["bottom"].set_color("#1f2937")
        ax.set_ylabel("Count", color="#9ca3af")
        plt.tight_layout()
        st.pyplot(fig)

    with col_r:
        if os.path.exists("data/feature_importance.csv"):
            st.subheader("Top features (mutual information)")
            fi = pd.read_csv("data/feature_importance.csv", index_col=0)
            fi.columns = ["score"]
            top = fi.head(10).iloc[::-1]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            ax.barh(range(len(top)), top["score"], color="#6366f1", height=0.55, edgecolor="none")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index, fontsize=8.5)
            ax.tick_params(colors="#9ca3af")
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color("#1f2937")
            ax.spines["bottom"].set_color("#1f2937")
            ax.set_xlabel("MI score", color="#9ca3af")
            plt.tight_layout()
            st.pyplot(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":

    st.title("Model Comparison")

    if not models:
        st.warning("No trained models found.")
        st.stop()
    try:
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv")["Result"]
    except FileNotFoundError:
        st.error("Test data not found.")
        st.stop()

    results, cms, rocs = [], {}, {}
    for name, model in models.items():
        pred = model.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec  = recall_score(y_test, pred, zero_division=0)
        f1   = f1_score(y_test, pred, zero_division=0)
        cms[name] = confusion_matrix(y_test, pred)
        auc = np.nan
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, prob)
                fpr, tpr, _ = roc_curve(y_test, prob)
                rocs[name] = (fpr, tpr, auc)
            except Exception:
                pass
        results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                        "Recall": rec, "F1": f1, "AUC": auc})

    res_df = pd.DataFrame(results).set_index("Model").sort_values("F1", ascending=False)

    st.subheader("Performance metrics")
    st.dataframe(res_df.style.format("{:.4f}").highlight_max(axis=0, color="#1e3a5f"),
                 use_container_width=True)
    best = res_df.index[0]
    st.markdown(f"**Best model by F1-score:** {best} ({res_df.loc[best, 'F1']:.4f})")

    # Confusion matrices
    st.markdown("---")
    st.subheader("Confusion matrices")
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    fig.patch.set_facecolor("#0e1117")
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.set_facecolor("#0e1117")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"],
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(name, color="#d1d5db", fontsize=10, fontweight="600", pad=8)
        ax.set_xlabel("Predicted", color="#9ca3af", fontsize=9)
        ax.set_ylabel("Actual", color="#9ca3af", fontsize=9)
        ax.tick_params(colors="#9ca3af")
    plt.tight_layout()
    st.pyplot(fig)

    # ROC
    if rocs:
        st.markdown("---")
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        palette = ["#3b82f6", "#ec4899", "#a78bfa", "#34d399"]
        for i, (name, (fpr, tpr, a)) in enumerate(rocs.items()):
            ax.plot(fpr, tpr, color=palette[i % len(palette)], linewidth=2,
                    label=f"{name}  AUC {a:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#4b5563", linewidth=0.8)
        ax.set_xlabel("False Positive Rate", color="#9ca3af")
        ax.set_ylabel("True Positive Rate", color="#9ca3af")
        ax.legend(loc="lower right", fontsize=9, facecolor="#111827",
                  edgecolor="#1f2937", labelcolor="#d1d5db")
        ax.tick_params(colors="#9ca3af")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#1f2937")
        ax.spines["bottom"].set_color("#1f2937")
        ax.grid(color="#1f2937", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Predict":

    st.title("Predict")
    st.markdown("Select website characteristics below and press **Run analysis**.")

    if not models:
        st.warning("No models found.")
        st.stop()

    model_choice = st.selectbox("Model", list(models.keys()))
    st.markdown("---")

    user_vals = {}
    for row_start in range(0, len(DEMO_KEYS), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(DEMO_KEYS):
                break
            feat = DEMO_KEYS[idx]
            info = FEATURES[feat]
            opts_map = info["options"]
            display = list(opts_map.values())
            raw     = list(opts_map.keys())
            with col:
                sel = st.selectbox(info["label"], display, index=0, key=f"p_{feat}")
                user_vals[feat] = raw[display.index(sel)]

    st.markdown("")
    if st.button("Run analysis", type="primary"):
        X_sample = df.drop(columns=["Result"]).mode().iloc[0].copy()
        for feat, val in user_vals.items():
            X_sample[feat] = val

        inp = pd.DataFrame([X_sample])
        mdl = models[model_choice]
        pred = mdl.predict(inp)[0]

        conf_pct = None
        if hasattr(mdl, "predict_proba"):
            try:
                conf_pct = max(mdl.predict_proba(inp)[0]) * 100
            except Exception:
                pass

        st.markdown("")
        conf_line = f"Model confidence: {conf_pct:.1f}%" if conf_pct else ""
        if pred == 1:
            st.markdown(f"""<div class="verdict-bad">
                <h3>Phishing detected</h3>
                <p>The selected characteristics match patterns commonly seen in phishing websites.<br>{conf_line}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="verdict-ok">
                <h3>Likely legitimate</h3>
                <p>The selected characteristics are consistent with legitimate websites.<br>{conf_line}</p>
            </div>""", unsafe_allow_html=True)

        # Summary table
        st.markdown("")
        st.markdown("**Input summary**")
        rows = []
        for feat in DEMO_KEYS:
            info = FEATURES[feat]
            rows.append({
                "Feature": info["label"],
                "Selection": info["options"].get(user_vals[feat], str(user_vals[feat]))
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  RISK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Risk Analysis":

    st.title("Risk Analysis")
    st.markdown("Not all classification errors carry the same cost. In phishing detection the two failure modes have very different consequences.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="verdict-bad">
            <h3>False Negative &nbsp;<span class="tag tag-red">CRITICAL</span></h3>
            <p style="text-align:left;">A phishing site is classified as legitimate. The user is not warned
            and may enter credentials on a fraudulent page, leading to identity theft or financial loss.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="background:#1a1708; border:1px solid #713f12; border-radius:8px; padding:1.6rem; text-align:center;">
            <h3 style="color:#fde68a; margin:0 0 0.4rem 0;">False Positive &nbsp;<span class="tag tag-amber">MODERATE</span></h3>
            <p style="color:#d1d5db; margin:0; font-size:0.9rem; text-align:left;">A legitimate site is flagged as phishing. The user is blocked from
            a safe page, causing inconvenience and reduced trust in the system.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.info("Because missing a phishing site is far more dangerous, we optimise for **Recall** (catching all phishing) over raw Accuracy.")

    st.markdown("---")
    st.subheader("Common phishing indicators")

    indicators = [
        ("SSL Certificate",  "Phishing sites frequently lack valid SSL certificates or use self-signed ones."),
        ("IP-based URLs",    "Legitimate sites use domain names. Phishing sites often use raw IP addresses to hide identity."),
        ("Anchor links",     "Phishing pages embed links that point to external domains while appearing to stay on-site."),
        ("Domain age",       "Newly registered domains (under 6 months) are disproportionately used for phishing."),
        ("Low web traffic",  "Phishing sites typically have no traffic history, unlike established legitimate sites."),
        ("Dashes in domain", "Hyphens in domain names are uncommon for legitimate sites but common in phishing (e.g. pay-pal-secure.com)."),
    ]

    for i in range(0, len(indicators), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(indicators):
                title, desc = indicators[i + j]
                col.markdown(f"**{title}**")
                col.caption(desc)

    st.markdown("---")
    st.caption("This system predicts from pre-extracted website features. It is not a real-time URL scanner. Always verify suspicious websites through multiple sources.")
