"""
app.py — Streamlit UI for the moderation baselines.

    pip install streamlit
    streamlit run app.py

Tabs
  1. Analyze a comment  — both models' scores + an illustrative policy decision
  2. Batch scoring      — upload a CSV, score every row, download results
  3. Model performance  — test metrics saved by the training scripts

Only models whose files exist in artifacts/ appear. Train them first:
    python model1_tfidf_lr.py
    python model2_tfidf_baselines.py
    python bilstm.py --task jigsaw / --task hatexplain   (optional, needs torch)
"""
import glob
import json
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data import HATEXPLAIN_CLASSES, JIGSAW_LABELS, normalize

ART = "artifacts"
LABEL_NAMES = {"toxic": "Toxic", "severe_toxic": "Severe toxic", "obscene": "Obscene",
               "threat": "Threat", "insult": "Insult", "identity_hate": "Identity hate"}
CLASS_COLORS = {"hate": "#c0392b", "offensive": "#e67e22", "normal": "#27ae60"}

st.set_page_config(page_title="Moderation Model Lab", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Model loading — each loader returns predict(list[str]) -> np.ndarray
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    m1, m2 = {}, {}

    p = os.path.join(ART, "model1_tfidf_lr.joblib")
    if os.path.exists(p):
        b = joblib.load(p)
        m1["TF-IDF + Logistic Regression"] = (
            lambda texts, b=b: b["classifier"].predict_proba(b["features"].transform(texts)),
            np.asarray(b["thresholds"]))

    names = {"logreg": "TF-IDF + Logistic Regression",
             "logreg_weighted": "TF-IDF + Logistic Regression (class-weighted)",
             "naive_bayes": "Multinomial Naive Bayes"}
    for key, nice in names.items():
        p = os.path.join(ART, f"model2_{key}.joblib")
        if os.path.exists(p):
            b = joblib.load(p)
            m2[nice] = lambda texts, b=b: b["pipeline"].predict_proba(texts)

    # BiLSTM is optional: needs torch and a trained checkpoint
    for task, target in (("jigsaw", m1), ("hatexplain", m2)):
        p = os.path.join(ART, f"bilstm_{task}.pt")
        if os.path.exists(p):
            try:
                from bilstm import load_bilstm
                fn, _, thr = load_bilstm(p)
                target["BiLSTM"] = (fn, np.asarray(thr)) if task == "jigsaw" else fn
            except Exception as e:     # torch missing or incompatible
                st.sidebar.caption(f"BiLSTM ({task}) not loaded: {e}")
    return m1, m2


m1_models, m2_models = load_models()

if not m1_models and not m2_models:
    st.error("No trained models found in `artifacts/`. Run `python model1_tfidf_lr.py` "
             "and `python model2_tfidf_baselines.py` first.")
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar — model choice and the (illustrative) policy layer
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Models")
    m1_name = st.selectbox("Model 1 · toxicity (Jigsaw)", list(m1_models) or ["— not trained —"])
    m2_name = st.selectbox("Model 2 · hate / offensive / normal (HateXplain)",
                           list(m2_models) or ["— not trained —"])

    st.header("Policy layer")
    st.caption("Illustrative rules that turn scores into an action. "
               "Set these from validation data for a real deployment (report §13).")
    hate_cut = st.slider("Restrict if P(hate) ≥", 0.0, 1.0, 0.60, 0.05)
    restrict_threat = st.checkbox("Restrict if Threat is flagged", value=True)
    restrict_severe = st.checkbox("Restrict if Severe toxic is flagged", value=True)
    review_offensive = st.slider("Review if P(offensive) ≥", 0.0, 1.0, 0.50, 0.05)
    st.caption("Any other Model 1 flag → Review. Otherwise → Allow.")


def score(texts):
    """Run both selected models. Returns (P1 or None, thresholds, P2 or None)."""
    texts = [normalize(t) for t in texts]
    P1 = thr = P2 = None
    if m1_name in m1_models:
        fn, thr = m1_models[m1_name]
        P1 = fn(texts)
    if m2_name in m2_models:
        P2 = m2_models[m2_name](texts)
    return P1, thr, P2


def decide(p1, thr, p2):
    """Policy layer: probabilities -> (action, reasons)."""
    reasons_r, reasons_v = [], []
    flags = {lab: bool(p1[j] >= thr[j]) for j, lab in enumerate(JIGSAW_LABELS)} if p1 is not None else {}
    if p2 is not None and p2[0] >= hate_cut:
        reasons_r.append(f"P(hate) = {p2[0]:.2f}")
    if restrict_threat and flags.get("threat"):
        reasons_r.append("Threat flagged")
    if restrict_severe and flags.get("severe_toxic"):
        reasons_r.append("Severe toxic flagged")
    if reasons_r:
        return "Restrict", reasons_r
    if p2 is not None and p2[1] >= review_offensive:
        reasons_v.append(f"P(offensive) = {p2[1]:.2f}")
    reasons_v += [f"{LABEL_NAMES[l]} flagged" for l, f in flags.items() if f]
    if reasons_v:
        return "Review", reasons_v
    return "Allow", ["No label above its threshold"]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def model1_chart(p1, thr):
    df = pd.DataFrame({"label": [LABEL_NAMES[l] for l in JIGSAW_LABELS], "prob": p1,
                       "threshold": thr, "flag": p1 >= thr})
    order = [LABEL_NAMES[l] for l in JIGSAW_LABELS]
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort=order, title=None))
    bars = base.mark_bar(cornerRadiusEnd=3, height=22).encode(
        x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
        color=alt.condition("datum.flag", alt.value("#c0392b"), alt.value("#95a5a6")),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("prob:Q", format=".3f"),
                 alt.Tooltip("threshold:Q", format=".2f")])
    ticks = base.mark_tick(color="#2c3e50", thickness=2, size=30).encode(x="threshold:Q")
    text = base.mark_text(align="left", dx=4, fontSize=12).encode(
        x="prob:Q", text=alt.Text("prob:Q", format=".2f"))
    return (bars + ticks + text).properties(height=260)


def model2_chart(p2):
    df = pd.DataFrame({"class": [c.capitalize() for c in HATEXPLAIN_CLASSES], "prob": p2,
                       "key": HATEXPLAIN_CLASSES})
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=28).encode(
        y=alt.Y("class:N", sort=None, title=None),
        x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="Probability (sums to 1)"),
        color=alt.Color("key:N", scale=alt.Scale(domain=list(CLASS_COLORS),
                                                  range=list(CLASS_COLORS.values())), legend=None),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("prob:Q", format=".3f")])
    text = bars.mark_text(align="left", dx=4, fontSize=12).encode(
        text=alt.Text("prob:Q", format=".2f"), color=alt.value("#2c3e50"))
    return (bars + text).properties(height=150)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🛡️ Moderation Model Lab")
st.caption("Model 1 asks *what kinds of toxic behavior are present?* · "
           "Model 2 asks *is it hate, offensive, or normal?* — the two answers are kept separate.")

tab1, tab2, tab3 = st.tabs(["Analyze a comment", "Batch scoring", "Model performance"])

# ---------------- Tab 1 ----------------
with tab1:
    examples = {
        "— pick an example —": "",
        "Polite": "Thanks for fixing the citation, the section reads much better now.",
        "Insulting": "This is the dumbest edit I have ever seen, you clueless clown.",
        "Threatening": "Revert my edit one more time and you will regret it, I know where you live.",
        "Crude but not targeted": "Damn, this article is a hell of a mess.",
    }
    ex = st.selectbox("Try an example", list(examples))
    text = st.text_area("Comment", value=examples[ex], height=110,
                        placeholder="Paste or type a comment…")

    if st.button("Analyze", type="primary", disabled=not text.strip()):
        P1, thr, P2 = score([text])
        p1 = P1[0] if P1 is not None else None
        p2 = P2[0] if P2 is not None else None

        action, reasons = decide(p1, thr, p2)
        style = {"Allow": st.success, "Review": st.warning, "Restrict": st.error}[action]
        style(f"**Suggested action: {action}** — " + "; ".join(reasons))

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.subheader("Model 1 · toxicity")
            if p1 is None:
                st.info("Model 1 is not trained yet (needs the Jigsaw data).")
            else:
                st.altair_chart(model1_chart(p1, thr), width="stretch")
                st.caption("Red bar = above that label's tuned threshold (dark tick). "
                           "Labels are independent, so several can fire at once.")
        with c2:
            st.subheader("Model 2 · hate / offensive / normal")
            if p2 is None:
                st.info("Model 2 is not trained yet.")
            else:
                top = HATEXPLAIN_CLASSES[int(np.argmax(p2))]
                st.markdown(f"Predicted class: **:{'red' if top == 'hate' else 'orange' if top == 'offensive' else 'green'}[{top.capitalize()}]**")
                st.altair_chart(model2_chart(p2), width="stretch")

        with st.expander("Raw output (JSON)"):
            out = {}
            if p1 is not None:
                out["model1"] = {l: {"prob": round(float(p1[j]), 4), "threshold": round(float(thr[j]), 2)}
                                 for j, l in enumerate(JIGSAW_LABELS)}
            if p2 is not None:
                out["model2"] = {c: round(float(p2[i]), 4) for i, c in enumerate(HATEXPLAIN_CLASSES)}
            out["policy"] = {"action": action, "reasons": reasons}
            st.json(out)

# ---------------- Tab 2 ----------------
with tab2:
    st.write("Upload a CSV with one comment per row. Every row is scored by both selected models.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        guess = next((c for c in df.columns if c.lower() in ("text", "comment", "comment_text")),
                     df.columns[0])
        col = st.selectbox("Text column", list(df.columns), index=list(df.columns).index(guess))
        limit = st.number_input("Rows to score", 1, len(df), min(len(df), 5000))
        if st.button("Score", type="primary"):
            sub = df.head(int(limit)).copy()
            texts = sub[col].fillna("").astype(str).tolist()
            with st.spinner(f"Scoring {len(texts)} rows…"):
                P1, thr, P2 = score(texts)
            if P1 is not None:
                for j, l in enumerate(JIGSAW_LABELS):
                    sub[f"m1_{l}"] = P1[:, j].round(4)
            if P2 is not None:
                for i, c in enumerate(HATEXPLAIN_CLASSES):
                    sub[f"m2_{c}"] = P2[:, i].round(4)
                sub["m2_class"] = [HATEXPLAIN_CLASSES[i] for i in P2.argmax(1)]
            sub["action"] = [decide(P1[i] if P1 is not None else None, thr,
                                    P2[i] if P2 is not None else None)[0] for i in range(len(sub))]

            counts = sub["action"].value_counts()
            a, b, c = st.columns(3)
            a.metric("Allow", int(counts.get("Allow", 0)))
            b.metric("Review", int(counts.get("Review", 0)))
            c.metric("Restrict", int(counts.get("Restrict", 0)))
            st.dataframe(sub, width="stretch", height=420)
            st.download_button("Download scored CSV", sub.to_csv(index=False).encode(),
                               "scored_comments.csv", "text/csv")

# ---------------- Tab 3 ----------------
with tab3:
    files = sorted(glob.glob(os.path.join(ART, "metrics_*.json")))
    if not files:
        st.info("No saved metrics yet — they're written when you run the training scripts.")
    metrics = [json.load(open(f)) for f in files]
    if any(m.get("demo") for m in metrics):
        st.warning("Some metrics below come from the synthetic --demo data and are meaningless.")

    m1_runs = [m for m in metrics if m["task"] == "jigsaw"]
    m2_runs = [m for m in metrics if m["task"] == "hatexplain"]

    st.subheader("Model 1 · Jigsaw (test set, tuned thresholds)")
    if not m1_runs:
        st.info("Not trained yet.")
    for m in m1_runs:
        t = m["test"]
        st.markdown(f"**{m['model']}** — macro-F1 **{t['macro_f1']:.3f}**, micro-F1 **{t['micro_f1']:.3f}**")
        tbl = pd.DataFrame(t["per_label"]).T
        tbl.index = [LABEL_NAMES[i] for i in tbl.index]
        st.dataframe(tbl[["precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "support"]]
                     .style.format({"support": "{:.0f}", "threshold": "{:.2f}"}, precision=3),
                     width="stretch")

    st.subheader("Model 2 · HateXplain (test set)")
    if not m2_runs:
        st.info("Not trained yet.")
    else:
        summ = pd.DataFrame([{"model": m["model"], "macro-F1": m["test"]["macro_f1"],
                              "accuracy": m["test"]["accuracy"],
                              **{f"F1 {c}": m["test"]["per_class"][c]["f1"] for c in HATEXPLAIN_CLASSES}}
                             for m in m2_runs]).set_index("model")
        st.dataframe(summ.style.format(precision=3).highlight_max(axis=0, color="#d5f5e3"),
                     width="stretch")

        pick = st.selectbox("Confusion matrix for", [m["model"] for m in m2_runs])
        cm = np.array(next(m for m in m2_runs if m["model"] == pick)["test"]["confusion_matrix"])
        cm_df = pd.DataFrame([{"true": HATEXPLAIN_CLASSES[i], "predicted": HATEXPLAIN_CLASSES[j],
                               "count": int(cm[i, j]), "row_pct": cm[i, j] / cm[i].sum()}
                              for i in range(3) for j in range(3)])
        heat = alt.Chart(cm_df).mark_rect().encode(
            x=alt.X("predicted:N", sort=HATEXPLAIN_CLASSES), y=alt.Y("true:N", sort=HATEXPLAIN_CLASSES),
            color=alt.Color("row_pct:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["true", "predicted", "count", alt.Tooltip("row_pct:Q", format=".1%")])
        labels = heat.mark_text(fontSize=14).encode(
            text="count:Q",
            color=alt.condition("datum.row_pct > 0.5", alt.value("white"), alt.value("black")))
        st.altair_chart((heat + labels).properties(width=360, height=300))
        st.caption("Rows = true class. The hate ↔ offensive cells show where the task is hardest.")
