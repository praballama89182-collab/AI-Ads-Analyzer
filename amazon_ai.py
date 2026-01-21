# ==========================================================
# amazon_ai — Amazon Search Term AI (FINAL, SAFE)
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import re

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="amazon_ai",
    page_icon="🧘",
    layout="wide"
)

# ----------------------------------------------------------
# UI STYLING
# ----------------------------------------------------------
st.markdown("""
<style>
.main { background-color: #F8FAFC; }
.block-container { padding-top: 2rem; }

[data-testid="stChatMessageContainer"] {
    background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
    border-radius: 32px;
    padding: 26px;
    border: 2px solid #7DD3FC;
    margin-bottom: 16px;
}

textarea {
    border-radius: 26px !important;
    padding: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# BENCHMARKS
# ----------------------------------------------------------
BENCHMARKS = {
    "neg_clicks": 15,
    "scale_acos": 0.25,
    "pause_acos": 0.6
}

# ----------------------------------------------------------
# FIELD MAP (AMAZON SEARCH TERM)
# ----------------------------------------------------------
FIELD_MAP = {
    "search_term": ["customer search term"],
    "match_type": ["match type"],
    "impressions": ["impressions"],
    "clicks": ["clicks"],
    "spend": ["spend"],
    "sales": ["total sales"],
    "orders": ["total orders"]
}

# ----------------------------------------------------------
# LOAD FILE
# ----------------------------------------------------------
def load_excel(upload):
    return pd.read_excel(upload)

# ----------------------------------------------------------
# NORMALIZE + FORCE CANONICAL COLUMNS
# ----------------------------------------------------------
def normalize_columns(df):
    def clean(col):
        col = col.lower().strip()
        col = re.sub(r"\(.*?\)", "", col)
        col = re.sub(r"[^a-z ]", "", col)
        return col.strip()

    df.columns = [clean(c) for c in df.columns]

    # Map to canonical names
    for canon, variants in FIELD_MAP.items():
        matched = [c for c in df.columns if any(v in c for v in variants)]
        if matched:
            df[canon] = (
                df[matched]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0), axis=0)
                .sum(axis=1)
            )
        else:
            df[canon] = 0  # FORCE column existence

    return df

# ----------------------------------------------------------
# METRICS (ZERO-SAFE, KEYERROR-SAFE)
# ----------------------------------------------------------
def add_metrics(df):
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, pd.NA)
    df["cpc"] = df["spend"] / df["clicks"].replace(0, pd.NA)
    df["roas"] = df["sales"] / df["spend"].replace(0, pd.NA)
    df["acos"] = df["spend"] / df["sales"].replace(0, pd.NA)
    return df

# ----------------------------------------------------------
# DECISION ENGINE
# ----------------------------------------------------------
def apply_decisions(df):
    df["decision"] = "Maintain"

    df.loc[
        (df["clicks"] >= BENCHMARKS["neg_clicks"]) &
        (df["orders"] == 0),
        "decision"
    ] = "NEGATIVE"

    df.loc[df["acos"] <= BENCHMARKS["scale_acos"], "decision"] = "SCALE"
    df.loc[df["acos"] >= BENCHMARKS["pause_acos"], "decision"] = "PAUSE"

    return df

# ----------------------------------------------------------
# AI EXPLAINER
# ----------------------------------------------------------
def ask_ai(question, summary):
    prompt = f"""
You are an Amazon Search Term Optimization Assistant.
Use ONLY the data below. Do not invent metrics.

DATA SUMMARY:
{summary}

QUESTION:
{question}

Explain decisions and actions clearly.
"""
    return ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]

# ==========================================================
# UI
# ==========================================================
st.title("🧘 amazon_ai")
st.caption("Amazon Search Term AI — benchmark-driven optimization")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = normalize_columns(df)
    df = add_metrics(df)
    df = apply_decisions(df)

    # ---------------- METRICS ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Spend", f"{df['spend'].sum():,.2f}")
    c2.metric("Sales", f"{df['sales'].sum():,.2f}")
    c3.metric("ROAS", f"{df['sales'].sum()/df['spend'].sum():.2f}" if df["spend"].sum() > 0 else "N/A")

    # ---------------- CHART ----------------
    st.plotly_chart(
        px.histogram(df, x="decision", color="decision"),
        use_container_width=True
    )

    # ---------------- TABLE ----------------
    st.dataframe(
        df.sort_values("spend", ascending=False).head(50),
        use_container_width=True
    )

    # ---------------- AI CHAT ----------------
    st.subheader("💬 AI Optimization Assistant")

    summary = (
        df.groupby("decision")[["spend", "sales", "orders"]]
        .sum()
        .reset_index()
        .to_string(index=False)
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask about negatives, wasted spend, scaling opportunities…")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            a = ask_ai(q, summary)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
