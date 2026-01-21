# ==========================================================
# Amazon Sponsored Products – Search Term AI Analyzer
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
    page_title="Amazon Search Term AI",
    page_icon="🧘",
    layout="wide"
)

# ----------------------------------------------------------
# UI STYLING (BIG, ROUNDED, SOOTHING CHAT)
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
    font-size: 16px;
}

textarea {
    border-radius: 26px !important;
    padding: 18px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# BENCHMARKS – AMAZON SEARCH TERM
# ----------------------------------------------------------
BENCHMARKS = {
    "neg_clicks": 15,      # clicks with zero orders → negate
    "scale_acos": 0.25,    # good performance
    "pause_acos": 0.60     # bad performance
}

# ----------------------------------------------------------
# CANONICAL FIELD DEFINITIONS
# ----------------------------------------------------------
FIELD_MAP = {
    "date": ["date"],
    "campaign": ["campaign name"],
    "ad_group": ["ad group name"],
    "search_term": ["customer search term"],
    "match_type": ["match type"],
    "impressions": ["impressions"],
    "clicks": ["clicks"],
    "spend": ["spend"],
    "sales": ["total sales"],
    "orders": ["total orders"]
}

# ----------------------------------------------------------
# LOAD EXCEL FILE
# ----------------------------------------------------------
def load_excel(upload):
    return pd.read_excel(upload)

# ----------------------------------------------------------
# CLEAN + NORMALIZE COLUMNS (AMAZON SAFE)
# ----------------------------------------------------------
def normalize_columns(df):
    def clean(col):
        col = col.lower().strip()
        col = re.sub(r"\(.*?\)", "", col)        # remove brackets
        col = re.sub(r"[^a-z ]", "", col)        # remove numbers/symbols
        col = re.sub(r"\s+", " ", col)
        return col.strip()

    df.columns = [clean(c) for c in df.columns]

    rename_map = {}
    for canon, variants in FIELD_MAP.items():
        for col in df.columns:
            for v in variants:
                if v in col:
                    rename_map[col] = canon

    df = df.rename(columns=rename_map)

    return df

# ----------------------------------------------------------
# METRIC ENGINE (HANDLES DUPLICATE AMAZON COLUMNS)
# ----------------------------------------------------------
def add_metrics(df):
    df = df.copy()

    for metric in ["impressions", "clicks", "spend", "sales", "orders"]:
        cols = [c for c in df.columns if c == metric]

        if len(cols) > 1:
            df[metric] = (
                df[cols]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0), axis=0)
                .sum(axis=1)
            )
            df = df.drop(columns=cols[1:])
        elif len(cols) == 1:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)
        else:
            df[metric] = 0

    # Derived metrics (safe division)
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, pd.NA)
    df["cpc"] = df["spend"] / df["clicks"].replace(0, pd.NA)
    df["roas"] = df["sales"] / df["spend"].replace(0, pd.NA)
    df["acos"] = df["spend"] / df["sales"].replace(0, pd.NA)

    return df

# ----------------------------------------------------------
# DECISION ENGINE – SEARCH TERM LOGIC
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
# AI EXPLAINER (BENCHMARK-AWARE)
# ----------------------------------------------------------
def ask_ai(question, summary):
    prompt = f"""
You are an Amazon Ads Search Term Optimization Assistant.

Rules:
- Use ONLY the provided data
- Do NOT invent metrics
- Explain decisions using benchmarks

DATA SUMMARY:
{summary}

QUESTION:
{question}

Respond with reasoning and next actions.
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("🧘 Amazon Search Term AI")
st.caption("Benchmark-driven search term optimization with explainable AI")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = normalize_columns(df)

    REQUIRED = {"search_term", "clicks", "spend", "sales", "orders"}
    if not REQUIRED.issubset(df.columns):
        st.error("This does not appear to be a valid Amazon Search Term report.")
        st.stop()

    df = add_metrics(df)
    df = apply_decisions(df)

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------
    st.subheader("📊 Performance Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Spend", f"{df['spend'].sum():,.2f}")
    c2.metric("Sales", f"{df['sales'].sum():,.2f}")
    c3.metric("ROAS", f"{df['sales'].sum() / df['spend'].sum():.2f}")

    # ------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------
    st.subheader("📈 Decision Distribution")
    st.plotly_chart(
        px.histogram(df, x="decision", color="decision"),
        use_container_width=True
    )

    # ------------------------------------------------------
    # TABLE
    # ------------------------------------------------------
    st.subheader("🔍 Search Terms (Top Spend)")
    st.dataframe(
        df.sort_values("spend", ascending=False).head(50),
        use_container_width=True
    )

    # ------------------------------------------------------
    # AI CHAT
    # ------------------------------------------------------
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

    question = st.chat_input(
        "Ask about negatives, wasted spend, scaling opportunities, next actions…"
    )

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            answer = ask_ai(question, summary)
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})

