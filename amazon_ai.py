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
# UI STYLING (CALM + BIG CHAT)
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
    "neg_clicks": 15,
    "scale_acos": 0.25,
    "pause_acos": 0.6
}

# ----------------------------------------------------------
# COLUMN NORMALIZATION MAP (STRICT)
# ----------------------------------------------------------
FIELD_MAP = {
    "date": ["date"],
    "campaign": ["campaign name"],
    "ad_group": ["ad group name"],
    "search_term": ["customer search term"],
    "impressions": ["impressions"],
    "clicks": ["clicks"],
    "spend": ["spend"],
    "sales": ["total sales"],
    "orders": ["total orders"]
}

# ----------------------------------------------------------
# LOAD AMAZON SEARCH TERM FILE
# ----------------------------------------------------------
def load_excel(upload):
    df = pd.read_excel(upload)
    return df

# ----------------------------------------------------------
# CLEAN + NORMALIZE COLUMNS
# ----------------------------------------------------------
def normalize_columns(df):
    def clean(col):
        col = col.lower().strip()
        col = re.sub(r"\(.*?\)", "", col)   # remove bracketed text
        col = re.sub(r"[^a-z ]", "", col)
        col = re.sub(r"\s+", " ", col)
        return col.strip()

    df.columns = [clean(c) for c in df.columns]

    # Deduplicate columns safely
    df = df.loc[:, ~df.columns.duplicated()]

    rename_map = {}
    for canon, variants in FIELD_MAP.items():
        for col in df.columns:
            for v in variants:
                if v in col:
                    rename_map[col] = canon

    return df.rename(columns=rename_map)

# ----------------------------------------------------------
# METRICS (SAFE – AMAZON ONLY)
# ----------------------------------------------------------
def add_metrics(df):
    df = df.copy()

    for col in ["impressions", "clicks", "spend", "sales", "orders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ctr"] = df["clicks"] / df["impressions"].replace(0, pd.NA)
    df["cpc"] = df["spend"] / df["clicks"].replace(0, pd.NA)
    df["roas"] = df["sales"] / df["spend"].replace(0, pd.NA)
    df["acos"] = df["spend"] / df["sales"].replace(0, pd.NA)

    return df

# ----------------------------------------------------------
# DECISION ENGINE – AMAZON SEARCH TERM
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
- Use ONLY the provided summary
- Do NOT invent numbers
- Explain decisions using benchmarks

SUMMARY:
{summary}

QUESTION:
{question}

Respond with clear reasoning and actions.
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ==========================================================
# UI
# ==========================================================
st.title("🧘 Amazon Search Term AI")
st.caption("Benchmark-driven search term optimization with explainable AI")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = normalize_columns(df)

    REQUIRED = {"search_term", "clicks", "spend", "sales", "orders"}
    if not REQUIRED.issubset(df.columns):
        st.error("This does not look like a valid Amazon Search Term report.")
        st.stop()

    df = add_metrics(df)
    df = apply_decisions(df)

    # ---------------- Metrics ----------------
    st.subheader("📊 Performance Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Spend", f"{df['spend'].sum():,.2f}")
    c2.metric("Sales", f"{df['sales'].sum():,.2f}")
    c3.metric("ROAS", f"{df['sales'].sum() / df['spend'].sum():.2f}")

    # ---------------- Chart ----------------
    st.subheader("📈 Decision Distribution")
    st.plotly_chart(
        px.histogram(df, x="decision", color="decision"),
        use_container_width=True
    )

    # ---------------- Table ----------------
    st.subheader("🔍 Search Term Preview")
    st.dataframe(
        df.sort_values("spend", ascending=False).head(50),
        use_container_width=True
    )

    # ---------------- AI CHAT ----------------
    st.subheader("💬 AI Optimization Assistant")

    summary = df.groupby("decision")[["spend", "sales", "orders"]].sum().reset_index().to_string(index=False)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about negatives, wasted spend, scaling opportunities…")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            answer = ask_ai(question, summary)
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
