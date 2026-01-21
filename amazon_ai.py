# ==========================================================
# Advertisement Monk.AI — FULL PRODUCTION APP
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
    page_title="Advertisement Monk.AI",
    page_icon="🧘",
    layout="wide"
)

# ----------------------------------------------------------
# SOOTHING UI + BIG ROUNDED CHAT
# ----------------------------------------------------------
st.markdown("""
<style>
.main { background-color: #F8FAFC; }
.block-container { padding-top: 2rem; }

[data-testid="stChatMessageContainer"] {
    background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
    border-radius: 30px;
    padding: 28px;
    border: 2px solid #7DD3FC;
    margin-bottom: 18px;
    font-size: 16px;
}

textarea {
    border-radius: 25px !important;
    padding: 18px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# BENCHMARKS
# ----------------------------------------------------------
BENCHMARKS = {
    "amazon": {
        "neg_clicks": 15,
        "scale_acos": 0.25,
        "pause_acos": 0.6
    },
    "qc": {
        "scale_roas": 3.0,
        "pause_roas": 1.2,
        "min_ctr": 0.012
    }
}

# ----------------------------------------------------------
# CANONICAL FIELD MAP
# ----------------------------------------------------------
FIELD_MAP = {
    "date": ["date"],
    "campaign": ["campaign name"],
    "ad_group": ["ad group name"],
    "search_term": ["customer search term"],
    "impressions": ["impressions", "views"],
    "clicks": ["clicks", "taps"],
    "spend": ["spend", "ad spend", "cost"],
    "sales": ["total sales", "sales", "gmv"],
    "orders": ["total orders", "orders"]
}

# ----------------------------------------------------------
# HEADER AUTO-DETECTION FOR MESSY CSVs
# ----------------------------------------------------------
def detect_header_row(upload):
    preview = pd.read_csv(upload, header=None, nrows=30, encoding_errors="ignore")
    for i, row in preview.iterrows():
        joined = " ".join(map(str, row.values)).lower()
        if "impressions" in joined and "click" in joined:
            return i
    return 0

# ----------------------------------------------------------
# FILE LOADER
# ----------------------------------------------------------
def load_file(upload):
    if upload.name.endswith(".csv"):
        header = detect_header_row(upload)
        upload.seek(0)
        df = pd.read_csv(upload, skiprows=header)
    else:
        df = pd.read_excel(upload)

    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ----------------------------------------------------------
# NORMALIZE COLUMNS (ROBUST)
# ----------------------------------------------------------
def normalize_columns(df):
    def clean(col):
        col = col.lower().strip()
        col = re.sub(r"[^a-z ]", "", col)
        return col

    df.columns = [clean(c) for c in df.columns]

    rename_map = {}
    for canon, variants in FIELD_MAP.items():
        for col in df.columns:
            for v in variants:
                if v in col:
                    rename_map[col] = canon

    return df.rename(columns=rename_map)

# ----------------------------------------------------------
# REPORT TYPE DETECTION
# ----------------------------------------------------------
def detect_report(df):
    if "search_term" in df.columns:
        return "amazon_search_term"
    if "campaign" in df.columns and "search_term" not in df.columns:
        return "amazon_campaign"
    if "gmv" in df.columns or "city" in df.columns:
        return "quick_commerce"
    return "unknown"

# ----------------------------------------------------------
# SAFE METRIC GENERATION
# ----------------------------------------------------------
def add_metrics(df):
    if {"impressions", "clicks"}.issubset(df.columns):
        df["ctr"] = df["clicks"] / df["impressions"]

    if {"clicks", "spend"}.issubset(df.columns):
        df["cpc"] = df["spend"] / df["clicks"]

    if {"sales", "spend"}.issubset(df.columns):
        df["roas"] = df["sales"] / df["spend"]
        df["acos"] = df["spend"] / df["sales"]

    return df

# ----------------------------------------------------------
# DECISION ENGINE (SAFE)
# ----------------------------------------------------------
def apply_decisions(df, report_type):
    df["decision"] = "Maintain"

    if report_type == "amazon_search_term":
        if {"clicks", "orders"}.issubset(df.columns):
            df.loc[
                (df["clicks"] >= BENCHMARKS["amazon"]["neg_clicks"]) &
                (df["orders"] == 0),
                "decision"
            ] = "NEGATIVE"

        if "acos" in df.columns:
            df.loc[df["acos"] <= BENCHMARKS["amazon"]["scale_acos"], "decision"] = "SCALE"
            df.loc[df["acos"] >= BENCHMARKS["amazon"]["pause_acos"], "decision"] = "PAUSE"

    if report_type == "quick_commerce":
        if "roas" in df.columns:
            df.loc[df["roas"] >= BENCHMARKS["qc"]["scale_roas"], "decision"] = "SCALE"
            df.loc[df["roas"] < BENCHMARKS["qc"]["pause_roas"], "decision"] = "PAUSE"

        if "ctr" in df.columns:
            df.loc[df["ctr"] < BENCHMARKS["qc"]["min_ctr"], "decision"] = "OPTIMIZE"

    return df

# ----------------------------------------------------------
# AI EXPLANATION LAYER
# ----------------------------------------------------------
def ask_ai(question, context):
    prompt = f"""
You are an Ads Optimization Assistant.
Do not invent metrics.
Use ONLY the provided summary.

SUMMARY:
{context}

QUESTION:
{question}

Explain decisions clearly and suggest actions.
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ==========================================================
# UI
# ==========================================================
st.title("🧘 Advertisement Monk.AI")
st.caption("Upload any ads report. Understand. Optimize. Decide.")

uploaded = st.file_uploader(
    "Upload Amazon or Quick Commerce Ads Report",
    type=["csv", "xlsx"]
)

if uploaded:
    df = load_file(uploaded)
    df = normalize_columns(df)
    report_type = detect_report(df)
    df = add_metrics(df)
    df = apply_decisions(df, report_type)

    st.subheader("📊 Performance Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Spend", f"{df['spend'].sum():,.2f}" if "spend" in df.columns else "N/A")
    c2.metric("Sales / GMV", f"{df['sales'].sum():,.2f}" if "sales" in df.columns else "N/A")

    if {"sales", "spend"}.issubset(df.columns):
        c3.metric("ROAS", f"{df['sales'].sum()/df['spend'].sum():.2f}")
    else:
        c3.metric("ROAS", "N/A")

    st.subheader("📈 Decision Distribution")
    st.plotly_chart(
        px.histogram(df, x="decision", color="decision"),
        use_container_width=True
    )

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # ------------------------------------------------------
    # AI CHAT
    # ------------------------------------------------------
    st.subheader("💬 Ads Intelligence Assistant")

    summary_cols = [c for c in ["spend", "sales", "orders"] if c in df.columns]
    context = df.groupby("decision")[summary_cols].sum().reset_index().to_string(index=False)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about scaling, negatives, wasted spend, next actions…")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            answer = ask_ai(question, context)
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
