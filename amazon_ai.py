# ==========================================================
# Advertisement Monk.AI – Full Ads Intelligence App
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import io
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

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    border: 1px solid #E2E8F0;
}

[data-testid="stChatMessageContainer"] {
    background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
    border-radius: 30px;
    padding: 28px;
    border: 2px solid #7DD3FC;
    margin-bottom: 16px;
    font-size: 16px;
}

.stChatFloatingInputContainer {
    background: transparent;
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
        "acos": [(0.2, "Excellent"), (0.3, "Good"), (0.4, "Acceptable"), (0.6, "Poor")],
        "neg_clicks": 15
    },
    "qc": {
        "roas": [(4, "Excellent"), (3, "Good"), (2, "Acceptable"), (1.2, "Poor")],
        "ctr_min": 0.012
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
    "sales": ["7 day total sales", "sales", "gmv"],
    "orders": ["7 day total orders", "orders"]
}

# ----------------------------------------------------------
# HEADER AUTO-DETECTION
# ----------------------------------------------------------
def detect_header_row(file):
    preview = pd.read_csv(file, header=None, nrows=30, encoding_errors="ignore")
    for i, row in preview.iterrows():
        joined = " ".join(map(str, row.values)).lower()
        if "impressions" in joined and ("click" in joined or "spend" in joined):
            return i
    return 0

# ----------------------------------------------------------
# LOAD FILE (CSV / XLSX)
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
# NORMALIZE COLUMNS
# ----------------------------------------------------------
def normalize_columns(df):
    norm = {}
    for canon, variants in FIELD_MAP.items():
        for col in df.columns:
            for v in variants:
                if v in col:
                    norm[canon] = col
    return df.rename(columns=norm)

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
# METRICS
# ----------------------------------------------------------
def add_metrics(df):
    if "impressions" in df.columns and "clicks" in df.columns:
        df["ctr"] = df["clicks"] / df["impressions"]
    if "clicks" in df.columns and "spend" in df.columns:
        df["cpc"] = df["spend"] / df["clicks"]
    if "sales" in df.columns and "spend" in df.columns:
        df["roas"] = df["sales"] / df["spend"]
        df["acos"] = df["spend"] / df["sales"]
    return df

# ----------------------------------------------------------
# DECISION ENGINE
# ----------------------------------------------------------
def decisions(df, report_type):
    df["decision"] = "Maintain"
    if report_type == "amazon_search_term":
        df.loc[(df["clicks"] >= BENCHMARKS["amazon"]["neg_clicks"]) & (df["orders"] == 0),
               "decision"] = "NEGATIVE"
        df.loc[df["acos"] <= 0.25, "decision"] = "SCALE"
        df.loc[df["acos"] >= 0.6, "decision"] = "PAUSE"
    if report_type == "quick_commerce":
        df.loc[df["roas"] >= 3, "decision"] = "SCALE"
        df.loc[(df["ctr"] < BENCHMARKS["qc"]["ctr_min"]), "decision"] = "OPTIMIZE"
        df.loc[df["roas"] < 1.2, "decision"] = "PAUSE"
    return df

# ----------------------------------------------------------
# AI EXPLAINER
# ----------------------------------------------------------
def ask_ai(question, context):
    prompt = f"""
You are an Ads Optimization Assistant.
Use ONLY the provided data summary.
Do not invent numbers.

DATA SUMMARY:
{context}

QUESTION:
{question}

Answer with clear actions and reasoning.
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
    df = decisions(df, report_type)

    st.subheader("📊 Performance Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Spend", f"{df['spend'].sum():,.2f}")
    c2.metric("Sales / GMV", f"{df['sales'].sum():,.2f}")
    c3.metric("ROAS", f"{(df['sales'].sum()/df['spend'].sum()):.2f}")

    st.subheader("📈 Distribution")
    st.plotly_chart(
        px.histogram(df, x="decision", color="decision"),
        use_container_width=True
    )

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # ------------------------------------------------------
    # INTERACTIVE AI CHAT
    # ------------------------------------------------------
    st.subheader("💬 Ads Intelligence Assistant")

    context = df.groupby("decision").agg({
        "spend": "sum",
        "sales": "sum",
        "orders": "sum"
    }).reset_index().to_string(index=False)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about optimizations, negatives, scaling decisions…")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            answer = ask_ai(question, context)
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
