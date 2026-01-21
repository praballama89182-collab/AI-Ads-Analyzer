# ==========================================================
# amazon_ai — Amazon Ads AI Optimization Assistant
# Local Streamlit + Ollama (FINAL)
# ==========================================================

import streamlit as st
import pandas as pd
import ollama

# ===================== CONFIG =============================

AI_MODEL = "llama3.2:latest"

BENCHMARKS = {
    "acos": {
        "excellent": 0.20,
        "good": 0.30,
        "poor": 0.60
    },
    "roas": {
        "good": 3.5
    },
    "negative_clicks": 15
}

REQUIRED_COLUMNS = {
    "Campaign Name",
    "Customer Search Term",
    "Impressions",
    "Clicks",
    "Spend",
    "7 Day Advertised SKU Sales",
    "7 Day Total Orders (#)"
}

# ===================== PAGE ===============================

st.set_page_config(
    page_title="Amazon AI Ads Optimizer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #F8FAFC; }

/* AI Chat Bubble */
[data-testid="stChatMessageContainer"] {
    background: linear-gradient(135deg, #ECFEFF, #E0F2FE);
    border-radius: 32px;
    padding: 24px;
    border: 2px solid #38BDF8;
    box-shadow: 0 10px 25px rgba(56,189,248,0.25);
    margin-bottom: 18px;
}

/* Input */
textarea {
    border-radius: 28px !important;
    padding: 18px !important;
    border: 2px solid #38BDF8 !important;
}

/* Suggested Questions */
.suggest {
    background-color: #E0F2FE;
    padding: 12px 18px;
    border-radius: 20px;
    display: inline-block;
    margin: 6px 6px 6px 0;
    font-size: 14px;
    color: #0369A1;
}
</style>
""", unsafe_allow_html=True)

# ===================== PROMPT =============================

AMAZON_LLM_PROMPT = """
You are an expert Amazon Ads Optimization Assistant.

STRICT RULES:
- Use ONLY the data provided
- Do NOT invent metrics
- Respect ACOS and ROAS benchmarks
- Explain insights clearly in human language

BENCHMARKS:
ACOS:
- Excellent <= 0.20
- Good <= 0.30
- Poor >= 0.60

ROAS:
- Good >= 3.5

BUSINESS LOGIC:
- Sales = 7 Day Advertised SKU Sales only
- NEGATIVE = clicks >= 15 and zero orders
- SCALE = ACOS <= good benchmark
- PAUSE = ACOS >= poor benchmark

DATA SUMMARY:
{context}

USER QUESTION:
{question}

Respond in the tone the user asks for.
Give actionable recommendations.
Include search terms and campaigns when relevant.
"""

# ===================== AI CHECK ===========================

def ai_available():
    try:
        ollama.chat(
            model=AI_MODEL,
            messages=[{"role": "user", "content": "ping"}]
        )
        return True
    except Exception:
        return False

AI_READY = ai_available()

# ===================== DATA ===============================

def load_excel(upload):
    df = pd.read_excel(upload)
    df.columns = [c.strip() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    return df

def add_metrics(df):
    for col in REQUIRED_COLUMNS:
        if col not in ["Campaign Name", "Customer Search Term"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ROAS"] = df["7 Day Advertised SKU Sales"] / df["Spend"].replace(0, pd.NA)
    df["ACOS"] = df["Spend"] / df["7 Day Advertised SKU Sales"].replace(0, pd.NA)
    return df

def apply_decisions(df):
    df["Decision"] = "Maintain"

    df.loc[
        (df["Clicks"] >= BENCHMARKS["negative_clicks"]) &
        (df["7 Day Total Orders (#)"] == 0),
        "Decision"
    ] = "NEGATIVE"

    df.loc[df["ACOS"] <= BENCHMARKS["acos"]["good"], "Decision"] = "SCALE"
    df.loc[df["ACOS"] >= BENCHMARKS["acos"]["poor"], "Decision"] = "PAUSE"
    return df

# ===================== AI ================================

def ask_ai(question, context):
    if not AI_READY:
        return "AI is unavailable. Run this app locally with Ollama to enable insights."

    prompt = AMAZON_LLM_PROMPT.format(
        context=context,
        question=question
    )

    response = ollama.chat(
        model=AI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ===================== UI ================================

st.title("🧠 Amazon AI Optimization Assistant")
st.caption("Benchmark-driven Amazon Ads analysis with human-like AI insights")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = apply_decisions(add_metrics(load_excel(uploaded)))

    # -------- Overview --------
    spend = df["Spend"].sum()
    sales = df["7 Day Advertised SKU Sales"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spend", f"{spend:,.2f}")
    c2.metric("Advertised Sales", f"{sales:,.2f}")
    c3.metric("ROAS", f"{sales/spend:.2f}" if spend else "N/A")
    c4.metric("ACOS", f"{spend/sales:.2%}" if sales else "N/A")

    # -------- Campaigns --------
    st.subheader("🏆 Top Campaigns")
    camp = df.groupby("Campaign Name")[["Spend", "7 Day Advertised SKU Sales"]].sum().reset_index()
    camp["ROAS"] = camp["7 Day Advertised SKU Sales"] / camp["Spend"].replace(0, pd.NA)
    camp["ACOS"] = camp["Spend"] / camp["7 Day Advertised SKU Sales"].replace(0, pd.NA)
    st.dataframe(camp.sort_values("7 Day Advertised SKU Sales", ascending=False).head(10), use_container_width=True)

    # -------- Search Terms --------
    st.subheader("🔍 Search Terms Overview")
    st.dataframe(
        df.sort_values("Spend", ascending=False)[
            ["Customer Search Term", "Campaign Name", "Spend",
             "7 Day Advertised SKU Sales", "ROAS", "ACOS", "Decision"]
        ],
        use_container_width=True
    )

    # -------- AI Chat --------
    st.subheader("💬 AI Optimization Assistant")

    st.markdown("""
<div class="suggest">Which search terms are wasted spend?</div>
<div class="suggest">Which campaigns should I scale?</div>
<div class="suggest">Explain performance like ChatGPT</div>
<div class="suggest">Give executive summary</div>
""", unsafe_allow_html=True)

    context = (
        df.groupby("Decision")[["Spend", "7 Day Advertised SKU Sales"]]
        .sum().reset_index().to_string(index=False)
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask anything about ACOS, ROAS, campaigns, search terms…")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            ans = ask_ai(q, context)
            st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})
