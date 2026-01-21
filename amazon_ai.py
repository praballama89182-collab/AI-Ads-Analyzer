# ==========================================================
# amazon_ai — Amazon Search Term AI (RAW-COLUMN SAFE)
# ==========================================================

import streamlit as st
import pandas as pd
import ollama

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
    "acos": {
        "good": 0.30,
        "poor": 0.60
    },
    "roas": {
        "good": 3.5
    }
}

# ----------------------------------------------------------
# REQUIRED AMAZON SEARCH TERM COLUMNS (STRIPPED)
# ----------------------------------------------------------
REQUIRED_COLUMNS = {
    "Date",
    "Campaign Name",
    "Customer Search Term",
    "Impressions",
    "Clicks",
    "Spend",
    "7 Day Total Sales",
    "7 Day Total Orders (#)"
}

# ----------------------------------------------------------
# LOAD FILE
# ----------------------------------------------------------
def load_excel(upload):
    df = pd.read_excel(upload)

    # 🔑 CRITICAL FIX: strip column names
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    return df

# ----------------------------------------------------------
# METRICS (RECALCULATED, SAFE)
# ----------------------------------------------------------
def add_metrics(df):
    numeric_cols = [
        "Impressions",
        "Clicks",
        "Spend",
        "7 Day Total Sales",
        "7 Day Total Orders (#)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ROAS_calc"] = df["7 Day Total Sales"] / df["Spend"].replace(0, pd.NA)
    df["ACOS_calc"] = df["Spend"] / df["7 Day Total Sales"].replace(0, pd.NA)
    df["CTR_calc"] = df["Clicks"] / df["Impressions"].replace(0, pd.NA)
    df["CPC_calc"] = df["Spend"] / df["Clicks"].replace(0, pd.NA)

    return df

# ----------------------------------------------------------
# DECISION ENGINE
# ----------------------------------------------------------
def apply_decisions(df):
    df["Decision"] = "Maintain"

    df.loc[
        (df["Clicks"] >= 15) & (df["7 Day Total Orders (#)"] == 0),
        "Decision"
    ] = "NEGATIVE"

    df.loc[df["ACOS_calc"] <= BENCHMARKS["acos"]["good"], "Decision"] = "SCALE"
    df.loc[df["ACOS_calc"] >= BENCHMARKS["acos"]["poor"], "Decision"] = "PAUSE"

    return df

# ----------------------------------------------------------
# AI CHAT (FAIL-SAFE)
# ----------------------------------------------------------
def ask_ai(question, context):
    prompt = f"""
You are an Amazon Ads Optimization Assistant.

Benchmarks:
- Good ACOS <= {BENCHMARKS['acos']['good']}
- Poor ACOS >= {BENCHMARKS['acos']['poor']}
- Good ROAS >= {BENCHMARKS['roas']['good']}

Use ONLY the provided data.
Do not invent metrics.

DATA:
{context}

QUESTION:
{question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception:
        return (
            "⚠️ AI service unavailable.\n\n"
            "Benchmarks and decisions are still applied.\n"
            "Start Ollama (`ollama serve`) to enable explanations."
        )

# ==========================================================
# UI
# ==========================================================
st.title("🧘 amazon_ai")
st.caption("Amazon Search Term AI — raw-column aware")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = add_metrics(df)
    df = apply_decisions(df)

    # ---------------- METRICS ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Spend", f"{df['Spend'].sum():,.2f}")
    c2.metric("Sales", f"{df['7 Day Total Sales'].sum():,.2f}")
    c3.metric(
        "ROAS",
        f"{df['7 Day Total Sales'].sum() / df['Spend'].sum():.2f}"
        if df["Spend"].sum() > 0 else "N/A"
    )

    # ---------------- TABLE ----------------
    st.subheader("🔍 Search Terms (Top Spend)")
    st.dataframe(
        df.sort_values("Spend", ascending=False)[
            [
                "Customer Search Term",
                "Campaign Name",
                "Spend",
                "Clicks",
                "7 Day Total Sales",
                "Decision"
            ]
        ].head(50),
        use_container_width=True
    )

    # ---------------- AI CHAT ----------------
    st.subheader("💬 AI Optimization Assistant")

    context = (
        df.groupby("Decision")[["Spend", "7 Day Total Sales"]]
        .sum()
        .reset_index()
        .to_string(index=False)
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask about ACOS, ROAS, wasted spend, top search terms…")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            a = ask_ai(q, context)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
