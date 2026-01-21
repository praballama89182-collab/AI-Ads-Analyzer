# ==========================================================
# amazon_ai — Amazon Search Term AI (FINAL, STABLE)
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
# UI STYLING (CALM, BIG CHAT)
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
# BENCHMARKS (FROM EARLIER DISCUSSION)
# ----------------------------------------------------------
BENCHMARKS = {
    "acos": {
        "excellent": 0.20,
        "good": 0.30,
        "acceptable": 0.40,
        "poor": 0.60
    },
    "roas": {
        "excellent": 5.0,
        "good": 3.5,
        "acceptable": 2.5,
        "poor": 1.5
    }
}

# ----------------------------------------------------------
# REQUIRED RAW AMAZON COLUMNS
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
# LOAD & VALIDATE
# ----------------------------------------------------------
def load_excel(upload):
    df = pd.read_excel(upload)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()
    return df

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------
def add_metrics(df):
    for col in ["Impressions", "Clicks", "Spend", "7 Day Total Sales", "7 Day Total Orders (#)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ROAS_calc"] = df["7 Day Total Sales"] / df["Spend"].replace(0, pd.NA)
    df["ACOS_calc"] = df["Spend"] / df["7 Day Total Sales"].replace(0, pd.NA)

    return df

# ----------------------------------------------------------
# DECISIONS USING BENCHMARKS
# ----------------------------------------------------------
def apply_decisions(df):
    df["Decision"] = "Maintain"

    # Negative logic
    df.loc[
        (df["Clicks"] >= 15) & (df["7 Day Total Orders (#)"] == 0),
        "Decision"
    ] = "NEGATIVE"

    # Scale / Pause
    df.loc[df["ACOS_calc"] <= BENCHMARKS["acos"]["good"], "Decision"] = "SCALE"
    df.loc[df["ACOS_calc"] >= BENCHMARKS["acos"]["poor"], "Decision"] = "PAUSE"

    return df

# ----------------------------------------------------------
# AI CHAT (SAFE)
# ----------------------------------------------------------
def ask_ai(question, context):
    prompt = f"""
You are an Amazon Ads Optimization Assistant.

Benchmarks:
ACOS Excellent <= {BENCHMARKS['acos']['excellent']}
ACOS Poor >= {BENCHMARKS['acos']['poor']}
ROAS Excellent >= {BENCHMARKS['roas']['excellent']}
ROAS Poor <= {BENCHMARKS['roas']['poor']}

Use ONLY the data provided.
Explain decisions, do not invent metrics.

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
            "⚠️ AI service is currently unavailable.\n\n"
            "Insights are still calculated using benchmarks.\n"
            "Please start Ollama (`ollama serve`) to enable AI explanations."
        )

# ==========================================================
# UI
# ==========================================================
st.title("🧘 amazon_ai")
st.caption("Amazon Search Term AI — benchmark-aware, filterable, explainable")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = add_metrics(df)
    df = apply_decisions(df)

    # ------------------------------------------------------
    # SIDEBAR FILTERS
    # ------------------------------------------------------
    st.sidebar.header("🔎 Filters")

    campaigns = st.sidebar.multiselect(
        "Campaign",
        options=sorted(df["Campaign Name"].unique()),
        default=None
    )

    decision_filter = st.sidebar.multiselect(
        "Decision",
        options=df["Decision"].unique(),
        default=list(df["Decision"].unique())
    )

    min_roas = st.sidebar.slider("Min ROAS", 0.0, 10.0, 0.0)
    max_acos = st.sidebar.slider("Max ACOS", 0.0, 1.0, 1.0)

    wasted_spend_only = st.sidebar.checkbox("Show only wasted spend (Sales = 0)")

    # Apply filters
    if campaigns:
        df = df[df["Campaign Name"].isin(campaigns)]

    df = df[
        (df["Decision"].isin(decision_filter)) &
        (df["ROAS_calc"].fillna(0) >= min_roas) &
        (df["ACOS_calc"].fillna(0) <= max_acos)
    ]

    if wasted_spend_only:
        df = df[(df["Spend"] > 0) & (df["7 Day Total Sales"] == 0)]

    # ------------------------------------------------------
    # SUMMARY METRICS
    # ------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Spend", f"{df['Spend'].sum():,.2f}")
    c2.metric("Sales", f"{df['7 Day Total Sales'].sum():,.2f}")
    c3.metric(
        "ROAS",
        f"{df['7 Day Total Sales'].sum() / df['Spend'].sum():.2f}"
        if df["Spend"].sum() > 0 else "N/A"
    )

    # ------------------------------------------------------
    # WASTED SPEND TABLE
    # ------------------------------------------------------
    st.subheader("💸 Wasted Spend (Search Terms)")
    st.dataframe(
        df.sort_values("Spend", ascending=False)[
            [
                "Customer Search Term",
                "Campaign Name",
                "Spend",
                "Clicks",
                "Decision"
            ]
        ].head(50),
        use_container_width=True
    )

    # ------------------------------------------------------
    # AI CHAT
    # ------------------------------------------------------
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

    q = st.chat_input("Ask about ACOS, ROAS, wasted spend, top search terms, actions…")

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            a = ask_ai(q, context)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
