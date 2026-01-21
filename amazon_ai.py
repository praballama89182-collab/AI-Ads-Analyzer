# ==========================================================
# amazon_ai — Amazon Search Term AI (PRODUCTION FINAL)
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
# BENCHMARKS (LOCKED LOGIC)
# ----------------------------------------------------------
BENCHMARKS = {
    "acos": {
        "excellent": 0.20,
        "good": 0.30,
        "poor": 0.60
    },
    "roas": {
        "excellent": 5.0,
        "good": 3.5,
        "poor": 1.5
    }
}

# ----------------------------------------------------------
# REQUIRED RAW AMAZON COLUMNS (STRIPPED)
# ----------------------------------------------------------
REQUIRED_COLUMNS = {
    "Date",
    "Campaign Name",
    "Customer Search Term",
    "Impressions",
    "Clicks",
    "Spend",
    "7 Day Advertised SKU Sales",
    "7 Day Total Orders (#)"
}

# ----------------------------------------------------------
# LOAD & VALIDATE FILE
# ----------------------------------------------------------
def load_excel(upload):
    df = pd.read_excel(upload)
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    return df

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------
def add_metrics(df):
    for col in [
        "Impressions",
        "Clicks",
        "Spend",
        "7 Day Advertised SKU Sales",
        "7 Day Total Orders (#)"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["ROAS_calc"] = df["7 Day Advertised SKU Sales"] / df["Spend"].replace(0, pd.NA)
    df["ACOS_calc"] = df["Spend"] / df["7 Day Advertised SKU Sales"].replace(0, pd.NA)

    return df

# ----------------------------------------------------------
# DECISION ENGINE
# ----------------------------------------------------------
def apply_decisions(df):
    df["Decision"] = "Maintain"

    # Negative rule
    df.loc[
        (df["Clicks"] >= 15) &
        (df["7 Day Total Orders (#)"] == 0),
        "Decision"
    ] = "NEGATIVE"

    # Scale / Pause
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
ACOS Excellent <= {BENCHMARKS['acos']['excellent']}
ACOS Good <= {BENCHMARKS['acos']['good']}
ACOS Poor >= {BENCHMARKS['acos']['poor']}

ROAS Excellent >= {BENCHMARKS['roas']['excellent']}
ROAS Good >= {BENCHMARKS['roas']['good']}
ROAS Poor <= {BENCHMARKS['roas']['poor']}

Sales definition:
- Using 7 Day Advertised SKU Sales only
- Halo sales excluded

Rules:
- Use ONLY provided data
- Do NOT invent metrics
- Explain decisions clearly

DATA SUMMARY:
{context}

QUESTION:
{question}
"""
    try:
        response = ollama.chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception:
        return (
            "⚠️ AI service unavailable.\n\n"
            "Benchmarks and decisions are still applied.\n"
            "Please ensure Ollama is running."
        )

# ==========================================================
# UI
# ==========================================================
st.title("🧘 amazon_ai")
st.caption("Amazon Search Term AI — benchmark-aware & raw-sheet accurate")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = add_metrics(df)
    df = apply_decisions(df)

    # ---------------- SIDEBAR FILTERS ----------------
    st.sidebar.header("🔎 Filters")

    campaigns = st.sidebar.multiselect(
        "Campaign",
        options=sorted(df["Campaign Name"].unique())
    )

    decisions = st.sidebar.multiselect(
        "Decision",
        options=df["Decision"].unique(),
        default=list(df["Decision"].unique())
    )

    min_roas = st.sidebar.value_slider("Min ROAS", 0.0, 10.0, 0.0)
    max_acos = st.sidebar.value_slider("Max ACOS", 0.0, 1.0, 1.0)

    wasted_only = st.sidebar.checkbox("Show only wasted spend (Sales = 0)")

    if campaigns:
        df = df[df["Campaign Name"].isin(campaigns)]

    df = df[
        (df["Decision"].isin(decisions)) &
        (df["ROAS_calc"].fillna(0) >= min_roas) &
        (df["ACOS_calc"].fillna(0) <= max_acos)
    ]

    if wasted_only:
        df = df[(df["Spend"] > 0) & (df["7 Day Advertised SKU Sales"] == 0)]

    # ---------------- SUMMARY ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Spend", f"{df['Spend'].sum():,.2f}")
    c2.metric("Advertised Sales", f"{df['7 Day Advertised SKU Sales'].sum():,.2f}")
    c3.metric(
        "ROAS",
        f"{df['7 Day Advertised SKU Sales'].sum() / df['Spend'].sum():.2f}"
        if df["Spend"].sum() > 0 else "N/A"
    )

    # ---------------- TABLE ----------------
    st.subheader("💸 Search Terms (Filtered)")
    st.dataframe(
        df.sort_values("Spend", ascending=False)[
            [
                "Customer Search Term",
                "Campaign Name",
                "Spend",
                "Clicks",
                "7 Day Advertised SKU Sales",
                "ACOS_calc",
                "Decision"
            ]
        ],
        use_container_width=True
    )

    # ---------------- AI CHAT ----------------
    st.subheader("💬 AI Optimization Assistant")

    context = (
        df.groupby("Decision")[["Spend", "7 Day Advertised SKU Sales"]]
        .sum()
        .reset_index()
        .to_string(index=False)
    )

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input(
        "Ask about ACOS, ROAS, wasted spend, top search terms, next actions…"
    )

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            a = ask_ai(q, context)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
