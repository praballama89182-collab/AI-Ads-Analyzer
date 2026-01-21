# ==========================================================
# amazon_ai — Amazon Search Term AI (FINAL, STABLE)
# ==========================================================

import streamlit as st
import pandas as pd
import ollama

# ==========================================================
# CONFIG
# ==========================================================

AI_MODEL = "llama3.2:latest"   # MUST match `ollama list`

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

# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="amazon_ai",
    page_icon="🧘",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #F8FAFC; }
[data-testid="stChatMessageContainer"] {
    background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
    border-radius: 28px;
    padding: 22px;
    border: 2px solid #7DD3FC;
}
textarea {
    border-radius: 26px !important;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# AI HEALTH CHECK (CRITICAL)
# ==========================================================

@st.cache_resource
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

# ==========================================================
# DATA FUNCTIONS
# ==========================================================

def load_excel(upload):
    df = pd.read_excel(upload)
    df.columns = [c.strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    return df


def add_metrics(df):
    numeric_cols = [
        "Impressions",
        "Clicks",
        "Spend",
        "7 Day Advertised SKU Sales",
        "7 Day Total Orders (#)"
    ]

    for col in numeric_cols:
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

# ==========================================================
# AI PROMPT FUNCTION
# ==========================================================

def ask_ai(question, context):
    if not AI_READY:
        return (
            "⚠️ AI service unavailable.\n\n"
            "Analytics and benchmarks are active.\n"
            "Ensure Ollama is running and restart Streamlit."
        )

    prompt = AMAZON_LLM_PROMPT.format(
        benchmarks=BENCHMARKS,
        context=context,
        question=question
    )

    response = ollama.chat(
        model=AI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# ==========================================================
# UI
# ==========================================================

st.title("🧘 amazon_ai")
st.caption("Amazon Search Term Optimization — Benchmark & AI Assisted")

st.sidebar.markdown("### 🧠 AI Status")
if AI_READY:
    st.sidebar.success("AI connected")
else:
    st.sidebar.warning("AI offline (analytics still work)")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = add_metrics(df)
    df = apply_decisions(df)

    # ---------------- OVERVIEW ----------------
    total_spend = df["Spend"].sum()
    total_sales = df["7 Day Advertised SKU Sales"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spend", f"{total_spend:,.2f}")
    c2.metric("Advertised Sales", f"{total_sales:,.2f}")
    c3.metric("ROAS", f"{total_sales/total_spend:.2f}" if total_spend else "N/A")
    c4.metric("ACOS", f"{total_spend/total_sales:.2%}" if total_sales else "N/A")

    # ---------------- TOP CAMPAIGNS ----------------
    st.subheader("🏆 Top Grossing Campaigns")

    camp = (
        df.groupby("Campaign Name")
        .agg({"Spend": "sum", "7 Day Advertised SKU Sales": "sum"})
        .reset_index()
    )

    camp["ROAS"] = camp["7 Day Advertised SKU Sales"] / camp["Spend"].replace(0, pd.NA)
    camp["ACOS"] = camp["Spend"] / camp["7 Day Advertised SKU Sales"].replace(0, pd.NA)

    st.dataframe(
        camp.sort_values("7 Day Advertised SKU Sales", ascending=False).head(10),
        use_container_width=True
    )

    # ---------------- SEARCH TERMS ----------------
    st.subheader("🔍 Search Terms")

    st.dataframe(
        df.sort_values("Spend", ascending=False)[
            [
                "Customer Search Term",
                "Campaign Name",
                "Spend",
                "7 Day Advertised SKU Sales",
                "ROAS",
                "ACOS",
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
        "Ask about wasted spend, negatives, scaling, ROAS, ACOS…"
    )

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            a = ask_ai(q, context)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
