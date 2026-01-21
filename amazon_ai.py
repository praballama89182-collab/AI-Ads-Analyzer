# ==========================================================
# amazon_ai — Amazon Search Term AI (FINAL + INSIGHTS)
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
    "acos": {"good": 0.30, "poor": 0.60},
    "roas": {"good": 3.5}
}

# ----------------------------------------------------------
# REQUIRED RAW COLUMNS
# ----------------------------------------------------------
REQUIRED_COLUMNS = {
    "Campaign Name",
    "Customer Search Term",
    "Impressions",
    "Clicks",
    "Spend",
    "7 Day Advertised SKU Sales",
    "7 Day Total Orders (#)"
}

# ----------------------------------------------------------
# LOAD FILE
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
# DECISIONS
# ----------------------------------------------------------
def apply_decisions(df):
    df["Decision"] = "Maintain"

    df.loc[
        (df["Clicks"] >= 15) &
        (df["7 Day Total Orders (#)"] == 0),
        "Decision"
    ] = "NEGATIVE"

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
- Good ACOS <= {BENCHMARKS['acos']['good']}
- Poor ACOS >= {BENCHMARKS['acos']['poor']}
- Good ROAS >= {BENCHMARKS['roas']['good']}

Sales used: 7 Day Advertised SKU Sales only.

DATA:
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
            "Restart Streamlit after confirming Ollama is running."
        )

# ==========================================================
# UI
# ==========================================================
st.title("🧘 amazon_ai")
st.caption("Amazon Search Term AI — benchmarks, campaigns & search terms")

uploaded = st.file_uploader(
    "Upload Amazon Sponsored Products – Search Term Report (.xlsx)",
    type=["xlsx"]
)

if uploaded:
    df = load_excel(uploaded)
    df = add_metrics(df)
    df = apply_decisions(df)

    # ------------------------------------------------------
    # OVERVIEW METRICS
    # ------------------------------------------------------
    total_spend = df["Spend"].sum()
    total_sales = df["7 Day Advertised SKU Sales"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spend", f"{total_spend:,.2f}")
    c2.metric("Advertised Sales", f"{total_sales:,.2f}")
    c3.metric("ROAS", f"{total_sales / total_spend:.2f}" if total_spend > 0 else "N/A")
    c4.metric("ACOS", f"{total_spend / total_sales:.2%}" if total_sales > 0 else "N/A")

    # ------------------------------------------------------
    # TOP GROSSING CAMPAIGNS
    # ------------------------------------------------------
    st.subheader("🏆 Top Grossing Campaigns")

    campaign_df = (
        df.groupby("Campaign Name")
        .agg({
            "Spend": "sum",
            "7 Day Advertised SKU Sales": "sum"
        })
        .reset_index()
    )

    campaign_df["ROAS"] = (
        campaign_df["7 Day Advertised SKU Sales"] /
        campaign_df["Spend"].replace(0, pd.NA)
    )
    campaign_df["ACOS"] = (
        campaign_df["Spend"] /
        campaign_df["7 Day Advertised SKU Sales"].replace(0, pd.NA)
    )

    st.dataframe(
        campaign_df.sort_values("7 Day Advertised SKU Sales", ascending=False).head(10),
        use_container_width=True
    )

    # ------------------------------------------------------
    # SEARCH TERMS
    # ------------------------------------------------------
    st.subheader("🔍 Search Terms (Top Spend)")

    st.dataframe(
        df.sort_values("Spend", ascending=False)[
            [
                "Customer Search Term",
                "Campaign Name",
                "Spend",
                "7 Day Advertised SKU Sales",
                "ROAS_calc",
                "ACOS_calc",
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
        "Ask about ACOS, ROAS, top campaigns, wasted spend, next actions…"
    )

    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            a = ask_ai(q, context)
            st.markdown(a)
            st.session_state.chat.append({"role": "assistant", "content": a})
