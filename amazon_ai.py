import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import io

# --- 🧘 PAGE CONFIGURATION ---
st.set_page_config(page_title="Advertisement Monk.AI", page_icon="🧘", layout="wide")

# --- 🎨 ZEN MASTER STYLING (Sky Blue, Rounded, & Clean) ---
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 15px; padding: 20px; }
    
    /* Rounded Sky Blue Interactive Chat Box */
    [data-testid="stChatMessageContainer"] {
        background-color: #E0F2FE; 
        border-radius: 40px;
        padding: 30px;
        border: 2px solid #7DD3FC;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        width: 90% !important;
        margin-left: 5%;
    }
    
    .stChatFloatingInputContainer { background-color: transparent; }
    h1, h2, h3 { color: #0369A1; font-family: 'Segoe UI', sans-serif; font-weight: 800; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; white-space: pre-wrap; background-color: #F1F5F9; 
        border-radius: 10px; padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 OPTIMIZATION BENCHMARKS ---
BENCHMARKS = {
    "acos": {"excellent": 20, "good": 30, "acceptable": 40, "poor": 60},
    "roas": {"excellent": 5.0, "good": 3.5, "acceptable": 2.5, "poor": 1.5},
    "qc_roas": {"excellent": 4.0, "good": 3.0, "poor": 1.2}
}

# --- 🛠️ MASTER DATA ENGINE (FIXES DATE & BUFFER ERRORS) ---
def robust_load(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
    except:
        content = file.getvalue().decode('latin1', errors='ignore').splitlines()
    
    # 1. Skip metadata by finding the REAL header row
    skip = 0
    header_keywords = ["METRICS_DATE", "CAMPAIGN NAME", "DATE", "SEARCH TERM", "ROW LABELS"]
    for i, line in enumerate(content[:25]):
        if any(k in line.upper() for k in header_keywords):
            skip = i
            break
            
    file.seek(0)
    # Using engine='python' to prevent buffer overflow
    df = pd.read_csv(file, skiprows=skip, encoding_errors='ignore', engine='python')
    df.columns = df.columns.str.strip()
    
    # 2. Precision Mapping
    MAP = {
        'spend': ['TOTAL_BUDGET_BURNT', 'Spend', 'Cost', 'Ad Spend', 'Sum of Spend'],
        'sales': ['TOTAL_GMV', '7 Day Total Sales', 'Sales', 'Revenue', 'Sum of 7 Day Total Sales'],
        'orders': ['TOTAL_CONVERSIONS', '7 Day Total Orders', 'Orders', 'Sum of 7 Day Total Orders'],
        'clicks': ['TOTAL_CLICKS', 'Clicks', 'Sum of Clicks'],
        'imps': ['TOTAL_IMPRESSIONS', 'Impressions', 'Sum of Impressions'],
        'term': ['KEYWORD', 'Customer Search Term', 'Search Term', 'Row Labels'],
        'camp': ['CAMPAIGN_NAME', 'Campaign Name', 'Campaign'],
        'date': ['METRICS_DATE', 'Date']
    }
    
    for std, vars in MAP.items():
        for col in df.columns:
            if any(v.lower() == col.lower() or v.lower() in col.lower() for v in vars):
                df = df.rename(columns={col: std})
                break
    
    # 3. Clean Date Column (Crucial fix for DateParseError)
    if 'date' in df.columns:
        # Remove any rows where 'date' is a header string like "To Date"
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    
    # Fill missing metrics
    for col in ['spend', 'sales', 'orders', 'clicks', 'imps']:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['acos'] = df.apply(lambda x: (x['spend']/x['sales']*100) if x['sales'] > 0 else 0, axis=1)
    df['ctr'] = df.apply(lambda x: (x['clicks']/x['imps']*100) if x['imps'] > 0 else 0, axis=1)
    
    return df

# --- 🧘 MAIN APP ---
def main():
    if "messages" not in st.session_state: st.session_state.messages = []
    
    st.sidebar.title("🧘 Advertisement Monk.AI")
    uploaded_file = st.sidebar.file_uploader("Upload Ad Report", type=["csv"])
    
    if uploaded_file:
        df = robust_load(uploaded_file)
        
        # --- 3-DASHBOARD SYSTEM ---
        t1, t2, t3 = st.tabs(["🌎 Portfolio Overview", "📉 Campaign Leaderboard", "🔍 Search Term Analysis"])
        
        # Dashboard 1: Portfolio Overview
        with t1:
            st.header("Executive Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Spend", f"₹{df['spend'].sum():,.0f}")
            m2.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
            total_acos = (df['spend'].sum()/df['sales'].sum()*100) if df['sales'].sum() > 0 else 0
            m3.metric("Account ACoS", f"{total_acos:.1f}%")
            m4.metric("Avg CTR", f"{(df['clicks'].sum()/df['imps'].sum()*100):.2f}%" if df['imps'].sum() > 0 else "0%")

            if 'date' in df.columns:
                st.subheader("Weekly Budget Pacing (Sun-Mon)")
                days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                weekly = df.groupby(df['date'].dt.day_name()).agg({'spend':'sum', 'sales':'sum'}).reindex(days).reset_index()
                fig = px.bar(weekly, x='index', y=['spend', 'sales'], barmode='group',
                             color_discrete_map={'spend': '#BAE6FD', 'sales': '#BBF7D0'},
                             labels={'value': 'Amount (₹)', 'index': 'Day'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("Sales Trend")
                st.line_chart(df['sales'], color="#7DD3FC")

        # Dashboard 2: Campaign Leaderboard
        with t2:
            st.subheader("Top Campaign Performance")
            camp_view = df.groupby('camp').agg({'spend':'sum', 'sales':'sum', 'acos':'mean', 'clicks':'sum', 'ctr':'mean'}).sort_values('sales', ascending=False)
            st.dataframe(camp_view.style.format({'spend': '₹{:.0f}', 'sales': '₹{:.0f}', 'acos': '{:.1f}%', 'ctr': '{:.2f}%'}), use_container_width=True)

        # Dashboard 3: Search Term Analysis
        with t3:
            if 'term' in df.columns:
                st.subheader("Granular Keyword Performance")
                term_view = df.groupby('term').agg({'spend':'sum', 'sales':'sum', 'acos':'mean', 'clicks':'sum'}).sort_values('spend', ascending=False).head(50)
                st.dataframe(term_view, use_container_width=True)
            else:
                st.info("No Search Term data available in this file.")

        # --- 💬 SKY BLUE ROUNDED INTERACTIVE AI ---
        st.divider()
        st.subheader("💬 Consult the Monk")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Show campaigns with ACoS less than 30%"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # Analysis Engine
            with st.chat_message("assistant"):
                # Handle Threshold Queries (ACoS < 30% or > 50%)
                if "acos" in prompt.lower() and any(x in prompt for x in ["<", ">", "less", "more"]):
                    try:
                        val = [int(s) for s in prompt.replace('%','').split() if s.isdigit()][0]
                        filtered = df[df['acos'] < val] if ("less" in prompt or "<" in prompt) else df[df['acos'] > val]
                        res = filtered[['camp', 'acos', 'ctr', 'clicks', 'sales', 'spend']].drop_duplicates().sort_values('acos')
                        st.markdown(f"**The Monk found {len(res)} campaigns matching your criteria:**")
                        st.dataframe(res, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": f"Displayed {len(res)} campaigns."})
                    except:
                        st.error("Please specify a numeric ACoS threshold.")
                
                # Handle Recommendations using Benchmarks
                elif any(x in prompt.lower() for x in ["recommend", "optimize", "advice"]):
                    context = df.nlargest(10, 'spend').to_string()
                    response = ollama.chat(model='llama3.2', messages=[
                        {"role": "system", "content": f"You are Advertisement Monk.AI. Use benchmarks: {BENCHMARKS}. Analyze data and give 3 precise actions."},
                        {"role": "user", "content": f"Data Summary:\n{context}\n\nQuestion: {prompt}"}
                    ])
                    st.markdown(response['message']['content'])
                    st.session_state.messages.append({"role": "assistant", "content": response['message']['content']})
                else:
                    st.write("I am ready to analyze your campaign metrics. Ask me to find specific ACoS ranges or optimization advice!")

    else:
        st.info("🙏 Namaste. Upload a report from Amazon, Swiggy, or Zepto to activate the Monk.")

if __name__ == "__main__":
    main()
