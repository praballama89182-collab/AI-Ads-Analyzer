import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import io

# --- 🧘 PAGE CONFIGURATION ---
st.set_page_config(page_title="Advertisement Monk.AI", page_icon="🧘", layout="wide")

# --- 🎨 ZEN CUSTOM STYLING (Sky Blue & Rounded Design) ---
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 15px; padding: 20px; }
    
    /* Rounded Sky Blue Interactive Chat Container */
    [data-testid="stChatMessageContainer"] {
        background-color: #E0F2FE; 
        border-radius: 30px;
        padding: 25px;
        border: 2px solid #7DD3FC;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .stChatFloatingInputContainer { background-color: transparent; }
    h1, h2, h3 { color: #0C4A6E; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 OPTIMIZATION BENCHMARKS (INTEGRATED) ---
BENCHMARKS = {
    "acos": {"excellent": 20, "good": 30, "acceptable": 40, "poor": 60},
    "roas": {"excellent": 5.0, "good": 3.5, "acceptable": 2.5, "poor": 1.5},
    "neg_rules": {"clicks": 15, "acos": 60},
    "qc_cpm": {"excellent": 60, "good": 90, "acceptable": 120}
}

# --- 🛠️ ROBUST DATA ENGINE (FIXES BUFFER & KEY ERRORS) ---
def robust_load(file):
    try:
        # Detect Header Row
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
    except:
        content = file.getvalue().decode('latin1', errors='ignore').splitlines()
    
    skip = 0
    for i, line in enumerate(content[:20]):
        # Search for key headers across Amazon, Swiggy, and Summary reports
        if any(k in line.upper() for k in ["METRICS_DATE", "CAMPAIGN NAME", "DATE", "ROW LABELS"]):
            skip = i
            break
            
    file.seek(0)
    # Using engine='python' to fix C-based Buffer Overflow errors
    df = pd.read_csv(file, skiprows=skip, encoding_errors='ignore', engine='python')
    df.columns = df.columns.str.strip()
    
    # Precision Mapping Dictionary
    MAP = {
        'spend': ['TOTAL_BUDGET_BURNT', 'Sum of Spend', 'Spend', 'Ad Spend', 'Cost'],
        'sales': ['TOTAL_GMV', 'Sum of 7 Day Total Sales', '7 Day Total Sales', 'Sales', 'Revenue'],
        'orders': ['TOTAL_CONVERSIONS', 'Sum of 7 Day Total Orders', '7 Day Total Orders', 'Orders'],
        'clicks': ['TOTAL_CLICKS', 'Sum of Clicks', 'Clicks'],
        'term': ['KEYWORD', 'Customer Search Term', 'Search Term', 'Row Labels', 'Targeting'],
        'camp': ['CAMPAIGN_NAME', 'Campaign Name', 'Campaign'],
        'date': ['METRICS_DATE', 'Date'],
        'imps': ['TOTAL_IMPRESSIONS', 'Impressions', 'Sum of Impressions']
    }
    
    for standard, variations in MAP.items():
        for col in df.columns:
            if any(v.lower() == col.lower() or v.lower() in col.lower() for v in variations):
                df = df.rename(columns={col: standard})
                break
    
    # Ensure essential columns exist to prevent KeyError
    for col in ['spend', 'sales', 'orders', 'clicks', 'imps']:
        if col not in df.columns: df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculated Metrics
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
        
        # --- MULTI-DASHBOARD TABS ---
        t_overview, t_camp, t_search = st.tabs(["🌎 Portfolio Overview", "📈 Campaign Analytics", "🔍 Search Term Performance"])
        
        # 1. Portfolio Overview
        with t_overview:
            st.header("Executive Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Spend", f"₹{df['spend'].sum():,.0f}")
            c2.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
            total_acos = (df['spend'].sum()/df['sales'].sum()*100) if df['sales'].sum() > 0 else 0
            c3.metric("Account ACoS", f"{total_acos:.1f}%")
            c4.metric("Avg CTR", f"{(df['clicks'].sum()/df['imps'].sum()*100):.2f}%" if df['imps'].sum() > 0 else "0%")

            # Weekly Trends (Sun-Mon)
            if 'date' in df.columns:
                st.subheader("Weekly Spend vs Sales")
                df['date'] = pd.to_datetime(df['date'])
                days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                weekly = df.groupby(df['date'].dt.day_name()).agg({'spend':'sum', 'sales':'sum'}).reindex(days).reset_index()
                
                fig = px.bar(weekly, x='date', y=['spend', 'sales'], barmode='group',
                             color_discrete_map={'spend': '#93C5FD', 'sales': '#A7F3D0'}, # Ice Blue & Mint
                             labels={'value': 'Amount (₹)', 'date': 'Day'})
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("Cumulative Sales Efficiency")
                st.line_chart(df['sales'], color="#7DD3FC")

        # 2. Campaign Dashboard
        with t_camp:
            st.subheader("Top Campaign Performance")
            if 'camp' in df.columns:
                camp_df = df.groupby('camp').agg({'spend':'sum', 'sales':'sum', 'clicks':'sum', 'acos':'mean'}).sort_values('sales', ascending=False)
                st.dataframe(camp_df, use_container_width=True)

        # 3. Search Term Dashboard
        with t_search:
            st.subheader("Search Term Intelligence")
            if 'term' in df.columns:
                term_df = df.groupby('term').agg({'spend':'sum', 'sales':'sum', 'acos':'mean', 'clicks':'sum'}).sort_values('spend', ascending=False).head(50)
                st.dataframe(term_df, use_container_width=True)
            else:
                st.info("No Search Term data available in this report.")

        # --- 💬 SKY BLUE INTERACTIVE CHAT ---
        st.divider()
        st.subheader("💬 Consult Advertisement Monk.AI")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Show campaigns with ACoS less than 30%"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # Analysis Logic
            with st.chat_message("assistant"):
                # Handle Threshold Queries (ACoS/Sales)
                if "acos" in prompt.lower() and any(x in prompt.lower() for x in ["less", "more", ">", "<"]):
                    try:
                        val = [int(s) for s in prompt.replace('%','').split() if s.isdigit()][0]
                        filtered = df[df['acos'] < val] if ("less" in prompt.lower() or "<" in prompt) else df[df['acos'] > val]
                        # Final results with requested metrics
                        res = filtered[['camp', 'acos', 'ctr', 'clicks', 'sales', 'spend']].drop_duplicates()
                        st.markdown(f"**Found {len(res)} campaigns matching your criteria:**")
                        st.dataframe(res.sort_values('acos'), use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": f"Displayed {len(res)} campaigns in the result table."})
                    except:
                        st.error("Monk couldn't parse the number. Try: 'ACoS less than 30%'")
                
                # Handle Optimization/Recommendations using Benchmarks
                elif any(x in prompt.lower() for x in ["recommend", "optimize", "advice"]):
                    context = df.nlargest(10, 'spend').to_string()
                    response = ollama.chat(model='llama3.2', messages=[
                        {"role": "system", "content": f"You are Advertisement Monk.AI. Use these benchmarks: {BENCHMARKS}. Analyze data and provide 3-4 specific optimization steps."},
                        {"role": "user", "content": f"Data Summary:\n{context}\n\nQuestion: {prompt}"}
                    ])
                    st.markdown(response['message']['content'])
                    st.session_state.messages.append({"role": "assistant", "content": response['message']['content']})
                else:
                    st.write("I am ready to analyze your campaign metrics or search terms. Ask me to find specific ACoS ranges or optimization tips!")

    else:
        st.info("🙏 Namaste. Upload a report from Amazon, Swiggy, or Zepto to begin.")

if __name__ == "__main__":
    main()
