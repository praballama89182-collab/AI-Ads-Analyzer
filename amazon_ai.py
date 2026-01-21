import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import re

# --- 🧘 ZEN PAGE CONFIGURATION ---
st.set_page_config(page_title="Advertisement Monk.AI", page_icon="🧘", layout="wide")

# --- 🎨 SKY BLUE INTERACTIVE UI STYLING ---
st.markdown("""
    <style>
    .main { background-color: #F0F9FF; }
    .stMetric { background-color: #FFFFFF; border: 2px solid #BAE6FD; border-radius: 18px; padding: 25px; }
    
    /* Curved Sky Blue Interactive Box for AI Chat */
    [data-testid="stChatMessageContainer"] {
        background-color: #E0F2FE !important; 
        border-radius: 35px !important;
        padding: 30px !important;
        border: 3px solid #7DD3FC !important;
        margin-bottom: 25px !important;
        box-shadow: 0 10px 25px -5px rgba(12, 74, 110, 0.1);
        width: 85% !important;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stChatFloatingInputContainer { background-color: transparent; }
    h1, h2, h3 { color: #0369A1; font-family: 'Segoe UI', sans-serif; font-weight: 800; }
    .stTabs [data-baseweb="tab-list"] { gap: 30px; }
    .stTabs [data-baseweb="tab"] { background-color: #E0F2FE; border-radius: 12px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 ADS OPTIMIZATION BENCHMARKS (SOP) ---
BENCHMARKS = {
    "acos": {"excellent": 20, "good": 30, "acceptable": 40, "poor": 60},
    "roas": {"excellent": 5.0, "good": 3.5, "acceptable": 2.5, "poor": 1.5},
    "neg_rules": {"clicks": 15, "acos": 60}
}

# --- 🛠️ PRECISION LOAD ENGINE ---
def robust_load(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
    except:
        content = file.getvalue().decode('latin1', errors='ignore').splitlines()
    
    # 1. Skip Metadata (Looking for real headers)
    skip = 0
    header_keywords = ["METRICS_DATE", "CAMPAIGN NAME", "DATE", "SEARCH TERM", "ROW LABELS"]
    for i, line in enumerate(content[:25]):
        if any(k in line.upper() for k in header_keywords):
            skip = i
            break
            
    file.seek(0)
    # Using engine='python' for malformed CSVs & buffer issues
    df = pd.read_csv(file, skiprows=skip, encoding_errors='ignore', engine='python')
    df.columns = df.columns.str.strip()
    
    # 2. Deep Scan Mapping (Including trailing spaces in Amazon reports)
    MAP = {
        'spend': ['TOTAL_BUDGET_BURNT', 'Spend', 'Cost', 'Ad Spend', 'Sum of Spend', 'Cost Per Click (CPC)'],
        'sales': ['TOTAL_GMV', '7 Day Total Sales', 'Sales', 'Revenue', 'Sum of 7 Day Total Sales', '7 Day Total Sales '],
        'orders': ['TOTAL_CONVERSIONS', 'Orders', 'Units', 'Sum of 7 Day Total Orders', '7 Day Total Orders (#)'],
        'clicks': ['TOTAL_CLICKS', 'Clicks', 'Sum of Clicks'],
        'imps': ['TOTAL_IMPRESSIONS', 'Impressions', 'Sum of Impressions'],
        'term': ['KEYWORD', 'Customer Search Term', 'Search Term', 'Row Labels', 'Targeting'],
        'camp': ['CAMPAIGN_NAME', 'Campaign Name', 'Campaign'],
        'date': ['METRICS_DATE', 'Date', 'Reporting Date']
    }
    
    for std, vars in MAP.items():
        for col in df.columns:
            if any(v.lower() == col.lower() or v.lower() in col.lower() for v in vars):
                df = df.rename(columns={col: std})
                break
    
    # 3. Secure Cleaning
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    
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
        
        # --- DASHBOARD NAVIGATION ---
        tab_port, tab_camp, tab_term = st.tabs(["🌐 Portfolio", "🎯 Campaigns", "🔍 Search Terms"])
        
        # TAB 1: PORTFOLIO & TRENDS
        with tab_port:
            st.header("Executive Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Spend", f"₹{df['spend'].sum():,.1f}")
            m2.metric("Total Sales", f"₹{df['sales'].sum():,.1f}")
            total_acos = (df['spend'].sum()/df['sales'].sum()*100) if df['sales'].sum() > 0 else 0
            m3.metric("Account ACoS", f"{total_acos:.1f}%")
            m4.metric("Average CTR", f"{(df['clicks'].sum()/df['imps'].sum()*100):.2f}%" if df['imps'].sum() > 0 else "0%")

            if 'date' in df.columns and len(df) > 0:
                st.subheader("Weekly Ad Efficiency (Sunday-Monday)")
                days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                weekly = df.groupby(df['date'].dt.day_name()).agg({'spend':'sum', 'sales':'sum'}).reindex(days).reset_index()
                weekly.columns = ['Day', 'Spend', 'Sales'] # Fix for px.bar ValueErrors
                
                fig = px.bar(weekly, x='Day', y=['Spend', 'Sales'], barmode='group',
                             color_discrete_map={'Spend': '#BAE6FD', 'Sales': '#BBF7D0'},
                             labels={'value': 'Amount (₹)'}, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        # TAB 2: CAMPAIGN WISE PERFORMANCE
        with tab_camp:
            st.subheader("Campaign Performance Hub")
            camp_view = df.groupby('camp').agg({'spend':'sum', 'sales':'sum', 'acos':'mean', 'clicks':'sum', 'ctr':'mean'}).sort_values('sales', ascending=False)
            st.dataframe(camp_view.style.format({'spend': '₹{:.1f}', 'sales': '₹{:.1f}', 'acos': '{:.1f}%', 'ctr': '{:.2f}%'}), use_container_width=True)

        # TAB 3: SEARCH TERM INTELLIGENCE
        with tab_term:
            if 'term' in df.columns:
                st.subheader("Search Term Efficiency")
                term_view = df.groupby('term').agg({'spend':'sum', 'sales':'sum', 'acos':'mean', 'clicks':'sum'}).sort_values('spend', ascending=False).head(50)
                st.dataframe(term_view.style.format({'acos': '{:.1f}%'}), use_container_width=True)
            else:
                st.info("No Search Term data detected.")

        # --- 💬 SKY BLUE ROUNDED AI INTERACTIVE BOX ---
        st.divider()
        st.subheader("🧘 Advertisement Monk AI Consultant")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Show campaigns with ACoS less than 30%"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                # 1. Performance Logic (Less than / More than ACoS)
                if "acos" in prompt.lower() and any(x in prompt for x in ["<", ">", "less", "more", "above", "below"]):
                    try:
                        val = [int(s) for s in re.findall(r'\d+', prompt)][0]
                        filtered = df[df['acos'] < val] if any(x in prompt.lower() for x in ["less", "below", "<"]) else df[df['acos'] > val]
                        # Final results table as requested
                        res = filtered[['camp', 'acos', 'ctr', 'clicks', 'sales', 'spend']].drop_duplicates().sort_values('acos')
                        st.markdown(f"**The Monk found {len(res)} campaigns matching `{prompt}`:**")
                        st.dataframe(res.style.format({'acos': '{:.1f}%', 'ctr': '{:.2f}%'}), use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": f"Evaluated and displayed {len(res)} campaigns."})
                    except:
                        st.error("Please specify a numeric threshold (e.g., 'less than 30%').")
                
                # 2. Optimization Logic (Recommendations)
                elif any(x in prompt.lower() for x in ["recommend", "optimize", "advice"]):
                    context = df.nlargest(10, 'spend').to_string()
                    response = ollama.chat(model='llama3.2', messages=[
                        {"role": "system", "content": f"You are Advertisement Monk.AI. Use benchmarks: {BENCHMARKS}. Analyze data and provide 3-4 specific optimization steps."},
                        {"role": "user", "content": f"Data Summary:\n{context}\n\nQuestion: {prompt}"}
                    ])
                    st.markdown(response['message']['content'])
                    st.session_state.messages.append({"role": "assistant", "content": response['message']['content']})
                else:
                    st.write("I am ready to audit your campaign ACoS or search terms. Ask me to find specific performance ranges or optimization tips!")

    else:
        st.info("🙏 Namaste. Upload your Amazon or Quick-Commerce report to activate the Monk.")

if __name__ == "__main__":
    main()
