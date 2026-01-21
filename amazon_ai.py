import streamlit as st
import pandas as pd
import plotly.express as px
import ollama
import io
import re

# --- 🧘 PAGE CONFIGURATION ---
st.set_page_config(page_title="Advertisement Monk.AI", page_icon="🧘", layout="wide")

# --- 🎨 ZEN MASTER STYLING ---
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #BAE6FD; border-radius: 18px; padding: 25px; }
    
    /* Sky Blue Rounded Chat Box */
    [data-testid="stChatMessageContainer"] {
        background-color: #E0F2FE !important; 
        border-radius: 35px !important;
        padding: 25px !important;
        border: 2px solid #7DD3FC !important;
        margin-bottom: 20px !important;
        width: 85% !important;
        margin-left: auto;
        margin-right: auto;
    }
    
    h1, h2, h3 { color: #0369A1; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 OPTIMIZATION BENCHMARKS (SOP) ---
BENCHMARKS = {
    "acos": {"excellent": 20, "good": 30, "acceptable": 40, "poor": 60},
    "roas": {"excellent": 5.0, "good": 3.5, "acceptable": 2.5, "poor": 1.5},
    "qc_roas": {"excellent": 4.0, "good": 3.0, "poor": 1.2}
}

# --- 🛠️ MULTI-FORMAT LOAD ENGINE (CSV & XLSX) ---
def robust_load(file):
    # Determine file type
    is_excel = file.name.endswith(('.xlsx', '.xls'))
    
    if is_excel:
        # Load Excel directly
        df = pd.read_excel(file)
    else:
        # Load CSV with encoding protection
        try:
            content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        except:
            content = file.getvalue().decode('latin1', errors='ignore').splitlines()
        
        # Detect header row (skipping metadata)
        skip = 0
        header_keywords = ["METRICS_DATE", "CAMPAIGN NAME", "DATE", "SEARCH TERM", "ROW LABELS"]
        for i, line in enumerate(content[:25]):
            if any(k in line.upper() for k in header_keywords):
                skip = i
                break
        file.seek(0)
        df = pd.read_csv(file, skiprows=skip, encoding_errors='ignore', engine='python')

    df.columns = df.columns.str.strip()
    
    # Precision Mapping (Amazon + Swiggy + Blinkit)
    MAP = {
        'spend': ['TOTAL_BUDGET_BURNT', 'Spend', 'Cost', 'Ad Spend', 'Sum of Spend'],
        'sales': ['TOTAL_GMV', '7 Day Total Sales', 'Sales', 'Revenue', 'Sum of 7 Day Total Sales', '7 Day Total Sales '],
        'orders': ['TOTAL_CONVERSIONS', 'Orders', 'Units', 'Sum of 7 Day Total Orders', '7 Day Total Orders (#)'],
        'clicks': ['TOTAL_CLICKS', 'Clicks', 'Sum of Clicks'],
        'imps': ['TOTAL_IMPRESSIONS', 'Impressions', 'Sum of Impressions'],
        'term': ['KEYWORD', 'Customer Search Term', 'Search Term', 'Row Labels'],
        'camp': ['CAMPAIGN_NAME', 'Campaign Name', 'Campaign'],
        'date': ['METRICS_DATE', 'Date', 'Reporting Date']
    }
    
    for std, vars in MAP.items():
        for col in df.columns:
            if any(v.lower() == col.lower() or v.lower() in col.lower() for v in vars):
                df = df.rename(columns={col: std})
                break
    
    # Cleaning
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
    # UPDATED: Allowed both CSV and XLSX
    uploaded_file = st.sidebar.file_uploader("Upload Ad Report", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = robust_load(uploaded_file)
        
        # --- DASHBOARD NAVIGATION ---
        tab1, tab2, tab3 = st.tabs(["🌎 Portfolio", "📈 Campaigns", "🔍 Keywords"])
        
        with tab1:
            st.header("Executive Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Spend", f"₹{df['spend'].sum():,.1f}")
            m2.metric("Total Sales", f"₹{df['sales'].sum():,.1f}")
            total_acos = (df['spend'].sum()/df['sales'].sum()*100) if df['sales'].sum() > 0 else 0
            m3.metric("Account ACoS", f"{total_acos:.1f}%")
            m4.metric("Average CTR", f"{(df['clicks'].sum()/df['imps'].sum()*100):.2f}%" if df['imps'].sum() > 0 else "0%")

            if 'date' in df.columns:
                st.subheader("Weekly Ad Efficiency (Sunday-Monday)")
                days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                weekly = df.groupby(df['date'].dt.day_name()).agg({'spend':'sum', 'sales':'sum'}).reindex(days).reset_index()
                fig = px.bar(weekly, x='index', y=['spend', 'sales'], barmode='group',
                             color_discrete_map={'spend': '#BAE6FD', 'sales': '#BBF7D0'},
                             labels={'value': 'Amount (₹)', 'index': 'Day'})
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Campaign Performance Hub")
            camp_view = df.groupby('camp').agg({'spend':'sum', 'sales':'sum', 'acos':'mean', 'clicks':'sum'}).sort_values('sales', ascending=False)
            st.dataframe(camp_view.style.format({'acos': '{:.1f}%'}), use_container_width=True)

        with tab3:
            if 'term' in df.columns:
                st.subheader("Search Term Intelligence")
                term_view = df.groupby('term').agg({'spend':'sum', 'sales':'sum', 'acos':'mean'}).sort_values('spend', ascending=False).head(50)
                st.dataframe(term_view.style.format({'acos': '{:.1f}%'}), use_container_width=True)

        # --- 💬 SKY BLUE ROUNDED AI ---
        st.divider()
        st.subheader("🧘 Consult the Monk")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Show campaigns with ACoS less than 30%"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                # Handle Performance Queries
                if "acos" in prompt.lower() and any(x in prompt for x in ["<", ">", "less", "more"]):
                    try:
                        val = [int(s) for s in re.findall(r'\d+', prompt)][0]
                        filtered = df[df['acos'] < val] if any(x in prompt.lower() for x in ["less", "<"]) else df[df['acos'] > val]
                        res = filtered[['camp', 'acos', 'ctr', 'clicks', 'sales', 'spend']].drop_duplicates()
                        st.markdown(f"**Found {len(res)} matching campaigns:**")
                        st.dataframe(res.style.format({'acos': '{:.1f}%'}), use_container_width=True)
                    except:
                        st.error("Please provide a numeric ACoS threshold.")
                # Handle Strategic Benchmarks
                elif any(x in prompt.lower() for x in ["recommend", "optimize"]):
                    context = df.nlargest(10, 'spend').to_string()
                    response = ollama.chat(model='llama3.2', messages=[
                        {"role": "system", "content": f"You are Advertisement Monk.AI. Use benchmarks: {BENCHMARKS}. Give 3 actions."},
                        {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {prompt}"}
                    ])
                    st.markdown(response['message']['content'])
                else:
                    st.write("I am ready to audit your campaign performance. How can I help today?")

    else:
        st.info("🙏 Namaste. Upload your Excel or CSV report to activate the Monk.")

if __name__ == "__main__":
    main()
