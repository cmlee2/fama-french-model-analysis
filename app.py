import streamlit as st
import data_loader
import analysis
import plotly.graph_objects as go
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Fama-French Factor Analyzer", layout="wide")

st.title("📊 Fama-French Factor Analysis Dashboard")
st.markdown("""
*This tool decomposes a stock's performance into specific risk factors (Market, Size, Value, etc.).*
**Disclaimer:** This is a risk characterization tool, not financial advice or a buy signal.
""")

# 2. Sidebar Controls
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Ticker", value="NVDA").upper()
model_choice = st.sidebar.radio("Select Model", ["3-Factor", "5-Factor"])

# 3. Data Processing
with st.spinner(f"Analyzing {ticker}..."):
    data_dict = data_loader.load_combine(ticker, model_type=model_choice)
    
    if data_dict:
        df = data_dict['DataFrame']
        metrics, stability_df, full_ols = analysis.run_fama_french_analysis(df, model_type=model_choice)

        # --- Performance Graph vs S&P 500 ---
        st.subheader(f"{data_dict['Name']} vs. Market Proxy (Cumulative Returns)")
        
        # Calculate Cumulative Returns for visualization
        df['Stock_Cum'] = (1 + df['Percentage Change'] / 100).cumprod()
        df['Mkt_Cum'] = (1 + (df['Mkt-RF'] + df['RF']) / 100).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Stock_Cum'], name=ticker, line=dict(color='#00CC96')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Mkt_Cum'], name="Market (S&P 500)", line=dict(color='#636EFA')))
        st.plotly_chart(fig, use_container_view=True)

        # --- Coefficients & Explanations ---
        st.header("🧬 Factor Identity (The DNA)")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Alpha (Excess Return)", f"{metrics['Alpha']:.4f}")
        col2.metric("Market Beta", f"{metrics['Mkt_Beta']:.2f}")
        col3.metric("Size Beta (SMB)", f"{metrics['SMB_Beta']:.2f}")
        col4.metric("Value Beta (HML)", f"{metrics['HML_Beta']:.2f}")

        with st.expander("What do these coefficients mean?"):
            st.write("""
            * **Alpha:** Returns that can't be explained by the factors. If significant, it suggests unique value (or luck).
            * **Market Beta:** Sensitivity to the broad market. >1.0 means the stock is more volatile than the S&P 500.
            * **SMB (Small Minus Big):** Positive means it moves like a small-cap; negative means it moves like a 'Blue Chip' giant.
            * **HML (High Minus Low):** Positive suggests a 'Value' stock; negative suggests a 'Growth' stock.
            """)

        # --- Stability Report (The Folds) ---
        st.header("🧪 Stability Report")
        st.info("We split the data into 4 chronological 'folds' to see if the stock's behavior is consistent or changing.")
        
        # Format for display
        display_df = stability_df.copy()
        st.dataframe(display_df.style.format({
            'Alpha': '{:.4f}',
            'Alpha_P': '{:.4f}',
            'In_Sample_R2': '{:.2%}',
            'Out_Sample_R2': '{:.2%}',
            'RMSE': '{:.4f}'
        }), use_container_view=True)

    else:
        st.error("Unable to load data. Please check the ticker symbol.")