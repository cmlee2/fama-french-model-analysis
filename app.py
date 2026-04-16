import streamlit as st
import data_loader
import analysis
import plotly.graph_objects as go
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Fama-French Factor Analyzer", layout="wide")

st.title("Fama-French Factor Analysis")
st.markdown("""
*This dashboard decomposes equity returns into fundamental risk factors*
**Risk Warning:** These metrics characterize historical behavior and are NOT predictive buy signals.
""")

# 2. Sidebar 
st.sidebar.header("Analysis Settings")
ticker = st.sidebar.text_input("Enter Ticker (e.g., NVDA, AAPL, SPY)", value="NVDA").upper()
model_choice = st.sidebar.radio("Select Model", ["3-Factor", "5-Factor"])

# 3. Data Processing Pipeline
with st.spinner(f"Running {model_choice} Regression for {ticker}..."):
    data_dict = data_loader.load_combine(ticker, model_type=model_choice)
    
    if data_dict:
        df = data_dict['DataFrame']
        metrics, stability_df = analysis.run_fama_french_analysis(df, model_type=model_choice)


        st.subheader(f"Value of $1 between Bonds, {data_dict['Name']}, and S&P")
        
        # Comparison between RF, Market and Stock
        df['Bonds_Growth'] = (1 + df['RF']/100).cumprod()
        df['Market_Growth'] = (1 + (df['Mkt-RF'] + df['RF'])/100).cumprod()
        df['Stock_Growth'] = (1 + df['Percentage Change']/100).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Stock_Growth'], name=f"{ticker} (Stock)", line=dict(color='#00CC96', width=3)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Market_Growth'], name="S&P 500 (Market)", line=dict(color='#636EFA', dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Bonds_Growth'], name="Bonds (Cash/Risk-Free)", line=dict(color='#FECB52')))
        
        fig.update_layout(yaxis_title="Value of $1 Invested", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')

        # Alpha Confidence
        st.header("Factor Coefficeints & Statistical Significance")
        
        alpha_p = metrics['Alpha_P']
        if alpha_p < 0.05:
            st.success(f"**Significant Alpha Detected:** The model is 95% confident that this stock generated unique returns not explained by market factors ($p = {alpha_p:.4f}$).")
        else:
            st.warning(f"**Insignificant Alpha:** The unique return ($p = {alpha_p:.4f}$) is not statistically significant. Its performance is largely driven by broad market factors.")

        # Dynamic Grid: Adjusts based on 3-Factor or 5-Factor selection
        col_count = 4 if model_choice == "3-Factor" else 6
        cols = st.columns(col_count)
        
        cols[0].metric("Alpha (Daily)", f"{metrics['Alpha']:.4f}")
        cols[1].metric("Market Beta", f"{metrics['Mkt_Beta']:.2f}")
        cols[2].metric("Size (SMB)", f"{metrics['SMB_Beta']:.2f}")
        cols[3].metric("Value (HML)", f"{metrics['HML_Beta']:.2f}")
        
        if model_choice == "5-Factor":
            cols[4].metric("Profit (RMW)", f"{metrics.get('RMW_Beta', 0):.2f}")
            cols[5].metric("Invest (CMA)", f"{metrics.get('CMA_Beta', 0):.2f}")

        # --- Academic Context & Interpretation ---
        with st.expander("How to Interpret these Values"):

            st.write("""
            * **Alpha:** The 'Secret Sauce.' Positive alpha suggests skill or unique company value.
            * **Market Beta:** Sensitivity to the S&P 500. A beta of 1.5 means the stock moves 1.5x for every 1% market move.
            * **SMB (Small Minus Big):** Positive values move like Small-Caps; Negative values move like Blue-Chip Giants.
            * **HML (High Minus Low):** Positive suggests a 'Value' (undervalued) profile; Negative suggests a 'Growth' profile.
            * **RMW (Robust Minus Weak):** Higher values mean the company has high, stable operating profitability.
            * **CMA (Conservative Minus Aggressive):** Higher values suggest the company invests conservatively in its own growth.
            """)

        # Time Series Split Results
        st.header("Time Series Split Details (4-Fold Cross-Validation)")
        st.info("I split the last 5 years into 4 chronological blocks to see if the stock's coefficients stays the same over time.")
        
        fmt = {'Alpha': '{:.4f}', 'Mkt-Beta': '{:.2f}', 'SMB-Beta': '{:.2f}', 'HML-Beta': '{:.2f}', 'Out_R2': '{:.2%}'}
        if model_choice == "5-Factor":
            fmt.update({'RMW-Beta': '{:.2f}', 'CMA-Beta': '{:.2f}'})

        st.dataframe(
            stability_df.style.format(fmt), 
            use_container_width=True, 
            hide_index=True 
        )

    else:
        st.error(f"Error: Could not retrieve data for {ticker}. Check the logs or try another ticker.")