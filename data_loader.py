# Import Dependencies
import yfinance as yf
import pandas_datareader as pdr
import pandas as pd
import streamlit as st
from datetime import timedelta

@st.cache_data
def get_fama_factors(model_type = '3-Factor'):
    """Get Fama-French Model DataFrame, Default is 3-Factor"""
    # Pick what model to get data for
    dataset = 'F-F_Research_Data_Factors_daily' if model_type == '3-Factor' else 'F-F_Research_Data_5_Factors_2x3_daily'
    try:
        # Read and adjust data, make datetime more compatible
        reader = pdr.famafrench.FamaFrenchReader(dataset)
        factors = reader.read()[0].reset_index()
        factors['Date'] = factors['Date'].dt.date
        return factors
    
    except Exception as e:
        st.error(f"Error fetching Fama French: {e}")
        return None

@st.cache_data
def get_ticker_data(stock_ticker, start_date):
    """Fetch historical stock pric and calculate daily returns, default is 5 years"""

    try:
        stock = yf.Ticker(ticker=stock_ticker)
        history = stock.history(start = start_date)

        if history.empty:
            return None
        
        history = history.reset_index()
        history['Date'] = history['Date'].dt.date

        # Calculat Daily Returns by shifting
        history['Previous Close'] = history['Close'].shift(1)
        history['Percentage Change'] = ((history['Close'] - history['Previous Close']) / history['Previous Close']) * 100

        # Using .get in case info is not there, usually due to ETFs
        ticker_info = {
            'Ticker' : stock.info.get('symbol', stock_ticker),
            'Name' : stock.info.get('shortName', 'N/A'),
            'Industry' : stock.info.get('industry', 'N/A'),
            'Summary' : stock.info.get('longBusinessSummary', 'N/A'),
            'DataFrame' : history[['Date', 'Percentage Change', 'Close']].dropna()
        }

        return ticker_info
    
    except Exception as e:
        st.error(f"Error fetching data for {stock_ticker}: {e}")
        return None

def get_factor_estimates(start_date, end_date, model_type = '3-Factor'):
    """Estimates gaps in 3 or 5 Factors using ETF proxies"""
    try:
        new_start = start_date - timedelta(days=5)
        # Tickers used in model type: SPY and BIL for Return, IWM and SPY for Size, IVE and IVW for Book to Market
        # 3-Factor Estimation
        tickers = ['SPY', 'IWM', 'IVE', 'IVW', 'BIL']
        if model_type == '5-Factor':
            # 5 Factor additions: QUAL and SPY for Robust Minus Weak: VIG and ARK for Conservative minus Aggressive
            tickers += ['QUAL', 'VIG', 'ARKK']

        proxy = yf.download(tickers=tickers, start=new_start,end=end_date)
        returns = proxy['Close'].pct_change().dropna() * 100
        returns = returns[returns.index.date >= start_date]
   
        estimates = pd.DataFrame(index = returns.index)


        # 3-Factor Calculations
        estimates['RF'] = returns['BIL']
        estimates['Mkt-RF'] = returns['SPY'] - estimates['RF']
        estimates['SMB'] = returns['IWM'] - returns['SPY']
        estimates['HML'] = returns['IVE'] - returns['IVW']

        # Additional columns for 5 Factor
        if model_type == '5-Factor':
            estimates['RMW'] = returns['QUAL'] - returns['SPY']
            estimates['CMA'] = returns['VIG'] - returns['ARKK']
        
        estimates = estimates.reset_index()
        estimates['Date'] = estimates['Date'].dt.date

        


        return estimates
    
    except Exception as e:
        st.warning(f"Could not estimate recent factors: {e}")
        return pd.DataFrame()

@st.cache_data
def load_combine(stock_ticker, model_type = '3-Factor'):
    """Combine the ticker info with the Fama French Model info"""

    # Run above functions to get necessary data
    factors = get_fama_factors(model_type= model_type)

    if factors is None:
        return None
    
    
    stock_data = get_ticker_data(stock_ticker=stock_ticker, start_date=factors['Date'].min())

    if stock_data is None:
        st.error(f"Could not find data for ticker: {stock_ticker}")
        return None
    
    stock_df = stock_data['DataFrame']

    latest_ff_date = factors['Date'].max()
    latest_stock_date = stock_df['Date'].max()

    if latest_stock_date > latest_ff_date:

        start_date = latest_ff_date + timedelta(days = 1)
        estimates = get_factor_estimates(start_date = start_date, end_date = latest_stock_date, model_type=model_type)

        if not estimates.empty:
            factors = pd.concat([factors, estimates], ignore_index=True).drop_duplicates('Date')


    # Subtract Return from Risk Free Rate
    combined = factors.merge(stock_df, on='Date', how='inner')
    combined['Excess Return'] = combined['Percentage Change'] - combined['RF']
    stock_data['DataFrame'] = combined

    # Return updated dictionary
    return stock_data

# Test Boilerplate
if __name__ == "__main__":
    print("--- Testing Data Loader ---")
    test_ticker = "NVDA"
    test_return = load_combine(test_ticker, model_type='3-Factor')

    stock_df = test_return['DataFrame']
    
    if stock_df is not None:
        print(f"Successfully fetched {len(stock_df)} days of data for {test_ticker}.")
        print(stock_df.head())
        print(stock_df.tail())
    else:
        print("Test Failed.")