# Fama French Stock Screener
The purpose of this app is to use the Fama-French 3-Factor or 5-Factor Model to analyze stocks to help indviduals make more informed decisions with investing. This tool helps investors understand if a stock's performance is driven by market trends, company size, value charateristics or unique Alpha. 

The app is deployed and available here: https://fama-french-stock-screener.streamlit.app/

# Details
For this project, I constructed a Data Pipeline from the Fama-French Datasets using pandas-datareader. The Fama-French Datasets typically lag a month behind current stock data, so to fill these gaps, I approximated using the returns of large ETFs. For example, I used the returns of SPY minus BIL as a proxy for the Market Return minus the Risk Free Rate. I utilized yfinance to pull in ticker data for the desired stock.

After constructing the Data Pipeline, I used statsmodel and scikit learn to compare fit the ticker onto a linear regression. I utilized Time Series Splits to validate my model and ensure my model is not overfitting.

Lastly, I built the app and deployed via streamlit.

