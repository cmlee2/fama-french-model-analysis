import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_fama_french_analysis(combined_df, model_type = '3-Factor'):
    """Run the analysis and analze fit for model"""
    factors = ['Mkt-RF', 'SMB', 'HML']
    if model_type == '5-Factor':
        factors += ['RMW', 'CMA']
    
    X = combined_df[factors]
    y = combined_df['Excess Return']

    # Time Series Split Data
    tscv = TimeSeriesSplit(n_splits=4, test_size=252, gap=5)
    stability_results = []
    for i, (train_index, test_index) in enumerate(tscv.split(combined_df)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # OLS model for Granular Data
        X_train_const = sm.add_constant(X_train)
        ols_fold = sm.OLS(y_train, X_train_const).fit()
        
        # LinearRegression for prediction (used for test split)
        model_fold = LinearRegression().fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        
        fold_data = {
            'Fold': i + 1,
            'Start_Date': combined_df.iloc[test_index]['Date'].min(),
            'End_Date': combined_df.iloc[test_index]['Date'].max(),
            'Alpha': ols_fold.params['const'],
            'Alpha_P': ols_fold.pvalues['const'],
            'Mkt-Beta': ols_fold.params['Mkt-RF'],
            'SMB-Beta': ols_fold.params['SMB'],
            'HML-Beta': ols_fold.params['HML'],
            'In_Sample_R2': ols_fold.rsquared,
            'Out_Sample_R2': model_fold.score(X_test, y_test),
            'RMSE': rmse
        }

        if model_type == '5-Factor':
            fold_data['RMW-Beta'] = ols_fold.params['RMW']
            fold_data['CMA-Beta'] = ols_fold.params['CMA']

        
        stability_results.append(fold_data)


    # Final Model (include all data)
    X_full_const = sm.add_constant(X)
    final_model = sm.OLS(y, X_full_const).fit()
    
    # Create a summary dictionary for the UI
    final_metrics = {
        'Alpha': final_model.params['const'],
        'Mkt_Beta': final_model.params['Mkt-RF'],
        'SMB_Beta': final_model.params['SMB'],
        'HML_Beta': final_model.params['HML'],
        'Alpha_P': final_model.pvalues['const'],
        'Adj_R2': final_model.rsquared_adj,
        'Conf_Lower': final_model.conf_int().loc['const', 0],
        'Conf_Upper': final_model.conf_int().loc['const', 1]
    }
    
    if model_type == '5-Factor':
        final_metrics['RMW_Beta'] = final_model.params['RMW']
        final_metrics['CMA_Beta'] = final_model.params['CMA']

    return final_metrics, pd.DataFrame(stability_results)
