import pandas as pd
import numpy as np

def interpolate_missing_dates(df):
    df['Date'] = pd.to_datetime(df['Date'])
    date_intervals = pd.date_range(df['Date'].min(), df['Date'].max())

    df.set_index('Date', inplace=True)
    interpolated_df = df.reindex(date_intervals).interpolate(method='linear')

    if not interpolated_df.empty:
        interpolated_df = interpolated_df.reset_index().rename(columns={'index': 'Date'})

    interpolated_df.sort_values(by='Date', ascending = True).reset_index(drop = True)

    return interpolated_df

def missing_value_handling(data):
    date_intervals = pd.date_range(data['Date'].min(), data['Date'].max())
    if len(data) == len(date_intervals):
        print('No missing values') 
        return data
    else:
        print('Missing values interpolated')
        return interpolate_missing_dates(data)

def zscore_outliers(x, threshold): #x: arraylike set
    z_scores = (x - np.mean(x)) / np.std(x)
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    return outlier_indices.tolist()

def oulier_detection(data):
    #check the outliers of close stock
    outlier_idx1 = zscore_outliers(data['Close'], 3)
    if outlier_idx1 == []: 
        print('No outliers')
    else: 
        print("Z-score outliers idx:", outlier_idx1)