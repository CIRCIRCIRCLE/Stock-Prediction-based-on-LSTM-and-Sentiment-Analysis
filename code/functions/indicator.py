import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

def monthly_return_calculation(df_mr):
        # Calculate percentage change
    r = df_mr['Close'].pct_change()  
    
    # Group by year and month, then calculate mean
    Monthly_Returns = r.groupby([r.index.year.rename('Year'), r.index.month.rename('Month')]).mean()
    
    # Create a DataFrame with the results
    Monthly_Returns_List = pd.DataFrame({
        'Year': Monthly_Returns.index.get_level_values(0),
        'Month': Monthly_Returns.index.get_level_values(1),
        'Monthly_Return': Monthly_Returns.values
    })

    return Monthly_Returns_List


def null_hypothesis(df_column):
    result = adfuller(df_column)
    print('The null hypothesis assumes the presence of a unit root, signifying that the data is non-stationary.')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if result[1] > 0.05:
        print("The hypothesis is not rejected.")


'''
Seasonal Trend:
The stock time-series are non-stationary, I attempt to exploit seasonality trends using time series decomposition.
  
If we assume an additive decomposition, then we can write  `yt = St+Tt+Rt` 
where  yt is the data, St is the seasonal component, Tt is the trend-cycle component, and Rt is the remainder component, all at period t.
'''

def seasonal_decomposition(data):
    # Decompose the time series
    res = sm.tsa.seasonal_decompose(data, model='additive', period=365)

    fig = res.plot()
    fig.set_size_inches((15,9))
    fig.tight_layout()

    plt.savefig('fig/data exploration/4.1_d_seasonality.png')
    return res

'''
MA1 is the moving average of AAPL_Seasonality with a window size of 20. 
This can be useful to smooth out short-term fluctuations and highlight longer-term trends in the data.
'''

def seasonality_with_moving_window(res, year):
    
    # Select seasonality for the given year
    AAPL_Seasonality = res.seasonal[res.seasonal.index.year == year]

    # Calculate 20-day moving average
    MA = AAPL_Seasonality.rolling(window=20).mean()

    # Plotting
    fig, ax = plt.subplots(1, figsize=(15, 6))

    plt.plot(AAPL_Seasonality, label=f'AAPL Seasonality of {year}', color='#03608c')
    plt.plot(MA, label='20 Days Moving Average', color='#9f1f31')

    plt.title(f"Seasonality of AAPL ({year})", fontsize=12)
    plt.legend()

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.savefig('fig/data exploration/4.1_d_seasonality_with_moving_window.png')

    return fig, ax

def seasonality_adjustment(stock_data, res, year):
    seasonality = res.seasonal[res.seasonal.index.year == year]
    stock_data_year = stock_data[stock_data.index.year == year]

    plt.figure(figsize=(15, 6))
    plt.plot(stock_data_year + seasonality, label=f'Seasonality Adjusted Stock Data ({year})', color = '#9f1f31')
    plt.plot(stock_data_year, label=f'Original Data ({year})', color = '#03608c')

    plt.title(f'Stock Data with and without Seasonality Adjustment ({year})', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(f'fig/data exploration/4.1_d_seasonalityadj({year}).jpg')

# Financial Indicator (on-balance volume)
def OBV_calculation(df, span=20):

    # Calculate daily price changes
    df['Price Change'] = df['Close'].diff()
    
    # Calculate volume flow
    df['Volume Flow'] = np.where(df['Price Change'] > 0, df['Volume'], -df['Volume'])
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = df['Volume Flow'].cumsum()
    
    # Calculate OBV Exponential Moving Average (EMA)
    df['OBV_EMA'] = df['OBV'].ewm(span=span).mean()

    df = df[['Date', 'Close', 'OBV', 'OBV_EMA']]

    return df


def buy_sell(signal, col1, col2):
    signPriceBuy = []
    signPriceSell = []
    flag = -1
    # Loop through the length of the data set
    #col1 => 'OBV' and col2 => 'OBV_EMA'
    for i in range(0, len(signal)):
        # If OBV > OBV_EMA Then Buy 
        if signal[col1][i] > signal[col2][i] and flag != 1:
            signPriceBuy.append(signal['Close'][i])
            signPriceSell.append(np.nan)
            flag = 1
        # If OBV < OBV_EMA Then Sell
        elif signal[col1][i] < signal[col2][i] and flag != 0:
            signPriceSell.append(signal['Close'][i])
            signPriceBuy.append(np.nan)
            flag = 0
        else:
            signPriceSell.append(np.nan)
            signPriceBuy.append(np.nan)

    return signPriceBuy, signPriceSell

