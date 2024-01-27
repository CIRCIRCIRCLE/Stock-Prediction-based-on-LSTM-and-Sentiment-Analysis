from code.data_acquire import acquire_stock_data, acquire_NYT_news, acquire_FT_news, combine_news_data, sentiment_news, combined_stock_news, final_data
from code.functions.data_gathering_and_storage.storage import connect_to_mongoDB, insert_data, fetch_data_from_db
from code.functions.preprocessing import missing_value_handling, oulier_detection
from code.functions.visualization import shaded_area_plot, line_plot, scatter_plot
from code.functions.visualization import covid_highlight, covid_highlight_vol, monthly_returns_boxplot, sentiment_correlation
from code.functions.visualization import plot_obv, OBV_based_signals
from code.functions.indicator import monthly_return_calculation, null_hypothesis, OBV_calculation, buy_sell
from code.functions.indicator import seasonal_decomposition, seasonality_with_moving_window, seasonality_adjustment
from code.functions.model_training import LSTMStockModel
from code.model import lstm_stock_model, lstm_stock_model_with_sentiment
from keras.models import load_model
import pandas as pd
def main():
    """
    Example:
        def main():
            # acquire the necessary data
            aapl_data = final_data()

            # connect and store the data in MongoDB Atlas
            store(data)

            # format, project and clean the data
            fetch_data_from_db(db_name, collection_name)
            missing_values_interpolate
            outlier_detection

            # basic visualization
            shaded_area_plot
            line_plot
            scatter_plot

            # perform exploratory data analysis
            a. explore unusual behavior effect -covid-19
            b. monthly return
            c. Null hypothesis
            d. seasonal trend
            e. relations between news and market

            Financial Indicators: OBV-based buy or sell signals


            # show your findings
            visualise(statistics)

            # forecasting
            model1, use data from 2019-04-01 to '2023-03-31' to train
            model2, use data from 2021-01-01 to '2023-03-31' to train
            model3, with sentiment score, use data from 2021-01-01 to '2023-03-31' to train

    """
        
    #Task1 -Acquire Data
    aapl_data = final_data()
    print(aapl_data.head())


    # Task2 -Store the data into MongoDB
    db_name, collection_name = 'AAPLdb', 'StockData1'
    if connect_to_mongoDB(): insert_data(aapl_data, db_name, collection_name)
    else: print('error')


    # Task3 -Preprocessing 
    ori_data = fetch_data_from_db(db_name, collection_name).drop('_id', axis=1)
    # 3.1.1 Missing values: Stock market is closed on weeks and holidays, interpolate the data to fill the gap.
    data = missing_value_handling(ori_data)
    # 3.1.2 Outlier Detection: check the outliers of close stock
    oulier_detection(data)

    # 3.2 Data visualization
    # stock close price trend
    shaded_area_plot(data['Date'], data['Close'], data['Low'], data['High'], 'AAPL stock variation trend', 'Date', 'USD $')
    # stock volume variations
    line_plot(data['Date'], data['Volume'], 'AAPL Stock Volume Variations', 'Date', 'Volume')
    # relation between stock price and sentiment score
    scatter_plot(data['Date'], data['compound'], 'Sentiment Score Scatter Visualization', 'Date', 'Sentiment Score' )

    # 3.3 Data Transformation
    df = data[['Date', 'Close', 'Volume', 'compound']]



    # Task 4: Exploratory Data Analysis
    # 4.1 eda
    # a. explore unusual behavior effect -covid-19
    covid_highlight(df['Date'], df['Close'], 'AAPL stock variation trend', 'Date', 'USD $')
    covid_highlight_vol(data['Date'], data['Volume'], 'Stock Volume Changes', 'Date', 'Volume')

    # b. monthly return
    df_mr = df.copy()
    df_mr.set_index('Date', inplace=True)

    monthly_returns_list = monthly_return_calculation(df_mr)
    monthly_returns_boxplot(monthly_returns_list, 'Month', 'Monthly_Return', 'AAPL monthly return')

    # c. Null hypothesis
    null_hypothesis(df_mr['Close'])

    # d. seasonal trend
    df_st = df.copy()
    df_st.set_index('Date', inplace=True)
    res = seasonal_decomposition(df_st['Close'])
    seasonality_with_moving_window(res, 2020)
    seasonality_adjustment(df_st['Close'], res, 2020) #bull market adjustment
    seasonality_adjustment(df_st['Close'], res, 2022) #ordinary market adjustment

    # e. relations between news and market
    sentiment_correlation(df)
    

    # 4.2 Financial Indicators
    df_obv = df.copy()
    df_obv = OBV_calculation(df_obv)
    plot_obv(df_obv)
    buy_signal, sell_signal = buy_sell(df_obv, 'OBV', 'OBV_EMA')
    OBV_based_signals(df_obv, buy_signal, sell_signal)


    #Task 5: Forecasting
    df_model = df.copy()
    # model1, use data from 2019-04-01 to '2023-03-31' to train
    model1 = 'model/model1.h5'
    figpath1 = 'fig/model training/5.1 prediction.png'
    lstm_stock_model(df_model, '2019-04-01', '2023-03-31', model1, figpath1)

    # model2, use data from 2021-01-01 to '2023-03-31' to train
    model2 = 'model/model2.h5'
    figpath2 = 'fig/model training/5.1 prediction2.png'
    lstm_stock_model(df_model, '2021-01-01', '2023-03-31', model2, figpath2)

    # model3, with sentiment score, use data from 2021-01-01 to '2023-03-31' to train
    model3 = 'model/model3.h5'
    figpath3 = 'fig/model training/5.2 prediction3.png'
    lstm_stock_model_with_sentiment(df_model, '2021-01-01', '2023-03-31', model3, figpath3)

    
    raise NotImplementedError()


if __name__ == "__main__":
    main()
