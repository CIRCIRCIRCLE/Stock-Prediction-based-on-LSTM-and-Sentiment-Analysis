from code.functions.data_gathering_and_storage.api_requests import acquire_data, daterange, daterangeintv, access_api, fetch_news_NYT, fetch_news_FT
from code.functions.data_gathering_and_storage.sentiment_analysis import preprocess_news_data, sentiment_score_calculation, aggregate_same_day_news
import datetime
import pandas as pd

start_date = '2019-04-01'
end_date = '2023-04-30'
ticker = 'AAPL'

start_year, start_month, start_day = 2019, 4, 1
end_year, end_month, end_day = 2023, 4, 30
query_list = ['aapl', 'apple+inc', 'apple']
filter_words = ['aapl', 'iphone', 'ipad', 'mac', 'apple', 'aal']
Interval = 10

#Define the two dates in datetime format
start_date_dt = datetime.date(start_year, start_month, start_day)
end_date_dt = datetime.date(end_year, end_month, end_day)


def acquire_stock_data():
    return acquire_data(ticker, start_date, end_date)

def acquire_NYT_news():
    # This function will scrape new headlines and store into a csv file
    Datelist = list(daterangeintv(start_date_dt, end_date_dt, Interval))
    print('Start scraping New York Times data')
    fetch_news_NYT(Interval, query_list, Datelist, filter_words)
    print('Finished')

def acquire_FT_news():
    print('Start scraping Financial Times news data')
    fetch_news_FT(start_date_dt, end_date_dt, query_list, filter_words)
    print('Finished')

#combine two news data files 
def combine_news_data():
    # Read FT data
    FT_data = pd.read_csv('../data/'+query_list[0]+'_News_FT.csv')
    FT_data = FT_data.assign(Source='FT')
    FT_data = FT_data[['Date', 'Headline', 'Source']]

    # Read NYTimes data
    NYT_data = pd.read_csv('../data/'+query_list[0]+'_News_NYTimes.csv')
    NYT_data = NYT_data.assign(Source='NYTimes')
    NYT_data = NYT_data[['Date', 'Headline', 'Source']]

    all_data = pd.concat([FT_data, NYT_data])
    all_data.loc[:, 'Date'] = pd.to_datetime(all_data['Date'], format='%Y-%m-%d')
    all_data.drop_duplicates(['Headline'], keep='last')
    all_data.to_csv('../data/'+query_list[0]+'_News_All.csv')

# This function return the organized news data with sentiment scores
def sentiment_news():
    news_data = preprocess_news_data()
    news_data = sentiment_score_calculation(news_data)
    news_with_sentiment = aggregate_same_day_news(news_data)
    return news_with_sentiment

# Combine stock data with sentiment news data
def combined_stock_news():
    stock_data = acquire_stock_data()
    news_with_sentiment = sentiment_news()
    combined_stock_news_data = pd.merge(stock_data, news_with_sentiment, on='Date', how = 'outer')
    return combined_stock_news_data

# Deal with nan value and data trimming 
def final_data():
    data = combined_stock_news()
    #drop the headline
    data.drop(['Headline'], inplace=True, axis=1)

    #extract Date before T  2019-04-01T00:00:00.000+00:00
    data['Date'] = data['Date'].astype(str).str.split('T').str[0]

    #Set NaN values for Sentiment to be the Mean
    data['compound'].fillna(data['compound'].mean(), inplace=True)
    data['negative'].fillna(data['negative'].mean(), inplace=True)
    data['neutral'].fillna(data['neutral'].mean(), inplace=True)
    data['positive'].fillna(data['positive'].mean(), inplace=True)

    #drop null values if any
    data.dropna(axis=0, inplace=True)
    return data
