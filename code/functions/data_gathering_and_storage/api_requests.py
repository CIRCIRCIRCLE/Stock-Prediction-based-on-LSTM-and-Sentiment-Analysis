import requests
from bs4 import BeautifulSoup
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# This page contains functions of stock data and news data acquisition through API


# Stock data gathering -through IEX cloud api
def acquire_data(ticker, start_date, end_date):
    ticker = 'AAPL'
    api_key = 'pk_a2b42a419878499e9740457333423a9b'
    url = f'https://api.iex.cloud/v1/data/core/historical_prices/{ticker}?range=5y&token={api_key}'
    request = requests.get(url)
    request.raise_for_status()

    response = request.json()
    
    # Extract relevant data from the response
    data = [{'Date': entry['priceDate'],
             'High': entry['fhigh'],
             'Low': entry['flow'],
             'Open': entry['fopen'],
             'Close': entry['fclose'],
             'Volume': entry['volume']} for entry in response]
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.loc[(df['Date']>=start_date) & (df['Date']<=end_date)].sort_values(by='Date', ascending = True).reset_index(drop = True)
    return df



# News headlines gathering

#Get the date range between two dates
def daterange(start, end):
    for i in range((end - start).days + 1):
        yield (start + datetime.timedelta(i))
        
#Get the date range between two dates for given intervals        
def daterangeintv(start, end, intv):
    for i in range(intv):
        yield (start + (end - start) / intv * i)
    yield end

# New York Times Data Gathering
def access_api(query, page, start_date, end_date):
    
    #CPU sleep for 1s
    time.sleep(1)
    
    #Define the NYTimes Scraping URL
    nyt_key = 'GU1Ibqs6Efe6nZePhZa02MJ96ZyrJHmk'
    URL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q='+query+'&sort=relevance&fq='+query+'&page='+str(page)+'&api-key='+nyt_key+'&begin_date='+start_date.strftime("%Y%m%d")+'&end_date='+end_date.strftime("%Y%m%d")
    
    raw_html = None
    content = None
    i = 0
    
    #Attempt to Request page 3 times without getting any errors.
    while i<3 and (content is None or raw_html.status_code!=200):
        try:
            raw_html = requests.get(URL)
            data = json.loads(raw_html.content.decode("utf-8"))
            content = data["response"]
        except ValueError:
            raise ValueError
        except KeyError:
            continue
        i += 1
    
    #Return the page content
    return content


def fetch_news_NYT(Interval, query_list, Datelist, filter_words):
    PD_Headers = pd.DataFrame(columns=['Date', 'Headline'])
    
    # Loop through each Date Interval
    for i in range(1, Interval + 1):
        # Loop through each query on the query list
        for query in query_list:
            # Define the Date & Headline Lists
            dates = []
            headlines = []
            page = 0
            
            # Fetch content from API and store in the lists
            Data = access_api(query, page, Datelist[i-1], Datelist[i])
            while Data["meta"]["hits"] > 1000:
                Data = access_api(query, page, Datelist[i-1], Datelist[i])
            while page * 10 < Data["meta"]["hits"] and (page + 1) < 100:
                Data = access_api(query, page, Datelist[i-1], Datelist[i])
                for doc in Data["docs"]:
                    headlines.append(doc['headline']['main'])
                    dates.append(doc['pub_date'][0:10])
                page += 1
            
            # Create a new Dataframe that stores the data of this specific query
            News = pd.DataFrame({'Date': dates, 'Headline': headlines})
            
            # Concatenate this dataframe with the one containing ALL NYTimes News
            PD_Headers = pd.concat([PD_Headers, News])
    
    # Remove any possible duplicates
    PD_Headers.drop_duplicates(['Headline'], keep='last', inplace=True)
    
    # Filter the Keywords
    PD_Headers = PD_Headers[PD_Headers['Headline'].str.contains('|'.join(filter_words), case=False)]
    
    # Extract the dataframe to a CSV file
    #PD_Headers.to_csv('../data/' + query_list[0] + '_News_NYTimes.csv', index=False)
    PD_Headers.to_csv('../../../data'+ query_list[0] + '_News_NYTimes.csv', index=False)


# Financial Times Data Gathering
def fetch_news_FT(start_date, end_date, query_list, filter_words):
    # Define the Requests Headers
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

    # Prepare the Pandas Dataframe that will host the Financial Times News
    PD_Headers = pd.DataFrame(columns=['Date', 'Headline'])

    # Loop through each Date (day-to-day)
    for date in daterange(start_date, end_date):
        # Loop through each query on the query list
        for query in query_list:
            # Loop through the first 3 pages
            for page in range(1, 4):
                # Define the Financial Times Scraping URL
                url = f'https://www.ft.com/search?expandRefinements=true&q={query}&concept=a39a4558-f562-4dca-8774-000246e6eebe&dateFrom={date.strftime("20%y-%m-%d")}&dateTo={date.strftime("20%y-%m-%d")}&page={page}'

                # Request the URL and Retrieve the Headlines
                raw_html = requests.get(url, headers=headers)
                soup = BeautifulSoup(raw_html.text, 'html.parser')
                headline_list = soup.find_all("div", {"class": "o-teaser__heading"})

                # Define the Date & Headline Lists
                dates = [date] * len(headline_list)
                headlines = [elem.get_text(strip=True) for elem in headline_list]

                # Create a new Dataframe that stores the data of this specific query
                news = pd.DataFrame({'Date': dates, 'Headline': headlines})

                # Concatenate this dataframe with the one containing ALL Financial Times News
                PD_Headers = pd.concat([PD_Headers, news])

    # Remove any possible duplicates
    PD_Headers.drop_duplicates(['Headline'], keep='last', inplace=True)

    # Filter the Keywords
    PD_Headers = PD_Headers[PD_Headers['Headline'].str.contains('|'.join(filter_words), case=False)]

    # Extract the dataframe to a CSV file
    PD_Headers.to_csv(f'../../../data/{query_list[0]}_News_FT.csv', index=False)
