import matplotlib.pyplot as plt
import seaborn as sns

#color set
colorB = '#03608c'
colorR = '#9f1f31'

def shaded_area_plot(x, mean, min, max, title, x_label, y_label):
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.plot(x, mean, label='Adjusted Close Value',color = colorB)
    
    # Fill the shaded area
    ax.fill_between(x, min, max, alpha=0.3, label='Day Range')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend()
    plt.savefig('fig/data exploration/3.2_stock_trend.png')
    
    return fig, ax



def line_plot(x, y, title, x_label, y_label):
    fig, ax = plt.subplots(1, figsize = (16, 8))
    ax.plot(x, y, color = colorB, label = 'Stock Volume')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    plt.savefig('fig/data exploration/3.2_stock_volume_variation.png')
    return fig, ax



def scatter_plot(x, y, title, x_label, y_label):
    fig, ax = plt.subplots(1, figsize = (10, 6))

    ax.scatter(x, y, color = colorB)
    plt.axhline(0, color='#9f1f31', linestyle='-', linewidth=2, label='y=0')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.savefig('fig/data exploration/3.2_sentiment correlation.png')
    return fig, ax



def covid_highlight(x, mean, title, x_label, y_label):
    fig, ax = plt.subplots(1, figsize=(16, 8))

    ax.plot(x, mean, label='Adjusted Close Value',color = colorB)

    # add area during covid-19
    plt.axvspan('2020','2021',color=colorR,alpha=0.2, label='COVID-19 Period')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    plt.savefig('fig/data exploration/4.1_a_covid_highlight.png')
    return fig, ax

def covid_highlight_vol(x, y, title, x_label, y_label):
    fig, ax = plt.subplots(1, figsize = (16, 8))
    ax.plot(x, y, color = colorB, label = 'Stock Volume')

    plt.axvspan('2020','2021',color='#9f1f31',alpha=0.2, label='COVID-19 Period')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    plt.savefig('fig/data exploration/4.1_a_covid_highlight_vol.png')
    return fig, ax


def monthly_returns_boxplot(dataframe, xlabel, ylabel, title):

    fig, ax = plt.subplots(1, figsize=(15, 8))
    dataframe.boxplot(column=ylabel, by=xlabel, color=dict(boxes=colorB, medians=colorR), ax=ax)

    month_abbreviations = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    ax.set_xticklabels(month_abbreviations)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig('fig/data exploration/4.1_b_monthly return.jpg')
    return fig, ax


def sentiment_correlation(df, features=['Close', 'compound']):
    correlation_matrix = df[features].corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Correlation between Sentiment Score and Close Price')
    plt.savefig('fig/data exploration/4.1_e_sentiment_correlation.jpg')
    
def plot_obv(df):
    plt.figure(figsize=(15, 8))
    plt.plot(df['OBV'], label='OBV', color=colorB)
    plt.plot(df['OBV_EMA'], label='OBV_EMA', color=colorR)

    plt.title('OBV and EMA Visualization')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()

    plt.savefig('fig/data exploration/4.2_OBVandEMA.png')


def OBV_based_signals(df, buy_signal, sell_signal):
    plt.figure(figsize=(15, 8))

    plt.plot(df['Date'], df['Close'], label='Close Price', alpha=0.2)
    plt.scatter(df['Date'], buy_signal, label='Buy Signal', marker='^', color=colorB)
    plt.scatter(df['Date'], sell_signal, label='Sell Signal', marker='v', color=colorR)

    plt.title('On-Balance Volume Indicator with Buy and Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig('fig/data exploration/4.2_signals_obvbased.png')
