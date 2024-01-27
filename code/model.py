import pandas as pd
from code.functions.model_training import LSTMStockModel  # Assuming you have a module/class for LSTMStockModel
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def lstm_stock_model(df_model, historical_start_date,  historical_end_date, loaded_model, savepath):

    # Initialize LSTMStockModel
    stock_model = LSTMStockModel()
    
    # Load and preprocess data
    df1, arrdata = stock_model.loaddata(df_model, ['Date', 'Close'], historical_start_date, ['Close'])
    scaled_data, scaler = stock_model.normalization(arrdata)
    
    # Determine training length
    train_len = stock_model.training_length(df1, historical_end_date)
    
    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = stock_model.train_test_split(scaled_data, arrdata, train_len, 20)
    print('Shape of x test:', x_test.shape)
    
    # Load or create the model
    model = load_model(loaded_model)  # Update the path as needed
    
    # Make predictions
    predictions, train, valid, rmse = stock_model.prediction(model, x_test, y_test, scaler, df1, train_len)
    
    # Visualize predictions
    stock_model.prediction_viz(train, valid, predictions, savepath)

    return rmse


def lstm_stock_model_with_sentiment(df_model, historical_start_date,  historical_end_date, loaded_model, savepath):

    stock_model = LSTMStockModel()
    
    df1, arrdata = stock_model.loaddata(df_model, ['Date', 'Close', 'compound'], historical_start_date, ['Close', 'compound'])
    scaled_data, scaler = stock_model.normalization(arrdata)
    
    # Determine training length
    train_len = stock_model.training_length(df1, historical_end_date)
    
    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = stock_model.train_test_split(scaled_data, arrdata, train_len, 20)
    print('Shape of x test:', x_test.shape)

    '''
    model3 = stockmodel.model_architecture_2dim(x_train)
    history = stockmodel.model_fitting(model3, x_train, y_train, 'model/model3.h5')
    stockmodel.training_loss_viz(history)   
    '''
    
    # Load or create the model
    model = load_model(loaded_model)  # Update the path as needed
    
    # Make predictions
    predictions, train, valid, rmse = stock_model.prediction(model, x_test, y_test, scaler, df1, train_len)
    
    # Visualize predictions
    stock_model.prediction_viz(train, valid, predictions, savepath)

    return rmse
