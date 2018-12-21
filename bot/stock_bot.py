import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers, optimizers
import math
from pytrends.request import TrendReq
import pytrends

# ---- Main stock bot function, saves a figure of the prediction of the trained neural network.
def Stock_Bot:
    exchange = 'iex'
    verbose  = true
    # Contructor takes stock name, optionally everything else
    def __init__(self, comp_name, stock_name, exchange = 'iex'):
        self.stock_name = stock_name
        self.comp_name = comp_name
    
    def pull(self, time_of_day = 'open'):
        # ---- Read stock data and Google trends data.
        start = datetime.today() - timedelta(370 * 5)
        end = datetime.today() - timedelta(1)
        df = web.DataReader(stock_name, 'iex', start, end)
        stock = df.loc[:, time_of_day]
        self.num = len(stock)

        pytrends = TrendReq(hl='en-US', tz=300)
        frame = df.index[0] + ' ' + df.index[-1]
        pytrends.build_payload([comp_name], timeframe=frame, cat=0, geo='US')
        search = pytrends.interest_over_time()

        # ---- Rescale trends data to be between -.5 and .5
        trend_max = max(search.values[:, 0])
        trend_min = min(search.values[:, 0])
        self.trend_range = trend_max - trend_min
        self.trend_med = trend_min + (trend_range / 2)
        search_scaled = np.zeros((len(search.values)))
        search_scaled = (search.values[:, 0] - trend_med) / trend_range

        # ---- Format input features as (day, trend value)
        self.x = np.zeros((num, 2))
        ind_cur = 0
        trend_cur = search_scaled[ind_cur]
        for i in range(len(x)):
            x[i, 0] = (i / num) - .7
            if (ind_cur < len(search_scaled) and search.index[ind_cur].year == int(df.index[i][:4]) and search.index[
                ind_cur].month == int(df.index[i][5:7]) and search.index[ind_cur].day <= int(df.index[i][8:]) + 2 and
                    search.index[ind_cur].day >= int(df.index[i][8:]) - 2):
                trend_cur = search_scaled[ind_cur]
                ind_cur += 1
            x[i, 1] = trend_cur

        # ---- Normalize stock data
        self.y = stock
        self.stock_scaler = MinMaxScaler()
        y = np.array(stock).reshape(len(stock), 1)
        stock_scaler = stock_scaler.fit(y)
        y = stock_scaler.transform(y)
        y = y.reshape(len(stock))

        # ---- Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        y = y[shuffle_indices]
        x = x[shuffle_indices]
    
    def build(self):
        # ---- Construct network
        network = models.Sequential()
        network.add(layers.Dense(256, activation='relu', input_shape=(2,)))
        network.add(layers.Dense(256, activation='relu'))
        network.add(layers.Dense(256, activation='relu'))
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dense(128, activation='relu'))
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dense(1))

        # ---- Compile network
        network.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        
    def train(self, epochs, verbose):
        history = network.fit(x, y, epochs=epochs, batch_size=50, verbose=verbose)

        
    # ---- Build test data set
    xrange = 1259
    x_test = np.zeros((xrange, 2))
    ind_cur = 0
    trend_cur = search_scaled[ind_cur]
    for i in range(len(x_test)):
        x_test[i, 0] = (i / num) - .7
        if (ind_cur < len(search_scaled) and search.index[ind_cur].year == int(df.index[i][:4]) and search.index[
            ind_cur].month == int(df.index[i][5:7]) and search.index[ind_cur].day <= int(df.index[i][8:]) + 2 and
                search.index[ind_cur].day >= int(df.index[i][8:]) - 2):
            trend_cur = search_scaled[ind_cur]
            ind_cur += 1
        x_test[i, 1] = trend_cur
    y_pred = network.predict(x_test)
    y_pred = stock_scaler.inverse_transform(y_pred)[:, 0]

    # ---- Plot the predicted data
    step = 200
    base = datetime.strptime(df.index[0], '%Y-%m-%d')
    xlabels = [(base + timedelta(days=step * x_test * 7 / 5)).date() for x_test in range(0, math.ceil(xrange / step))]

    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(0, num), stock)
    plt.plot(np.arange(0, xrange), y_pred, lw=3)

    plt.xlabel('Date')
    plt.xticks(np.arange(0, 1400, step=step), xlabels, rotation=30)
    plt.ylabel('Price')
    plt.title(comp_name + ' Stock Price')
    plt.savefig('figures/' + comp_name + '_pred' + '.png')
    plt.show()
    plt.close()
