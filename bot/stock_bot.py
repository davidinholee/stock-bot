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
class StockBot:
    # ---- Contructor takes company name and stock name.
    def __init__(self, comp_name, stock_name, exchange = 'iex'):
        self.stock_name = stock_name
        self.comp_name = comp_name
    
    # ---- Read up to date stock data and Google trends data.
    def pull(self, time_of_day = 'open'):
        start = datetime.today() - timedelta(370 * 5)
        end = datetime.today() - timedelta(1)
        self.df = web.DataReader(self.stock_name, 'iex', start, end)
        self.stock = self.df.loc[:, time_of_day]
        self.num = len(self.stock)

        pytrends = TrendReq(hl='en-US', tz=300)
        frame = self.df.index[0] + ' ' + self.df.index[-1]
        pytrends.build_payload([self.comp_name], timeframe=frame, cat=0, geo='US')
        self.trend = pytrends.interest_over_time()
        
    # ---- Scale and reorganize the data into the input format for the NN.
    def scale(self, n_days):
        self.num = n_days
        
        # ---- Rescale trends data to be between -.5 and .5
        trend_max = max(self.trend.values[:, 0])
        trend_min = min(self.trend.values[:, 0])
        self.trend_range = trend_max - trend_min
        self.trend_med = trend_min + (self.trend_range / 2)
        self.trend_scaled = np.zeros((len(self.trend.values)))
        self.trend_scaled = (self.trend.values[:, 0] - self.trend_med) / self.trend_range

        # ---- Format input features as (day, trend value)
        self.x = np.zeros((self.num, 2))
        ind_cur = 0
        trend_cur = self.trend_scaled[ind_cur]
        for i in range(len(self.x)):
            self.x[i, 0] = (i / self.num) - .7
            if (ind_cur < len(self.trend_scaled) and self.trend.index[ind_cur].year == int(self.df.index[i][:4]) and self.trend.index[
                ind_cur].month == int(self.df.index[i][5:7]) and self.trend.index[ind_cur].day <= int(self.df.index[i][8:]) + 2 and
                    self.trend.index[ind_cur].day >= int(self.df.index[i][8:]) - 2):
                trend_cur = self.trend_scaled[ind_cur]
                ind_cur += 1
            self.x[i, 1] = trend_cur

        # ---- Normalize stock data
        self.stock_scaler = MinMaxScaler()
        self.y = np.array(self.stock[:self.num]).reshape(len(self.stock[:self.num]), 1)
        self.stock_scaler = self.stock_scaler.fit(self.y)
        self.y = self.stock_scaler.transform(self.y)
        self.y = self.y.reshape(self.num)

        # ---- Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(self.y)))
        self.y = self.y[shuffle_indices]
        self.x = self.x[shuffle_indices]
    
    # ---- Construct and compile the neural network.
    def build(self):
        # ---- Construct network
        self.network = models.Sequential()
        self.network.add(layers.Dense(256, activation='relu', input_shape=(2,)))
        self.network.add(layers.Dense(256, activation='relu'))
        self.network.add(layers.Dense(256, activation='relu'))
        self.network.add(layers.Dense(128, activation='relu'))
        self.network.add(layers.Dense(128, activation='relu'))
        self.network.add(layers.Dense(128, activation='relu'))
        self.network.add(layers.Dense(64, activation='relu'))
        self.network.add(layers.Dense(64, activation='relu'))
        self.network.add(layers.Dense(64, activation='relu'))
        self.network.add(layers.Dense(1))

        # ---- Compile network
        self.network.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    # ---- Train the neural network for the given number of epochs.
    def train(self, epochs, verbose=0):
        history = self.network.fit(self.x, self.y, epochs=epochs, batch_size=50, verbose=verbose)
    
    # ---- Create a test data set and predict on it.
    def predict(self, n_days):    
        # ---- Build test data set
        self.xrange = n_days
        self.x_test = np.zeros((self.xrange, 2))
        ind_cur = 0
        trend_cur = self.trend_scaled[ind_cur]
        for i in range(len(self.x_test)):
            self.x_test[i, 0] = (i / self.num) - .7
            if (ind_cur < len(self.trend_scaled) and self.trend.index[ind_cur].year == int(self.df.index[i][:4]) and self.trend.index[
                ind_cur].month == int(self.df.index[i][5:7]) and self.trend.index[ind_cur].day <= int(self.df.index[i][8:]) + 2 and
                    self.trend.index[ind_cur].day >= int(self.df.index[i][8:]) - 2):
                trend_cur = self.trend_scaled[ind_cur]
                ind_cur += 1
            self.x_test[i, 1] = trend_cur
        self.y_pred = self.network.predict(self.x_test)
        self.y_pred = self.stock_scaler.inverse_transform(self.y_pred)[:, 0]
    
    # ---- Create a plot of the current predicted data vs. the actual stock data.
    def graph(self):
        # ---- Plot the predicted data
        step = 200
        base = datetime.strptime(self.df.index[0], '%Y-%m-%d')
        xlabels = [(base + timedelta(days=step * self.x_test * 7 / 5)).date() for self.x_test in range(0, math.ceil(self.xrange / step))]

        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(0, self.num), self.stock)
        plt.plot(np.arange(0, self.xrange), self.y_pred, lw=3)

        plt.xlabel('Date')
        plt.xticks(np.arange(0, 1400, step=step), xlabels, rotation=30)
        plt.ylabel('Price')
        plt.title(self.comp_name + ' Stock Price')
        plt.savefig('figures/' + self.comp_name + '_pred' + '.png')
        plt.close()
