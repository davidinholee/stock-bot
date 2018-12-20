#!/bin/python3

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

# can use company name to derive stock code
names = {
    'Apple': 'AAPL',  
    'Google': 'GOOGL',
    'Microsoft': 'MSFT',
    'Amazon': 'AMZN',
    'Facebook': 'FB',
    'Berkshire Hathaway': 'BRK.A',
    'Alibaba Group': 'BABA',
    'Johnson & Johnson': 'JNJ',
    'JPMorgan': 'JPM',
    'ExxonMobil': 'XOM',
    'Bank of America': 'BAC',
    'Walmart': 'WMT',
    'Wells Fargo': 'WFC',
    'Royal Dutch Shell': 'RDS.A',
    'Visa': 'V',
    'Procter & Gamble': 'PG',
    'Anheuser-Busch Inbev': 'BUD',
    'AT&T': 'T',
    'Chevron Corporation': 'CVX',
    'UnitedHealth Group': 'UNH',
    'Pfizer': 'PFE',
    'China Mobile': 'CHL',
    'Home Depot': 'HD',
    'Intel': 'INTC',
    'Taiwan Semiconductor': 'TSM',
    'Verizon Communications': 'VZ',
    'Oracle Corporation': 'ORCL',
    'Citigroup': 'C',
    'Novartis': 'NVS',
    'Ford': 'F'})

# Stockbot class stores stock data and can be rerun
class Stockbot:
    exchange = 'iex'
    verbose  = true
    # Contructor takes stock name, optionally everything else
    def __init__(self, stock_name = 'Ford', exchange = 'iex'):
        self.stock_name = stock_name
        self.comp_name = names[stock_name]

    def pull(self, time_of_day = 'open'):
        # ---- Read stock data and Google trends data.
        start = datetime.today() - timedelta(370 * 5)
        end = datetime.today() - timedelta(1)
        df = web.DataReader(stock_name, 'iex', start, end)
        self.stock = df.loc[:, 'open']
        self.num = len(stock)
        prerr(num)

        pytrends = TrendReq(hl='en-US', tz=300)
        frame = df.index[0] + ' ' + df.index[-1]
        pytrends.build_payload([comp_name], timeframe=frame,
                cat=0, geo='US')
        self.search = pytrends.interest_over_time()

        # ---- Rescale trends data to be between -.5 and .5
        trend_max = max(search.values[:, 0])
        trend_min = min(search.values[:, 0])
        trend_range = trend_max - trend_min
        trend_med = trend_min + (trend_range / 2)
        search_scaled = np.zeros((len(search.values)))
        search_scaled = (search.values[:, 0] - trend_med) / trend_range

        # ---- normalize stock data
        self.y = stock
        self.scaler = minmaxscaler()
        y = np.array(stock).reshape(len(stock), 1)
        self.scaler = scaler.fit(y)
        y = scaler.transform(y)
        y = y.reshape(len(stock))

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

        # ---- Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        y = y[shuffle_indices]
        x = x[shuffle_indices]

    def build(self):
        # ---- Construct network
        self.network = models.Sequential()
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

        # ---- Compile and run network
        network.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def train(self):
        history = network.fit(x, y, epochs=epochs, batch_size=50, verbose=verbose)


    def predict(self, day):
        self.network.predict()

    def prerr(s):
        if self.verbose :
            print(s)

