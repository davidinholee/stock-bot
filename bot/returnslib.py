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
from botlib import stock_bot_ret_pred
import time

companies_names = np.array(['Apple', 'Google', 'Microsoft', 'Amazon', 'Facebook', 'Berkshire Hathaway', 'Alibaba Group',
                            'Johnson & Johnson', 'JPMorgan', 'ExxonMobil', 'Bank of America', 'Walmart', 'Wells Fargo',
                            'Royal Dutch Shell', 'Visa', 'Procter & Gamble', 'Anheuser-Busch Inbev','AT&T',
                            'Chevron Corporation', 'UnitedHealth Group', 'Pfizer', 'China Mobile', 'Home Depot', 'Intel',
                            'Taiwan Semiconductor', 'Verizon Communications', 'Oracle Corporation', 'Citigroup',
                            'Novartis'])
companies_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'BRK.A', 'BABA', 'JNJ', 'JPM', 'XOM', 'BAC', 'WMT', 'WFC', 'RDS.A',
                    'V', 'PG', 'BUD', 'T', 'CVX', 'UNH', 'PFE', 'CHL', 'HD', 'INTC', 'TSM', 'VZ', 'ORCL', 'C', 'NVS']
percent_returns = []

st = time.time()
for i in range(len(companies_names)):
    num, prediction = stock_bot_ret_pred(companies_stocks[i], companies_names[i], epochs=2000)
    percent_change = (prediction[num+20] - prediction[num])/prediction[num]
    percent_returns.append(percent_change)

sort_ind = np.argsort(percent_returns)
print('\n\n')
print(companies_names[sort_ind][-3:])
percent_returns = np.array(percent_returns)
print(str((time.time() - st)/60) + " min")

x = np.arange(6)
plt.figure(figsize=(8,8))
plt.bar(x, np.append(percent_returns[sort_ind][-3:], percent_returns[sort_ind][:3]))
plt.xticks(x, np.append(companies_names[sort_ind][-3:], companies_names[sort_ind][:3]), rotation=30)
plt.savefig('figures/returns_six.png')
plt.show()
plt.close()

x = np.arange(len(companies_names))
plt.figure(figsize=(20,10))
plt.bar(x, percent_returns)
plt.xticks(x, companies_names, rotation=60)
plt.savefig('figures/returns_all.png')
plt.show()
plt.close()
