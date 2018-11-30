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

companies_names = np.array(['Apple', 'Google', 'Microsoft', 'Amazon', 'Facebook', 'Berkshire Hathaway', 'Alibaba Group',
                            'Johnson & Johnson', 'JPMorgan', 'ExxonMobil', 'Bank of America', 'Walmart', 'Wells Fargo',
                            'Royal Dutch Shell', 'Visa', 'Procter & Gamble', 'Anheuser-Busch Inbev','AT&T',
                            'Chevron Corporation', 'UnitedHealth Group', 'Pfizer', 'China Mobile', 'Home Depot', 'Intel',
                            'Taiwan Semiconductor', 'Verizon Communications', 'Oracle Corporation', 'Citigroup',
                            'Novartis'])
companies_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'BRK.A', 'BABA', 'JNJ', 'JPM', 'XOM', 'BAC', 'WMT', 'WFC', 'RDS.A',
                    'V', 'PG', 'BUD', 'T', 'CVX', 'UNH', 'PFE', 'CHL', 'HD', 'INTC', 'TSM', 'VZ', 'ORCL', 'C', 'NVS']
percent_returns = []

for i in range(len(companies_names)):
    num, prediction = stock_bot_ret_pred(companies_stocks[i], companies_names[i], epochs=10)
    percent_change = (prediction[num+20] - prediction[num])/prediction[num]
    percent_returns.append(percent_change)

sort_ind = np.argsort(percent_returns)
print('\n\n')
print(companies_names[sort_ind][-3:])

x = np.arange(len(companies_names))
plt.bar(x, percent_returns)
plt.xticks(x, companies_names, rotation=90)
plt.savefig('figures/returns.png')
plt.show()
plt.close()