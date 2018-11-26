import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime

start = datetime(2015,2,9)
end = datetime(2017,5,24)

f = web.DataReader('F','iex',start,end)

print(f)
