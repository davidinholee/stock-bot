from stock_bot import StockBot
import numpy as np
import time
import matplotlib.pyplot as plt

companies_names = np.array(['Apple', 'Google', 'Microsoft', 'Amazon', 'Facebook', 'Berkshire Hathaway', 'Alibaba Group',
                            'Johnson & Johnson', 'JPMorgan', 'ExxonMobil', 'Bank of America', 'Walmart', 'Wells Fargo',
                            'Royal Dutch Shell', 'Visa', 'Procter & Gamble', 'Anheuser-Busch Inbev','AT&T',
                            'Chevron Corporation', 'UnitedHealth Group', 'Pfizer', 'China Mobile', 'Home Depot', 'Intel',
                            'Taiwan Semiconductor', 'Verizon', 'Oracle Corporation', 'Citigroup',
                            'Novartis'])
companies_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'BRK.A', 'BABA', 'JNJ', 'JPM', 'XOM', 'BAC', 'WMT', 'WFC', 'RDS.A',
                    'V', 'PG', 'BUD', 'T', 'CVX', 'UNH', 'PFE', 'CHL', 'HD', 'INTC', 'TSM', 'VZ', 'ORCL', 'C', 'NVS']
percent_returns_pred = []
percent_returns_real = []

st = time.time()
for i in range(len(companies_names)):
    sb = StockBot(companies_names[i], companies_stocks[i])
    print(companies_names[i])
    sb.pull()
    sb.scale(sb.num-20)
    sb.build()
    sb.train(epochs=2000)
    sb.predict(sb.num)
    percent_change_pred = (sb.y_pred[sb.num-1] - sb.y_pred[sb.num-21])/sb.y_pred[sb.num-21]
    percent_returns_pred.append(percent_change_pred)
    percent_change_real = (sb.stock[sb.num-1] - sb.stock[sb.num-21])/sb.stock[sb.num-21]
    percent_returns_real.append(percent_change_real)

print('\n\n')
for i in range(len(percent_returns_real)):
    print(companies_names[i], 'return:', percent_returns_real[i], 'predicted:', percent_returns_pred[i], '\n')
print('Average Return:', np.average(np.array(percent_returns_real)))
print('Average Predicted Return:', np.average(np.array(percent_returns_pred)))
print(str((time.time() - st)/60) + " min")

x = np.arange(len(companies_names))
plt.figure(figsize=(30,15))
plt.rc('ytick', labelsize=25)
plt.bar(x, percent_returns_real, label='Real', width=0.5)
plt.bar(x + 0.35, percent_returns_pred, label='Predicted', width=0.5)
plt.xticks(x + 0.17, companies_names, rotation=50, weight='bold', fontsize=11)
plt.ylabel('% Return', weight='bold', fontsize=15)
plt.title('Real vs. Predicted Returns; Average Return: ' + str(round(np.average(np.array(percent_returns_real)),5)) +
          '; Average Predicted Return: ' + str(round(np.average(np.array(percent_returns_pred)),5)), weight='bold', fontsize=25)
plt.legend(fontsize=25)
plt.savefig('figures/check/check.png')
plt.show()
plt.close()

percent_returns_pred = np.array(percent_returns_pred)
percent_returns_real = np.array(percent_returns_real)
largest = percent_returns_pred.argsort()
x = np.arange(5)
plt.figure(figsize=(16,8))
plt.rc('ytick', labelsize=15)
plt.bar(x, percent_returns_real[largest[-5:]], label='Real', width=0.5)
plt.bar(x + 0.35, percent_returns_pred[largest[-5:]], label='Predicted', width=0.5)
plt.xticks(x + 0.17, companies_names[largest[-5:]], rotation=20, weight='bold', fontsize=13)
plt.ylabel('% Return', weight='bold', fontsize=12)
plt.title('Top 5 Predicted Returns; Average Return: ' + str(round(np.average(np.array(percent_returns_real)),5)) +
          '; Average Predicted Return: ' + str(round(np.average(percent_returns_pred[largest[-5:]]),5)), weight='bold', fontsize=15)
plt.legend(fontsize=15)
plt.savefig('figures/check/check_top5.png')
plt.show()
plt.close()