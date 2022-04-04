#%%

import pandas as pd
import numpy as np

data = pd.read_excel (r'C:\Archive\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\data\international_marketplace.xlsx', sheet_name='FactSales', parse_dates = True)
df = pd.DataFrame(data, columns= ['OrderDate','Profit'])
df.set_index('OrderDate')

# create YearMonth column, and total column by YearMonth
df['YearMonth'] = df['OrderDate'].dt.to_period("M")
groupby = df.groupby(['YearMonth'])
df['MonthlyProfit'] = groupby['Profit'].transform(np.sum)
# remove OrderDate and Profit column so duplicate rows (of YearMonth and MonthlyProfit) becomes evident
# .duplicated() checks if entire row's values match a previous row, and returns True if the case except for the first occurrence
df = df.drop(columns=['OrderDate','Profit'])
df.drop_duplicates(keep='first', inplace = True)
df.set_index('YearMonth')
df.index.freq = 'MS'



# period YearMonth is converted to a string, and sufffixed with -01 to be converted to be a date type
# I did this because I am unable to figure out how to plot period object YearMonth on the x-axis, if that's possible.
import matplotlib.pyplot as plt

df['YearMonth'] = pd.to_datetime(df['YearMonth'].astype(str) + '-01')
yearMonth = df.loc[:, 'YearMonth'].values
monthlyProfit = df.loc[:, 'MonthlyProfit'].values
plt.plot(yearMonth, monthlyProfit, c = 'b', marker = '.', markersize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.xlabel('Year-Month', fontsize = 10)
plt.ylabel('Millions $', fontsize = 10)



# print(df)
# print(df.info())
# %%
