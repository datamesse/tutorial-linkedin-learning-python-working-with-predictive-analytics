#%%


"""
# pip install openpyxl

import pandas as pd
data = pd.read_excel (r'C:\Archive\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\data\international_marketplace.xlsx', sheet_name='FactSales')
df = pd.DataFrame(data, columns= ['OrderDate','ProductID','CityID','Profit'])
# print(df[:10])

# one hot encoding ProductID values into columns
product = data["ProductID"]
product_encoded = pd.get_dummies(product, prefix='pro')

print(product_encoded[:10])

city = data["CityID"]
city_encoded = pd.get_dummies(product, prefix='cit')

print(city_encoded[:10])
"""


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel (r'C:\Archive\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\data\international_marketplace.xlsx', sheet_name='FactSales')
df = pd.DataFrame(data, columns= ['OrderDate','ProductID','CityID','Profit'])
df.set_index('OrderDate')
df.index.freq = 'MS'

month = df.loc[:, 'OrderDate'].values
profit = df.loc[:, 'Profit'].values

plt.plot(month, profit)
"""



"""
# dataset.set_index('OrderDate')
# dataset.index.freq = 'MS'
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(df['Profit'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
range = pd.date_range('01-01-2024', periods=12, freq='MS')
predictions = model.forecast(12)
predictions_range = pd.DataFrame({'Profit':predictions,'Order Date':range})
"""

"""
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use('seaborn')
data = pd.read_excel (r'C:\Archive\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\data\international_marketplace.xlsx', sheet_name='FactSales')
df = pd.DataFrame(data, columns= ['OrderDate','ProductID','CityID','Profit'])
month = df.loc[:, 'OrderDate'].values
profit = df.loc[:, 'Profit'].values

plt.plot_date(month, profit)
"""


"""
import pandas as pd
import numpy as np

data = pd.read_excel (r'C:\Archive\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\data\international_marketplace.xlsx', sheet_name='FactSales', parse_dates = True)
df = pd.DataFrame(data, columns= ['OrderDate','ProductID','Profit'])
df.set_index('OrderDate')

# create YearMonth column, and total column by YearMonth and ProductID
df['YearMonth'] = df['OrderDate'].dt.to_period("M")
groupby_product = df.groupby(['ProductID','YearMonth'])
df['MonthlyProfitByProduct'] = groupby_product['Profit'].transform(np.sum)
# remove OrderDate and Profit column so duplicate rows (of YearMonth and MonthlyProfitByProduct) becomes evident
# .duplicated() checks if entire row's values match a previous row, and returns True if the case except for the first occurrence
df = df.drop(columns=['OrderDate','Profit'])
df.drop_duplicates(keep='first', inplace = True)


test_filter = df['ProductID']==1574
dfx = df.loc[test_filter, :]
"""



"""
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


print(df[:10])
df.info()
"""


import pandas as pd
import numpy as np

data = pd.read_excel (r'C:\Archive\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\data\international_marketplace.xlsx', sheet_name='FactSales', parse_dates = True)
df = pd.DataFrame(data, columns= ['OrderDate','Profit'])
df.set_index('OrderDate')

# create DailyProfit
groupby = df.groupby(['OrderDate'])
df['DailyProfit'] = groupby['Profit'].transform(np.sum)
# remove OrderDate and Profit column so duplicate rows (of DailyProfit) becomes evident
# .duplicated() checks if entire row's values match a previous row, and returns True if the case except for the first occurrence
df = df.drop(columns=['Profit'])
df.drop_duplicates(keep='first', inplace = True)

print(df[:10])
df.info()

import matplotlib.pyplot as plt
df.index.freq = 'MS'

df.plot(figsize=(12,8))

# %%
