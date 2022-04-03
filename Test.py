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

from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(df['MonthlyProfit'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
range = pd.date_range('01-01-2024', periods=12, freq='MS')
predictions = model.forecast(12)
predictions_range = pd.DataFrame({'MonthlyProfit':predictions,'YearMonth':range})

# print(predictions_range)
# print(predictions_range.info)
# print(df)

predictions_range.plot(figsize=(12,8))

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

df.index.freq = 'MS'

from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(df['DailyProfit'],trend='mul',seasonal='mul',seasonal_periods=24).fit()
range = pd.date_range('01-01-2024', periods=12, freq='MS')
predictions = model.forecast(12)
predictions_range = pd.DataFrame({'DailyProfit':predictions,'OrderDate':range})

print(predictions_range)
"""


"""
# MULTIPLE LINEAR REGRESSSION
X_num = data[['OrderDate']].copy()
X_final = pd.concat([X_num], axis = 1)
y_final = data[['Profit']].copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(float))
X_test= s_scaler.transform(X_test.astype(float))

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# getting the ccoefficients and intercept
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print('lr train score %.3f, lr test score: %.3f' % (
lr.score(X_train,y_train),
lr.score(X_test, y_test)))

"""





# %%
