# https://towardsdatascience.com/introduction-to-hierarchical-time-series-forecasting-part-ii-e73dfd3de86b




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
