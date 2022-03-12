"""
Converting categorical to numerical data
"""

"""
Categorical encoding to numbers
  Label encoding   - 2 distinct values (a.k.a binomial e.g. "Yes" = 1, "No" = 0)
  One hot encoding - 3 or more distinct values (the values become columns and those columns have binomial treatment e.g. Column colour's values are blue, red, and yellow. Those are each made into a column with 0 or 1 as theeir vvalues)

         Label encoding    One hot encoding
-------------------------------------------
sklearn  LabelEncoder      OneHotEncoding
pandas   factorize( )      get_dummies

Note: In live, there may be a different number of unique values for a field compared to the testing dataset.
In which case, we fit the encoder function only to the training data.
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")


"""
Option 1. Map each category to a differet function integer i.e. label encoding using Pandas

https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
.factorize creates an array substituting each categorical value as an integer
structure
array of integers, unique categorical values = pd.factorize(dataset)
e.g. region_encoded, region_categories = pd.factorize(region)

https://realpython.com/python-zip-function/
.zip pairs elements from 2 or more arrays with each otther
e.g. 
region_categories = ['southwest', 'southeast','northwest','northeast']
region_encoded = [0,1,1,2]
for l, n in zip(region_categories, region_encoded):
     print(f'Category: {l}')
     print(f'Number: {n}')
End result is [('southwest', 0), ('southeast', 1), ('northwest', 1),('northeast',2)]


"""

region = data["region"] # retrieves the entire region column by itself
region_encoded, region_categories = pd.factorize(region)
# factor_region_mapping = dict(zip(region_categories, region_encoded)) #mapping of encoded numbers and original categories. 


region_categories = ['southwest', 'southeast','northwest','northeast']
region_encoded = [0,1,1,2]
for l, n in zip(region_categories, region_encoded):
     print(f'Category: {l}')
     print(f'Number: {n}')

# print("Pandas factorize function for label encoding with series")  
# print(region[:10]) #original version 
# print(region_categories) #list of categories
# print(region_encoded[:10]) #encoded numbers for categories 
# print(factor_region_mapping) # print factor mapping
