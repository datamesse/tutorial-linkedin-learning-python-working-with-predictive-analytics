"""
Converting categorical to numerical data

Original exercise file by : https://www.linkedin.com/learning/instructors/isil-berkun
Annotated by me so that I can understand what's happening at each step
"""

"""
Categorical encoding to numbers
  One hot encoding - 2 distinct values (a.k.a binomial e.g. "Yes" = 1, "No" = 0)
  Label encoding - 3 or more distinct values (the values become columns and those columns have binomial treatment e.g. Column colour's values are blue, red, and yellow. Those are each made into a column with 0 or 1 as theeir vvalues)

         Label encoding    One hot encoding
-------------------------------------------
sklearn  LabelEncoder      OneHotEncoding
pandas   factorize( )      get_dummies

Note: In live, there may be a different number of unique values for a field compared to the testing dataset.
In which case, we fit the encoder function only to the training data.
"""

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





"""
Option 1. Label encoding using Pandas

https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
.factorize() creates an array substituting each categorical value as an integer
structure
array of integers, unique categorical values = pd.factorize(dataset)
e.g. region_encoded, region_categories = pd.factorize(region)

https://realpython.com/python-zip-function/
.zip() pairs elements from 2 or more arrays with each otther
e.g. 
region_categories = ['southwest', 'southeast','northwest','northeast']
region_encoded = [0,1,1,2]
for l, n in zip(region_categories, region_encoded):
     print(f'Category: {l}')
     print(f'Number: {n}')
End result is [('southwest', 0), ('southeast', 1), ('northwest', 1),('northeast',2)]

https://www.w3schools.com/python/ref_func_dict.asp
.dict() creates a dictionary,  which stores data values in key:value pairs
"""

"""
import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

region = data["region"] # retrieves the entire region column by itself
region_encoded, region_categories = pd.factorize(region)
factor_region_mapping = dict(zip(region_categories, region_encoded))

print("Pandas factorize function for label encoding with series")  
print(region[:10]) # first 10 rows of original data 
print(region_categories) # list of unique category values
print(region_encoded[:10]) # first 10 encoded numbers for categories
print(factor_region_mapping) # dictionary key:value mapping of numbers with categorical values
"""





"""
Option 2. One hot encoding using Pandas
https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
.get_dummies can result in this
  _northeast  _northwest  _southeast  _southwest
0           0           0           0           1
1           0           0           1           0
2           0           0           1           0
3           0           1           0           0
"""

"""
import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

region = data["region"] # retrieves the entire region column by itself
region_encoded = pd.get_dummies(region, prefix='')

print("Pandas get_dummies function for one hot encoding with series")  
print(region_encoded) #encoded numbers for categories 
"""





"""
Option 3. Label encoding using sklearn
Conceptually, an ndarray is a (usually fixed-size) multi-dimensional container of items of the same type and size

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
https://www.youtube.com/watch?v=xvpNA7bC8cs by Data School
.loc[] for selecting row and column combinations
   .loc[rowsbyintger,columnsbyindex]
   : means all
   .loc[0, :]   means all columns for row 0
   .loc[[0,2,4], :]   means all columns for rows 0, 2, and 4
   .loc[0:4, :]   means all columns for rows between 0 to 4 inclusive
   .loc[:, 'City':'State']   means columns between City and State inclusive for all rows

.loc can be used with a Boolean condition
   .loc[dataframe.columnname=='stringvalue', :]   means all columns for rows whose columnname is stringvalue

.iloc[] for selecting row and column combintations by integer position (0 start)
behaves similarly to .loc, except the integer after : is excluded
   examples
   .iloc[:, 0:4]   means columns 0, 1, 2, and 3 for all rows
   .iloc[0:3, :]   means all columns for rows 0, 1, and 2

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html
pandas .values removes axes labels i.e. row number and column name




"""



import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

from sklearn.preprocessing import LabelEncoder

#create ndarray for label encodoing (sklearn)
gender = data.iloc[:,1:2].values
smoker = data.iloc[:,4:5].values

## le for gender
le = LabelEncoder()
gender[:,0] = le.fit_transform(gender[:,0])

print(gender) # continue analysing code from here


gender = pd.DataFrame(gender)
gender.columns = ['sex']
le_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for gender:")
print(le_gender_mapping)
print(gender[:10]) # prints first 10 rows

## le for smoker
le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for smoker:")
print(le_smoker_mapping)
print(smoker[:10])









"""
Option 4. One hot encoding using sklearn

Honestly, Option 2 seems simpler.
"""

"""
import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

from sklearn.preprocessing import OneHotEncoder

#create ndarray for one hot encodoing (sklearn)
region = data.iloc[:,5:6].values #ndarray

## ohe for region
ohe = OneHotEncoder() 

region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ['northeast', 'northwest', 'southeast', 'southwest']
print("Sklearn one hot encoder results for region:")   
print(region[:10])

"""