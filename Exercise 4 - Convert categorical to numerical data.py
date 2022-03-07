"""
Python libraries and data import
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

"""
Categorical encoding to numbers
  Label encoding - 2 distinct values (a.k.a binomial e.g. "Yes" = 1, "No" = 0)
  One hot encoding - 3 or more distinct values (the values become columns and those columns have binomial treatment e.g. Column colour's values are blue, red, and yellow. Those are each made into a column with 0 or 1 as theeir vvalues)

         Label encoding    One hot encoding
-------------------------------------------
sklearn  LabelEncoder      OneHootEncoding
pandas   factorize( )      get_dummies
"""

"""
Using skLearn to create array for label encoding
"""

gender = data.iloc[:,1:2].values
smoker = data.iloc[:,4,5].values


"""
Creating a label encoder function
"""

le = LabelEncoder()
gender[:,0] = le.fit_transform(gender[:,0])
gender = pd.DataFrame(gender)
gender.columns = ['gender']
le_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for gender:")
print(le_gender_mapping)
print(gender[:10])


le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for smoker:")
print(le_smoker_mapping)
print(smoker[:10])