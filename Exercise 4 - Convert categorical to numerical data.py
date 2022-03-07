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

data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")


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



"""
Option 1. Map each category to a differet function integer i.e. label encoding using Pandas
create series for pandas
"""

region = data["region"] # series 
region_encoded, region_categories = pd.factorize(region)
factor_region_mapping = dict(zip(region_categories, region_encoded)) #mapping of encoded numbers and original categories. 

print("Pandas factorize function for label encoding with series")  
print(region[:10]) #original version 
print(region_categories) #list of categories
print(region_encoded[:10]) #encoded numbers for categories 
print(factor_region_mapping) # print factor mapping





"""
Using skLearn to create array for label encoding
"""

# gender = data.iloc[:,1:2].values
# smoker = data.iloc[:,4,5].values


"""
Creating a label encoder function
"""

# le = LabelEncoder()
# gender[:,0] = le.fit_transform(gender[:,0])
# gender = pd.DataFrame(gender)
# gender.columns = ['gender']
# le_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print("Sklearn label encoder results for gender:")
# print(le_gender_mapping)
# print(gender[:10])


# le = LabelEncoder()
# smoker[:,0] = le.fit_transform(smoker[:,0])
# smoker = pd.DataFrame(smoker)
# smoker.columns = ['smoker']
# le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print("Sklearn label encoder results for smoker:")
# print(le_smoker_mapping)
# print(smoker[:10])
