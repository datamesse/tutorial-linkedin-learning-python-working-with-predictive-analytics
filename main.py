"""
Python libraries and data import
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib as plt

data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

"""
Display first 15 rows of data into the terminal
"""
print(data.head(15))


"""
NaN Not a Number i.e. missing values
Most prediction methods cannot work with missing data
Drop column or row with NaN, or replace with new non-NaN value
"""

"""
This command displays in the console, the name of the column with NaN values, how many records there are, and the data type for the column. 
"""
count_nan = data.isnull().sum()
print(count_nan[count_nan>0])

"""
Substitutes NaN values with the mean for the column
inplace = True will make changes to the data frame
"""
data['bmi'].fillna(data['bmi'].mean(), inplace = True)

"""
Re-running this to check that the NaNs are gone
"""
count_nan = data.isnull().sum()
print(count_nan[count_nan>0])

