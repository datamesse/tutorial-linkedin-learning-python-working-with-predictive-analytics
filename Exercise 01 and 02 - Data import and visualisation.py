"""
Pandas for data importng
"""

import pandas as pd

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
Matplotlib for data visualisation

The exercise files use distplot which...:
`distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
"""

import matplotlib.pyplot as plt 

