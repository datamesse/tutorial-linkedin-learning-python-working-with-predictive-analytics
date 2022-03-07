"""
Python libraries and data import
"""

import pandas as pd
from sklearn.impute import SimpleImputer

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
OPTIONS FOR NaN HANDLING
Remove hashtags to uncomment code
The count_nan above would display:
   bmi    5
   dtype: int64
to reflect that bmi column of data type integer has 5 entries with NaN

The count_nan below (assuming at least 1 option has been uncommented) would display
   Series([], dtype: int64)
to indicate no columns have NaN values
"""


"""
Option 1. Substitute NaN values with the mean for the column (using Pandas)
inplace = True will make changes to the data frame
"""
data['bmi'].fillna(data['bmi'].mean(), inplace = True)
print(data.head(15))

"""
Option 2. Drop the whole column
"""
# data.drop('bmi', axis = 1, inplace = True)


"""
Option 3. Drop data rows
.dropna without parameters removes whole rows where there is a NaN
.dropna can be implemented to drop whole columns by passing axis parameter
e.g.
dataframename.dropna(axis='columns')
.reset_index(drop=True) removes the existing index and places a new one in beginning from 0
"""

# data.dropna(inplace=True)
# data.reset_index(drop=True, inplace=True)
# print(data.head(15))


"""
Option 4. Substitute NaN values with the mean for the column (using Scikit-learn)
strategy='mean' is the default, other strategies include: median, most_fequent, constant
https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
https://stackoverflow.com/questions/46691596/why-does-sklearn-imputer-need-to-fit
https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
"""

# imputer = SimpleImputer(strategy='mean')
# imputer.fit(data['bmi'].values.reshape(-1, 1))
# data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))
# print(data.head(15))



"""
Re-running this to check that the NaNs are gone
"""
count_nan = data.isnull().sum()
print(count_nan[count_nan>0])
