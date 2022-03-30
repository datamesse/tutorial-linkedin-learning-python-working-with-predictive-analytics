"""
All of the code below is from the LinkedIn Learning course, Python: Working with Predictive Analytics by Isil Berkun
https://www.linkedin.com/learning/python-working-with-predictive-analytics/

I have annotated it with additional notes for my own personal learning, as a lot of minor details like what the functions/methods each indiviually do.
I still highly recommend it as a starting off point for getting into predictive analysis with Python

Note: The exercise files use distplot for data visualisation, however `distplot` is a deprecated function and will be removed in a future version.
Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms)
message displays when trying to run that code, so have excluded it from the file below.
"""


"""
STEP 1. USING PANDAS FOR DATA IMPORTS
"""

import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

"""
# displays first 15 rows of data into the terminal
print(data.head(15))
"""



"""
STEP 2. HANDLING NOT A NUMBER a.k.a. NaNs (i.e. missing values)

Most prediction methods cannot work with missing data
Drop column or row with NaN, or replace with new non-NaN value
"""

"""
# displays the name of the column with NaN values, how many records there are, and the data type for the column
count_nan = data.isnull().sum()
print(count_nan[count_nan>0])
"""

"""
OPTIONS FOR NaN HANDLING
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
# print(data.head(15))

"""
Option 2. Drop the whole column
"""

"""
data.drop('bmi', axis = 1, inplace = True)
"""



"""
Option 3. Drop data rows
.dropna without parameters removes whole rows where there is a NaN
.dropna can be implemented to drop whole columns by passing axis parameter
e.g.
dataframename.dropna(axis='columns')
.reset_index(drop=True) removes the existing index and places a new one in beginning from 0
"""

"""
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.head(15))
"""



"""
Option 4. Substitute NaN values with the mean for the column (using Scikit-learn)
strategy='mean' is the default, other strategies include: median, most_fequent, constant
https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
https://stackoverflow.com/questions/46691596/why-does-sklearn-imputer-need-to-fit
https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
"""

"""
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))
print(data.head(15))
"""



"""
Re-run this to check that the NaNs are gone
"""

"""
count_nan = data.isnull().sum()
print(count_nan[count_nan>0])
"""



"""
CONVERT CATEGORICAL DATA TO NUMERICAL ENCODED DATA

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


# import pandas as pd
# data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")

region = data["region"] # retrieves the entire region column by itself
region_encoded = pd.get_dummies(region, prefix='')

# print("Pandas get_dummies function for one hot encoding with series")  
# print(region_encoded) #encoded numbers for categories 




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

# print(gender) # continue analysing code from here


gender = pd.DataFrame(gender)
gender.columns = ['sex']
le_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print("Sklearn label encoder results for gender:")
# print(le_gender_mapping)
# print(gender[:10]) # prints first 10 rows

## le for smoker
le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print("Sklearn label encoder results for smoker:")
# print(le_smoker_mapping)
# print(smoker[:10])




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



"""
SPLIT DATASET FOR TRAINING AND TESTING MODEL

The final dataframe is composed of individual dataframes stacked on top of each other.
Dimensionality reduction is the practice of removing dimensions to reduce process time.
Common division is 2/3rds of data for training, and 1/3 of data for testing.
i.e. combining all the independent variables as x, and assigning the response a.k.a dependent variable to y
Important: do not use the test data to train the model, so that it can perform well when it actually runs on the yet unseen test data (from the model's perspective)

X_num is being populated with the numeric data from the original, then concatenated with the data that was prepared 
i.e. one-hot or label encoded earlier from categorical variables, together as X_final.
That covers the x axis, so the y axis which is the value being predicted, in this case called charges, is being copied from the original
"""

X_num = data[['age', 'bmi', 'children']].copy()
X_final = pd.concat([X_num, region, gender, smoker], axis = 1)
y_final = data[['charges']].copy()


"""
Assigning 33% of the data to test, and remainder for training
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

# print(X_train[:10])
# print(X_test[:10])
# print(y_train[:10])
# print(y_test[:10])



"""
FEATURE SCALING

In original exercise files numpy's np.float is used, but it has since been deprecated as float is now builtint
"""

"""
Normalising the train, fit on test???
"""

"""
from sklearn.preprocessing import MinMaxScaler

n_scaler = MinMaxScaler()
X_train = n_scaler.fit_transform(X_train.astype(float))
X_test= n_scaler.transform(X_test.astype(float))

print(X_train[:10])
print(X_test[:10])
"""

"""
Standardising the train data, fi on test???
"""
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(float))
X_test= s_scaler.transform(X_test.astype(float))

# print(X_train[:10])
# print(X_test[:10])



"""
PREDICTION MODELS
"""

"""
MULTIPLE LINEAR REGRESSION
explains dependent y based on multiple independent x variables
"""

"""
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



"""
POLYNOMIAL REGRESSION
still fitting a linear model, but the features are treated as polynomial
scaling is applied to x data, i.e. fit and transform the x train data, but only transform the x test data
try adjusting the degree to explore different scenarios
"""

"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures (degree = 2)
X_poly = poly.fit_transform(X_final)

X_train,X_test,y_train,y_test = train_test_split(X_poly,y_final, test_size = 0.33, random_state = 0)

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(float))
X_test= sc.transform(X_test.astype(float))

#fit model
poly_lr = LinearRegression().fit(X_train,y_train)

y_train_pred = poly_lr.predict(X_train)
y_test_pred = poly_lr.predict(X_test)

#print score
print('poly train score %.3f, poly test score: %.3f' % (
poly_lr.score(X_train,y_train),
poly_lr.score(X_test, y_test)))
"""



"""
SUPPORT VECTOR REGRESSION (SVR)
highly sensitive to outliers, so it is essential to apply scaling before using this method, i.e. feature scaling
"""

"""
from sklearn.svm import SVR

svr = SVR(kernel='linear', C = 300)

#test train split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

#define a standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(float))
X_test= sc.transform(X_test.astype(float))

#fit model
svr = svr.fit(X_train,y_train.values.ravel())
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

#print score
print('svr train score %.3f, svr test score: %.3f' % (
svr.score(X_train,y_train),
svr.score(X_test, y_test)))
"""



"""
DECISION TREE REGRESSION a.k.a. classification and regression trace (CART)
Weakness of decision trees is that they tend to “overfit the model” i.e. they train high but test poor i.e. the model cannot be generalised enough for new data
"""

"""
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)

#test train split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(float))
X_test= sc.transform(X_test.astype(float))

#fit model
dt = dt.fit(X_train,y_train.values.ravel())
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

#print score
print('dt train score %.3f, dt test score: %.3f' % (
dt.score(X_train,y_train),
dt.score(X_test, y_test)))
"""



"""
RANDOM FOREST REGRESSION
bags data into smaller samples and runs a different model for each sample, then aggregates the collective
i.e. a collection of decision trees
define the regressor with parameters
n_estimators is the number of trees
criterion is the feature selection criteria, 'mse' is the default which is mean squared error
n_jobs is the number of jobs to run in parallel for both fit and predict, -1 means use all processors

more parameters for the randomforestregressor can be found here
https://scikit-learn.org/stable
"""

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
#test train split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(float))
X_test= sc.transform(X_test.astype(float))

#fit model
forest.fit(X_train,y_train.values.ravel())
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

#print score
print('forest train score %.3f, forest test score: %.3f' % (
forest.score(X_train,y_train),
forest.score(X_test, y_test)))



"""
HYPERPARAMETER OPTIMISATION
custom function print_best_params prints out the optimum parameters
define a search space for SVR in the dictionary, containing kernel function, degree, C, and epsilon
2 kernels are provided, linear and poly
C is the penalty parameter of the error term
epsilon defines a margin of tolerance where no penalty is given to errors
cv is cross validation
verbose means the higher it is, the more messages that are sen on the console

Important: GridSearch may take time to finish depending on the computer, so reduce the number of parameters if needed
"""

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV

#Function to print best hyperparamaters: 
def print_best_params(gd_model):
    param_dict = gd_model.best_estimator_.get_params()
    model_str = str(gd_model.estimator).split('(')[0]
    print("\n*** {} Best Parameters ***".format(model_str))
    for k in param_dict:
        print("{}: {}".format(k, param_dict[k]))
    print()

#test train split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(float))
X_test= sc.transform(X_test.astype(float))

###Challenge 1: SVR parameter grid###
param_grid_svr = dict(kernel=[ 'linear', 'poly'],
                     degree=[2],
                     C=[600, 700, 800, 900],
                     epsilon=[0.0001, 0.00001, 0.000001])
svr = GridSearchCV(SVR(), param_grid=param_grid_svr, cv=5, verbose=3)


#fit model
svr = svr.fit(X_train,y_train.values.ravel())

#print score
print('\n\nsvr train score %.3f, svr test score: %.3f' % (
svr.score(X_train,y_train),
svr.score(X_test, y_test)))
#print(svr.best_estimator_.get_params())

print_best_params(svr)