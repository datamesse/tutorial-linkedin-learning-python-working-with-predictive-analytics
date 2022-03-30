"""
Common regression models (used to predict numerical values)
• Multiple linear regression
• Polynomial regression
• Support Vector Regression (SVM)
• Decision tree
• Random forest regression

Original exercise file by : https://www.linkedin.com/learning/instructors/isil-berkun
Annotated by me so that I can understand what's happening at each step
"""

import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/insurance.csv")


from sklearn.impute import SimpleImputer

# data.drop('bmi', axis = 1, inplace = True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
imputer = SimpleImputer(strategy='mean')
imputer.fit(data['bmi'].values.reshape(-1, 1))
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1))
data['bmi'].fillna(data['bmi'].mean(), inplace = True)


from sklearn.preprocessing import LabelEncoder

#create ndarray for label encodoing (sklearn)
gender = data.iloc[:,1:2].values
smoker = data.iloc[:,4:5].values

## le for gender
le = LabelEncoder()
gender[:,0] = le.fit_transform(gender[:,0])

gender = pd.DataFrame(gender)
gender.columns = ['sex']
le_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

## le for smoker
le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

from sklearn.preprocessing import OneHotEncoder

region = data.iloc[:,5:6].values #ndarray
ohe = OneHotEncoder() 
region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ['northeast', 'northwest', 'southeast', 'southwest']

X_num = data[['age', 'bmi', 'children']].copy()
X_final = pd.concat([X_num, region, gender, smoker], axis = 1)
y_final = data[['charges']].copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )

from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(float))
X_test= s_scaler.transform(X_test.astype(float))



# ========================================================================================================


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