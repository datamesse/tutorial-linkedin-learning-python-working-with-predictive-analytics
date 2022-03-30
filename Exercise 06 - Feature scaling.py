"""
Feature scaling is used to help prevent the features with larger magnitudes from domniating the prediction model

Normalisation and standardisation are typically applied to the x variables

Normalisation (a.k.a min-max scaling)
z = (x - min(x)) ÷ (max(x) - min(x))
Values are rescaled to be between 0 and 1, unless we define a different future range using the hyperparameter. Affected by outliers.

Standardisation
z = (x - mean) ÷ standard deviation
No specific range trying to stay in. Less affected by outliers.

Original exercise file by : https://www.linkedin.com/learning/instructors/isil-berkun
Annotated by me so that I can understand what's happening at each step
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

gender = pd.DataFrame(gender)
gender.columns = ['sex']
le_gender_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for gender:")
print(le_gender_mapping)
print(gender[:10])

## le for smoker
le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for smoker:")
print(le_smoker_mapping)
print(smoker[:10])

from sklearn.preprocessing import OneHotEncoder

region = data.iloc[:,5:6].values #ndarray
ohe = OneHotEncoder() 
region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ['northeast', 'northwest', 'southeast', 'southwest']
print("Sklearn one hot encoder results for region:")   
print(region[:10])

X_num = data[['age', 'bmi', 'children']].copy()
X_final = pd.concat([X_num, region, gender, smoker], axis = 1)
y_final = data[['charges']].copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.33, random_state = 0 )



# ========================================================================================================



"""
In original exercise files numpy's np.float is used, but it has since been deprecated as float is now builtint

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

print(X_train[:10])
print(X_test[:10])