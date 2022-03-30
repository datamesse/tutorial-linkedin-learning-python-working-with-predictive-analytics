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



