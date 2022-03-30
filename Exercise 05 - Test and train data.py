
"""
Spliting dataset for training and testing purposes

The final dataframe is composed of individual dataframes stacked on top of each other.
Dimensionality reduction is the practice of removing dimensions to reduce process time.
Common division is 2/3rds of data for training, and 1/3 of data for testing.
i.e. combining all the independent variables as x, and assigning the response a.k.a dependent variable to y
Important: do not use the test data to train the model, so that it can perform well when it actually runs on the yet unseen test data (from the model's perspective)

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


# ========================================================================================================


"""
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

print(X_train[:10])
print(X_test[:10])
print(y_train[:10])
print(y_test[:10])