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


