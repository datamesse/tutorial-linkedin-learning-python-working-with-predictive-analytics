"""
All of the code below is from the LinkedIn Learning course, Python: Working with Predictive Analytics by Michael Galarnyk and Madecraft
https://www.linkedin.com/learning/python-for-data-visualization

I have annotated it with additional notes for my own personal learning.
"""



"""
CREATE DATASET USING PANDAS
"""

"""
Option 1. Create your own dataframe as a nested list i.e. standard Python list
"""

"""
carLoans = [
   [1, 34689.96, 687.23, 202.93, 484.3, 34205.66, 60, 0.0702, 'Toyota Sienna'],
   [2, 34205.66, 687.23, 200.1, 487.13, 33718.53, 60, 0.0702, 'Toyota Sienna'],
   [3, 33718.53, 687.23, 197.25, 489.98, 33228.55, 60, 0.0702, 'Toyota Sienna'],
   [4, 33228.55, 687.23, 194.38, 492.85, 32735.7, 60, 0.0702, 'Toyota Sienna'],
   [5, 32735.7, 687.23, 191.5, 495.73, 32239.97, 60, 0.0702, 'Toyota Sienna']
]

colNames = [
   'Month',
   'Starting Balance',
   'Repayment',
   'Interest Paid',
   'Principal Paid',
   'New Balance',
   'term',
   'interest_rate',
   'car_type'
]

import pandas as pd
df = pd.DataFrame(data = carLoans, columns = colNames)
print(df)
"""


"""
Option 2. Create your own dataframe as a NumPy array, which is more efficient than a Python list
"""

"""
import numpy as np

carLoans = np.array([
   [1, 34689.96, 687.23, 202.93, 484.3, 34205.66, 60, 0.0702, 'Toyota Sienna'],
   [2, 34205.66, 687.23, 200.1, 487.13, 33718.53, 60, 0.0702, 'Toyota Sienna'],
   [3, 33718.53, 687.23, 197.25, 489.98, 33228.55, 60, 0.0702, 'Toyota Sienna'],
   [4, 33228.55, 687.23, 194.38, 492.85, 32735.7, 60, 0.0702, 'Toyota Sienna'],
   [5, 32735.7, 687.23, 191.5, 495.73, 32239.97, 60, 0.0702, 'Toyota Sienna']
])

colNames = [
   'Month',
   'Starting Balance',
   'Repayment',
   'Interest Paid',
   'Principal Paid',
   'New Balance',
   'term',
   'interest_rate',
   'car_type'
]

import pandas as pd
df = pd.DataFrame(data = carLoans, columns = colNames)
print(df)
"""

"""
Option 3. Use a Python dictionary
"""

"""
carLoans = {
   'Month' : {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
   'Starting Balance' : {0: 34689.96, 1: 34205.66, 2: 33718.53, 3: 33228.55, 4: 32735.7},
   'Repayment' : {0: 687.23, 1: 687.23, 2: 687.23, 3: 687.23, 4: 687.23},
   'Interest Paid' : {0: 202.93, 1: 200.1, 2: 197.25, 3: 194.38, 4: 191.5},
   'Principal Paid' : {0: 484.3, 1: 487.13, 2: 489.98, 3: 492.85, 4: 495.73},
   'New Balance' : {0: 34205.66, 1: 33718.53, 2: 33228.55, 3: 32735.7, 4: 32239.97},
   'term' : {0: 60, 1: 60, 2: 60, 3: 60, 4: 60},
   'interest_rate' : {0: 0.0702, 1: 0.0702, 2: 0.0702, 3: 0.0702, 4: 0.0702},
   'car_type' : {0: 'Toyota Sienna', 1: 'Toyota Sienna', 2: 'Toyota Sienna', 3: 'Toyota Sienna', 4: 'Toyota Sienna'}
}

import pandas as pd
df = pd.DataFrame(data = carLoans)
print(df)
"""



"""
LOAD CSV AND EXCEL FILES USING PANDAS
"""

import pandas as pd
data = pd.read_excel (r'nal_marketplace.xlsx', sheet_name='FactSales')







"""
MATPLOTLIB
"""