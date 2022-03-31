#%%

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
Option 2. Create your own dataframe as a NumPy array, which is more efficient to process than a Python list
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

"""
import pandas as pd
filename = 'https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/car_financing.csv'
df = pd.read_csv(filename)
print(df[:10])
"""

import pandas as pd
filename = 'https://raw.githubusercontent.com/datamesse/tutorial-linkedin-learning-python-working-with-predictive-analytics/main/data/car_financing.xlsx'
df = pd.read_excel (filename, sheet_name='Sheet1')
# print(df[:10])



"""
BASIC OPERATIONS USING PANDAS
"""

"""
# Examine the head and tail of dataset
print(df.head(10))
print(df.tail(10))
"""

"""
# Check column data types
print(df.dtypes)
"""

"""
# Check number of rows and columns
print(df.shape)
"""

"""
# check for non-null values, in this case Interest Paid column has 1 null value
print(df.info())
"""



"""
SLICING USING PANDAS
when interested in smalller subset of the data
Note: if single square brackets are used i.e. [ ], it is a Pandas series, not a Pandas dataframe
A Pandas series cannot have columns specified
Define as a Pandas dataframe by using double square brackets [[ ]]

Series can be used if you know the start and end index of your selection

Best practice for selecting golumns is using the .loc attribute
allows column selection, indexing, and data slicing
"""

"""
# select first 10 rows of specific column for Pandas dataframe
print(df[['car_type']].head(10))

# select first 10 rows of 2 specific columns
print(df[['car_type', 'Principal Paid']].head(10))

# check the dataframe's type e.g. if it is a Pandas dataframe
print(type(df[['car_type', 'Principal Paid']].head(10))) # this returns as Pandas dataframe
print(type(df['car_type'].head(10))) # this returns as a Pandas series

# select first 10 rows of specific column for Pandas series
print(df['car_type'][0:10])

# select all rows and specific column using .loc for Pandas dataframe
print(df.loc[:, ['car_type']])

# select all rows and specific column using .loc for Pandas series
print(df.loc[:, 'car_type'])
"""


"""
FILTERING USING PANDAS

combining operators
&   and
|   or
^   exclusive or
~   not
"""

"""
# using .value_counts to get a count grouped on column values
print(df['car_type'].value_counts)

# create filter as binary check of every row to meet condition, result is list of True and False values
car_filter = df['car_type']=='Toyota Sienna'
print(car_filter)

# apply filter to the dataframe IMPORTANT: the original dataframe is unaffected because it isn't an assigned change
print(df[car_filter])

# alternative to applying filter to the dataframe is to retrieve all columns of filter
print(df.loc[car_filter, :])

# overwrite the existing dataframe with the filtered dataframe
car_filter = df['car_type']=='Toyota Sienna'
df = df.loc[car_filter, :]

interest_filter = df['interest_rate']==0.0702
df = df.loc[interest_filter, :]
"""

# shorter version of the above
car_filter = df['car_type']=='Toyota Sienna'
interest_filter = df['interest_rate']==0.0702
df = df.loc[car_filter & interest_filter, :]



"""
RENAMING & REMOVING COLUMNS USING PANDAS

Note: the List replacement method requires full list of names, but is prone to error
List replacement is just providing the full list of names as an array df.columns = ['month','starting_balance', etc...]
"""

"""
Renaming columns using dictionary substitution
"""
df = df.rename(columns={
   'Starting Balance': 'starting_balance',
   'Interest Paid': 'interest_paid',
   'Principal Paid': 'principal_paid',
   'New Balance': 'new_balance'
})
# print(df[:10])


# Delete column using drop method
df = df.drop(columns=['term'])

# Delete column using del command
del df['Repayment']




"""
AGGREGATE FUNCTIONS

Note: whilst it is possible to sum all the vealues for each column using df.sum() by itself,
this is not recommended as string columns will concatenate as a result
"""

"""
# sum entire column
print(df['interest_paid'].sum())
"""



"""
HANDLING MISSING DATA i.e. NaN Not a Number

some functions will trigger an error or misrepresent it if there are missing values
df.info() can be used to find NaN
.isna() andd .isnull() can be used to find nnulls
they are idential methods just with different names in Python
in R programming, they are different things

Note: you can use the - i.e. not operator to negate the filter
i.e. show all rows that don't have NaN
df.loc[-interest_missing,:]
"""

"""
interest_missing = df['interest_paid'].isna()
print(df.loc[interest_missing,:])
"""


"""
REMOVE OR FILL MISSING DATA

data amputation
backfilling and forwardfilling is common wheen you have time series data with missing values
Note: generally not recommended to replace NaN with zero
"""

"""
# from rows 30 to 39, drop rows that have any NaN values for any column
df[30:40].dropna(how = 'any')

# replace NaN with zero value
df['interest_paid'][30:40].fillna(0)

# replace NaN with the value from the next row a.k.a. "backfill"
df['interest_paid'][30:40].fillna(method = 'bfill')

# replace NaN with the value from the previous row a.k.a. "forwardfill"
df['interest_paid'][30:40].fillna(method = 'ffill')

# with domain knowledge, manually fill in the actual value
interest_missing = df['interest_paid'].isna()
df.loc[interest_missing, 'interest_paid'] = 93.24
"""

# linear interpolation, replace NaN with a linear model value
interest_missing = df['interest_paid'].isna()
df.loc[interest_missing, 'interest_paid'] = df['interest_paid'][30:40].interpolate(method = 'linear')

# df.info()



"""
CONVERT PANDAS DATAFRAME TO NUMPY ARRAY OR DICTIONARY

certain Python libraries prefer numpy arrays or dictionaries as inputs to their methods
dictionaries are used to preserve the indices of the Pandas dataframe
"""

"""
# Option 1. convert to NumPy array using .to_numpy()
df.to_numpy()

# Option 2. convert to NumPy array using .values()
df.values

# Option 3. convert to dictionary using .to_dict()
df.to_dict()
"""



"""
EXPORT PANDAS DATAFRAMES TO CSV AND EXCEL

index = False means the dataframe's indexes won't be exported
"""

"""
df.to_csv(path_or_buf='data/table_i702t60.csv', index=False)
df.to_excel(excel_writer='data/table_i702t60.xlsx', index=False)
"""


"""
DATA VISUALISATION USING MATPLOTLIB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
columns/field values must first be set up as numpy arrays
.loc converts it into a pandas series, then the .values converts that into a numpy array
check the conversion worked using type(month_number)
"""

month_number = df.loc[:, 'Month'].values
interest_paid = df.loc[:, 'interest_paid'].values
principal_paid = df.loc[:, 'principal_paid'].values

"""
# plot one line (x axis, y axis)
plt.plot(month_number, interest_paid)

# plot 2 lines on the same graph
plt.plot(month_number, interest_paid)
plt.plot(month_number, principal_paid)
"""

"""
plt.style.available to choose figure style
by default, plt.style.use('classic') is used
"""

"""
# list different styles to choose from
plt.style.available

plt.style.use('fivethirtyeight')
plt.plot(month_number, interest_paid)
plt.plot(month_number, principal_paid)
"""



"""
SET MARKER TYPE AND CHANGE COLOURS

'.'   point marker
','   pixel marker
'o'   circle marker
'v'   triangle_down marker
'^'   triangle_up marker
'<'   triangle_left marker
'>'   triangle_right marker
's'   square marker
'*'   star marker
'+'   plus marker
'x'   x marker

'b'         blue
'blue'      blue
'g'         green
'green'     green
'r'         red
'red'       red
'c'         cyan
'cyan'      cyan
'm'         magneta
'magneta'   magneta
'y'         yellow
'yellow'    yellow
'k'         black
'black'     black
'w'         white
'white'     white
"""

"""
# add markers and increase their size, and set colours
plt.plot(month_number, interest_paid, c = 'b', marker = '.', markersize = 10)
plt.plot(month_number, principal_paid, c = 'g', marker = '.', markersize = 10)
"""

"""
# set colours using hex strings
plt.plot(month_number, interest_paid, c = '#00FF00', marker = '.', markersize = 10)
plt.plot(month_number, principal_paid, c = '#FF0000', marker = '.', markersize = 10)
"""

"""
# set colours using rgb tuples
plt.plot(month_number, interest_paid, c = (0, 0, 0), marker = '.', markersize = 10)
plt.plot(month_number, principal_paid, c = (0, 0, 1), marker = '.', markersize = 10)
"""



"""
OBJECT-ORIENTATED

previous scripting of plots is in the MATLAB-style
below is the object-orientated style
Note: most plots created are a mix
"""

"""
fig, axes = plt.subplots(nrows = 1, ncols = 1)
plt.plot(month_number, interest_paid, c = 'k') # MATLAB style
axes.plot(month_number, principal_paid, c = 'b') # object orientated style
"""


"""
PLOT TITLES, LABELS, AND AXIS LIMITS

MATLAB-style
set axis limits
   plt.xlim(left = , right = )
   plt.ylim(bottom = , top = )
set axis labels
   plt.xlabel('')
   plt.ylabel('')
set title
   plt.title('')
set axis label size
   plt.xticks(fontsize = )
   plt.yticks(fontsize = )

OBJECT-ORIENTATED-style
   axes.set_xlim(left = , right =)
   axes.set_ylim(bottom = , top =)
   axes.set_xlabel('')
   axes.set_ylabel('')
   axes.set_title('')
   axes.tick_params(axis = 'x', labelsize = )
   axes.tick_params(axis = 'y', labelsize = )
"""

"""
# MATLAB style
plt.plot(month_number, interest_paid, c = 'k')
plt.plot(month_number, principal_paid, c = 'b')
plt.xlim(left=1,right=70)
plt.ylim(bottom=0,top=1000)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.xlabel('Month', fontsize = 12)
plt.ylabel('Dollars', fontsize = 12)
plt.title('Interest and Principal Paid Each Month', fontsize = 16)
"""

"""
# object-orientated style
fig, axes = plt.subplots(nrows = 1, ncols = 1)
axes.plot(month_number, interest_paid, c='k')
axes.plot(month_number, principal_paid, c = 'b')
axes.set_xlim(left = 1 , right = 70)
axes.set_ylim(bottom = 0, top = 1000)
axes.set_xlabel('Month', fontsize = 12)
axes.set_ylabel('Dollars', fontsize = 12)
axes.set_title('Interest and Principal Paid Each Month', fontsize = 16)
axes.tick_params(axis = 'x', labelsize = 12)
axes.tick_params(axis = 'y', labelsize = 12)
"""



"""
GRIDLINES

MATLAB-style
   plt.grid()   adds vertical and horzontal gridlines
   plt.grid(axis = 'y')   adds horizontal gridlines
   plt.grid(axis = 'x')   adds vertical gridlines
   plt.grid(
       c = , # defines colour
       alpha = # defines transparency 0 to 1 e.g. .3
       linestyle = '-'
   )

object-orientated-style
   axes.grid()
   axes.grid(axis = 'y')
   axes.grid(axis = 'x')
   axes.grid(
       c = 'g',
       alpha = .3,
       linestyle = '-'
   )
"""



"""
LEGENDS

MATLAB-style
   plt.legend() # legend is position is defaulted to a visually bad place, top right inside image
   plt.legend(loc = 'center right') # legend is still positioned inside graph
   plt.legend(loc = 1.02, 0) # legend can be moved off plotting area 1 ,  = rightmost part of graph, so 1.02 is off

object-orientated-style
   axes.legend()
   axes.legend(loc = "center right")
   axes.legend(loc = (1.02,0) )
"""



"""
SAVE PLOT AS AN IMAGE

by default, .savefig() crops the plotted image
precede it with tight_layout() so it reajusts to include all elements

MATLAB-style
plt.tight_layout()
plt.savefig('data/exampleplot.png', dpi = 300)

object-orientated-style
fig.tight_layout()
fig.savefig('data/exampleplot.png', dpi = 300)
"""



"""
BOXPLOT  USING DIFFERENT LIBRARIES

https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

Note: even if you are using the pandas way of creating boxplot, 
you can still use matplotlib syntax to modify it
e.g. removing titles using matplotlib syntax
   plt.title('');
   plt.suptitle('');

Using Seaborn to create boxplots has these advantages over matplotlib
 - close integration with pandas data structures
 - dataset orientated API for examining relationships between multiple variables
 - specialised support for using categorical variables to show observations or aggregate statistics
 - concise control over matplotlib figure styling with several built-in themes
 - tool for choosing colour palettes that faithfully reveal patterns in data
"""

"""
df2 = pd.read_excel(filename, sheet_name='Sheet1')
df2 = df2.rename(columns={
   'Interest Paid': 'interest_paid',
})
interest_missing = df2['interest_paid'].isna()
df2.loc[interest_missing, 'interest_paid'] = df2['interest_paid'][:].interpolate(method = 'linear')



# boxplot using Matplotlib
toyotasienna = df2.loc[df2['car_type']=='Toyota Sienna','interest_paid'].values
vwgolfr = df2.loc[df2['car_type']=='VW Golf R','interest_paid'].values
plt.boxplot([toyotasienna, vwgolfr], labels = ['Toyota Sienna', 'VW Golf R'])

# boxplot using pandas requires less code
df2.boxplot(column = 'interest_paid', by = 'car_type')

# boxplot using seaborn
import seaborn as sns
sns.boxplot(x = 'car_type', y = 'interest_paid', data = df2)
"""



"""
HEATMAP USING SEABORN

a confusion matrix is a table that is often used to describe the performance of an ML classification model
i.e. it shows when predictions went wrong
in the example below each column represents numbers 0 to 9, values as predictions for values 0 to 9
e.g. 37 at topmost is trial 0, predicted the value of 0 37 times
3 at the bottom left represents incorrectly predicting value of 8 (row) when actual is 1 (column)
Note: this is just context as to where the data comes from, doesn't impact heatmap coding style

Heatmaps can be made with pure matplotlib, but it is a tremendous amount to code
"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

confusion = np.array([
   [37, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [0, 39, 0, 0, 0, 0, 0, 0, 2, 0],
   [0, 0, 41, 3, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 44, 0, 0, 0, 0, 1, 0],
   [0, 0, 0, 0, 37, 0, 0, 1, 0, 0],
   [0, 0, 0, 0, 0, 46, 0, 0, 0, 2],
   [0, 1, 0, 0, 0, 0, 51, 0, 0, 0],
   [0, 0, 0, 1, 1, 0, 0, 46, 0, 0],
   [0, 3, 1, 0, 0, 0, 0, 0, 44, 0],
   [0, 0, 0, 0, 0, 1, 0, 0, 2, 44]
])

# gradient sequential blue colour map
plt.figure(figsize = (6,6))
sns.heatmap(
   confusion,
   annot = True,
   cmap = 'Blues'
)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
"""



"""
HISTOGRAM USING MATPLOTLIB

series of intervals created and counting how many values fall into each interval
intervals are typically of equal size, but do not need to be
"""

"""
import pandas as pd
import matplotlib.pyplot as plt

# histogram all data with style applied for readability
df['interest_paid'].hist()
plt.style.use('seaborn')
# plt.xticks(rotation = 90) # note that if you try customising after applying a style, the style is cancelled out

# visualise subset of data
interest_paid_filtered = df.loc[:, 'interest_paid'] <= 210
df.loc[interest_paid_filtered, 'interest_paid'].hist(bins = 20, edgecolor='black')
"""



"""
SUBPLOTS

comparing data subsets sidde-by-side
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt








# %%
