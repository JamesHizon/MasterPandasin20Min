#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


# Help Link:
# https://towardsdatascience.com/how-to-master-pandas-8514f33f00f6

# data = pd.read_csv('happiness_with_continent.csv')
data = pd.read_csv('https://raw.githubusercontent.com/FBosler/you-datascientist/master/happiness_with_continent.csv')

# pd.read_clipboard

# This one I use rarely, but certainly works for smaller tables. 
# Just mark and copy (ctrl+c) a table from google sheets for example and run pd.read_clipboard().


# In[7]:


# Noteworthy parameters for the read_csv based functions (and read_clipboard).

# sep: 
# - separator for the columns (defaults to, but could also be tab)

# header: 
# - defaults to 'infer' (i.e., Pandas guesses as to what your header is),
# alternatives are an integer or a list of integers
# E.g., you could do header=3 and the dataframe would start with row 4
# If data has no header, use:
# header=None

# names:
# - names of the columns.
# If you want to use this parameter to override whatever column names Pandas had inferred,
# Should specify header=0, (or whatever line your columns names are in),
# If you do not do this, you will have your names as the column names and then the original
# column names in the first row.


# In[11]:


# 1) Inspecting - First, last, random rows
# Preview first 5 rows:
data.head(5)
# Last 5 rows:
data.tail(5)
# 5 random rows:
data.sample(5)

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 8)


# In[9]:


# 2) Inspecting - shape, columns, index, info, describe
# 1704 rows, 27 columns:
data.shape
# Returns dimensions of the DataFrame.
# Ex) Out: (1704, 27)


# In[10]:


data.columns


# In[13]:


data.sort_values(by='Year')


# In[14]:


data.sort_values(by=['Year','Country name'])


# In[15]:


data.sort_values(by=['Country name','Year'])


# In[16]:


data.sort_values(by='Year', ascending=True)


# In[17]:


data.sort_values(
  by=['Country name','Year'], 
  ascending=[False,True]
)


# In[19]:


# Filtering - columns
# Selecting one column:

data['Year']


# In[20]:


# Multiple columns:

data[['Country name','Life Ladder']].sample(5)


# In[22]:


# Filtering rows:
# Being able to select specific columns is only halfway there.
# However, selecting rows is just as easy.
# Can select one row or multiple rows by index.

data.iloc
# Allows selecting rows (and optionally columns) 
# By position (i.e., by the number of the row).

# Selecting one row:
# Looks like: data.iloc[row_number (,col_number)],
# Where the part in the parentheses is optional.


# In[23]:


data.iloc[10]


# In[27]:


data.iloc[10, 5]


# In[28]:


# Selecting multiple rows:

# data.iloc[start_row:end_row (, start_col:end_col)],
# where the part in the parenthese is optional.


# In[29]:


data.iloc[903:907]


# In[30]:


# Select which rows and columns to select:

data.iloc[903:907, 0:3]


# In[31]:


# Allow selecting rows (and columns) by:
# 1) label/index or
# 2) with boolean/conditional lookup

data.loc


# In[32]:


# Set Country name as the dataframe's index:
data.set_index('Country name', inplace=True)


# In[34]:


# Sets a new index on the DataFrame.

#set_index

# inplace=True -> Makes sure that the DataFrame will be changed.

data.sample(5)


# In[36]:


# Can now see that the DataFrame lost its row numbers (the previous) index
# and gained a new index:

data.index


# In[38]:


# loc - Selecting row(s) by one index label:

# Syntax: data.loc[index_label (,col_label)]

data.loc['United States']


# In[39]:


# Selecting rows and column by index lable and column label:
data.loc['United States', 'Life Ladder']


# In[40]:


data.loc[['United States','Germany']].sample(5)


# In[44]:


# Selecting rows and columns by multiple index labels:

# You can also specify the column names for the selected rows
# That you want to return.

data.loc[
    ['Germany','United States'],#Row Indices
    ['Year','Life Ladder']#Column Indices
].sample(5)
# NOTE: Splitting statement into two lines makes for better readability. 


# In[47]:


# Selecting row(s) by a range of index labels:

# ('Denmark':'Germany') vs. ('903:907') for iloc.

# Assume your index is sorted, or you sorted it before selecting a range,
# Can do the following:

data.loc[
    'Denmark':'Germany',
    ['Year','Life Ladder']
].sample(5)


# In[49]:


# loc - boolean/conditional lookup

# Boolean or conditional lookup is where the meat is.
# Whenever selecting row, this happens by overlaying the DataFrame
# Of True and False values.

# In the following example,
# Create a small DataFrame with the index:
# ['A', 'B', 'A', 'D']
# and some random values between 0 and 10 as values.

# Then, create an overlay with the same Index with the values:
# [True, False, True, False]..

# Use df.loc[overlay] to only select the rows with True value for their index.

from numpy.random import randint
index = ['A', 'B', 'A', 'D']

## Create dummy DataFrame ##

df = pd.DataFrame(
    index = index,
    data = {
        'values' :randint(10, size=len(index))
})
print('DataFrame:')
print(df)


# In[50]:


## Create dummy overlay ##
overlay = pd.Series(
    index = index,
    data = [True, False, True, False]
)
print('\n Overlay:')
print(overlay)


# In[51]:


## Select only True rows ##
print('\nMasked DataFrame:')
print(df.loc[overlay])


# In[52]:


# Same logic -> Select rows based on a (or multiple) condition(s).

# First create a boolean mask like this:
data['Life Ladder'] > 4


# In[53]:


# Use mask to only select the rows that meet the specified condition
# Like this:

# Option 1:
data.loc[data['Life Ladder'] > 4]

# Alternative:
#condition = data['Life ladder'] > 4
#data.loc[condition]


# In[59]:


# Option 1, as well as, the alternative yield precisely the same result.
# However, the alternative is a little more legible.
# The improved legibility becomes even more apparent when applying
# Multiple conditions:

life_condition = data['Life Ladder'] > 4
year_condition = data['Year'] > 2014
social_condition = data['Social support'] > .5


# In[60]:


data.loc[life_condition & year_condition & social_condition]


# In[61]:


# NOTE: We used & (bitwise and) to filter for rows s.t. multiple conditions
# Apply at the same time.
# Can use | (bitwise or) to filter for columns, where one of the conditions
# Applies.


# In[62]:


# Advanced Conditional Lookup with Custom Formulas

# It is also possible and quite easy to use customized functions as
# A condition and apply them to select columns.

# In the following example,
# We only select years that are cleanly divisible by three and 
# Continents that contain the word America.
# The case is contrived but makes a point.

cond_year = data['Year'].apply(lambda x: x%3 == 0)
cond_america = data['Continent'].apply(lambda x: 'America' in x)

data[cond_year & cond_america]


# In[63]:


# Argentina -> Venezuela: Countries in NORTH and SOUTH America.

# Instead of lambda (anonymous) functions,
# You could also define and use much more complicated functions.
# Could even (not that I recommend it) make API calls
# in a custom function and use the results of the calls to filter your dataframe.


# In[ ]:


# 4) Analytical Functions

# Now that we are comfortable with filtering and sorting the data front
# To back and vice versa, let's move to some more advanced analytical
# Functionalities.


# In[65]:


# Standard Functions:

# Like the read functions,
# There are also a lot of analytical functions implemented in Pandas.

# I will highlight and explain the ones that I use most frequently.

# Used most frequently:

# However, and that's part of the beauty of it, even I will
# Find new useful functions from time to time.

# So, never stop reading and exploring!

# 1) max/min

# 2) sum

# 3) mean/median/quantile

# 4) idxmin/idxmax

# NOTE: All functions can be applied column-wise, but also row-wise.

# - The row-wise application makes very little sense in our example.

# - However, frequently, you have data, where you want to compare

# Different columns, in which case, the row-wise application does make sense.

# Whenever we call the aforementioned functions,

# axis=0 - is passed (for column-wise application).

# However, we can override this parameter and pass 

# axis=1 (for row-wise application).


# In[ ]:


# Calling max/min:

# Calling max() on data,

# - Will return (wherever possible) the maximum for each column.

# min() does the exact opposite.


# In[66]:


data.max() # COLUMNWISE MAXIMUM


# In[68]:


data.max(axis=1) # ROW-WISE MAXIMUM


# In[69]:


data.sum() # Will return (wherever possible) the sum for each column.


# In[70]:


data.mean()


# In[71]:


data.median()


# In[73]:


data.quantile(q=.8) # 80-th percentile


# In[75]:


# Calling idxmax or idxmin on data will return the index of the row where

# The first minimum/maximum is found.

# However, it is only possible to call this on columns

# With some ordinality to them.

data.iloc[:,:-1].idxmax() # We exclude the Continent (last) Column


# In[76]:


# We can then say, Denmark has the highest Life Ladder.

# Qatar has the highest Log GDP per capita.

# New Zealand has the highest value for Social support.

# idxmin works the same as idxmax.


# In[ ]:


# Apply/Custom Functions:


# In[ ]:


# Two types of Custom Functions:

# Named Functions

# Lambda Functions


# In[78]:


# Named Functions:

# FUNCTION:

def above_1000_below_10(x):
    try: # Test a block for errors
        pd.to_numeric(x) # Convert to numeric values
    except: # Handle the error.
        return 'no number column'
    
    if x > 1000:
        return 'above_1000'
    elif x < 10:
        return 'below_10'
    else:
        return 'mid'


# In[79]:


data['Year'].apply(above_1000_below_10)


# In[80]:


data['Year'].apply(above_1000_below_10)


# In[81]:


# Defined a function called "above_1000_below_10"
# And applied that to our function.

# Function initially checks, if the value is convertible to a number
# And if not, will return "no number column".

# Otherwise, the function returns above_1000 if value is above 1000,
# And below_10 if the value is below 10.
# Else, it returns mid.


# In[82]:


# Lambda Functions:

# Short, throw-away functions for one-time use only.
# The name sounds clunky, but once you got the hang of it, they are quite convenient.

# EXAMPLE:

# Could split the continent column on space and then grab the last word of the results.

data['Continent'].apply(lambda x: x.split(' ')[-1])


# In[83]:


# NOTE: Both, named and lambda functions, we applied to individual columns
# As opposed to the entire dataframe.

# When applying a function to a particular column, the function goes row by row.

# When applying a function to a whole DataFrame, the function goes column by column

# And is applied to the entire column, 

# Then and has to be written a little differently, like so:

def country_before_2015(df):
    if df['Year'] < 2015:
        return df.name
    else:
        return df['Continent']
    
# NOTE: the axis=1
data.apply(country_before_2015, axis =1)


# In[ ]:


# In this example, we also go row by row (as specified by the axis=1).

# Return the row (which happens to be the index)
# When the Year of that row is smaller than 2015 or else the continent
# Of that row.

# Tasks like this is for conditional data cleaning.


# In[84]:


# Combining Columns:

# Sometimes you want to add, subtract or combine two or more columns.

data['Year'] + data['Life Ladder']


# In[85]:


data['Continent'] + '_' + data['Year'].astype(str)


# In[ ]:


# NOTE: In the example above, we want to combine two columns as strings.
# - To do this, we have to interpret data['Year'] as a string.
# - We do that by using .astype(str) on the column.
# - For the sake of brevity, we will not dive into types and type conversion.


# In[87]:


# Groupby:

# So far, all the calculations we have applied were to the entire set,
# a row, or a column. However - and this is where it gets exciting -
# we can also group our data and calculate metrics for the individual groups.

# So, let's say we want to know the highest "Life Ladder" value per country.

# Looking for max of "Life Ladder" index, and we group the data by Country:
data.groupby(['Country name'])['Life Ladder'].max()


# In[90]:


# Say, we want per year the country with the highest "Life Ladder".

data.groupby(['Year'])['Life Ladder'].idxmax()


# In[92]:


# Or, multi-level groups, let's say we want per continent/year
# Combination the entry with the highest "Life Ladder."

# Let's say, we want per continent/per year combination 
# The entry with the highest "Life Ladder".

data.groupby(['Year', 'Continent'])['Life Ladder'].idxmax()


# In[95]:


# Like before, we can use many standard functions or custom functions
# (named or unnamed) to, for example, return a random country per group:
import numpy as np

def get_random_country(group):
    return np.random.choice(group.index.values)


# In[96]:


# Named function:

data.groupby(['Year', 'Continent']).apply(get_random_country)


# In[97]:


# NOTE: Groupby always returns ONE value per group.
# So, unless you are grouping by a column that only contains unique values,
# The result will be a smaller (an aggregated) dataset.


# In[102]:


# transform

# Sometimes, you don't want only one value per group, but instead
# Want the value you calculated for the group for every row belonging
# To that group. You can do this the following way:

data.groupby(['Country name'])['Life Ladder'].transform(sum)


# In[103]:


# Here, we get the sum of all Life Ladder scores for a country.

# Can also do:

data.groupby(['Country name'])['Life Ladder'].transform(np.median)


# In[104]:


# Here, we obtained the median per country.
# We can then calculate the difference to the value
# Every single year like this (as transform preserves the index):


# In[110]:


data.groupby(['Country name'])['Life Ladder'].transform(np.median)
- data['Life Ladder']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




