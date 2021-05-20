#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

# import operator for dictionary sorting operations
import operator


# In[2]:


# import ames training dataset
df = pd.read_csv('AmesHousing.csv')
# observe the shape of dataset
print("Shape of dataset:", df.shape)


# In[3]:


# panda configuration to show all columns
pd.set_option('display.max_columns', None)

# print out and show the variables for training dataset
df.head(10)


# In[4]:


# drop order and id since it is useless
df = df.drop(['Order','PID'], axis = 1)


# # Removing irrelavent columns and rows

# In[5]:


# after studying the dataset, the variable in MSZoning includes non-residential areas and so we need to exlude them
print(df['MS Zoning'].value_counts())  


# In[6]:


# only we can see that from this training data, there is FV and C (non-residential)
# we will need to exclude them from this dataset (need to do the same thing to the test set also)

df.drop(df[(df['MS Zoning'] == 'FV')].index, inplace = True)
df.drop(df[(df['MS Zoning'] == 'C (all)')].index, inplace = True)
df.drop(df[(df['MS Zoning'] == 'I (all)')].index, inplace = True)
df.drop(df[(df['MS Zoning'] == 'A (agr)')].index, inplace = True)
print(df['MS Zoning'].value_counts())
print(df.shape,"\n")


# In[7]:


# reset index after dropping rows
df = df.reset_index(drop=True)


# In[8]:


# check to see are there duplicate rows
# don't need sales price to check duplicated rows
duplicated = df[df.duplicated(keep=False)]
print("Duplicated row(s): ", len(duplicated))
# we can see that there is no duplication


# # Data Cleaning and EDA (removing null values)

# In[9]:


# show datatype and non-null count
print(df.info())


# In[10]:


# check the total number of null in each column
print(df.isnull().sum().to_string())


# In[11]:


# Alley, PoolQC, Fence, MiscFeature have a lot of null values
# we will need to observe what they mean in order to decide whether to drop them
# we will not drop "alley" and "fence" because they might have some relation with price
# the NA for those categories just mean that the property does not have it

# as for poolQC and MiscFeature, we will evaluate each of them
# for PoolQC, we will drop it because there is already a variable with the pool area which we can focus on
# for MiscFeature, we will also drop it because there is MiscValue which is similar but more helpful in price prediction
df = df.drop(['Pool QC', 'Misc Feature', 'Mo Sold'], axis = 1)
DROPPED_VARIABLE  = ['Pool QC', 'Misc Feature', 'Mo Sold']


# # Null Values for Categorical Value

# In[12]:


# need to observe the other null values that might be possible to be add replacements
# print value counts for all 'objects' with more than 1 null value
# use to look at the type, percentage of nulls to decide what to do with empty values
# dtype "O" means object
def objects_w_null(df):
  for i in df:
    if df[i].dtype == 'O':
      if df[i].isnull().sum() > 0: 
        print(df[i].value_counts())  
        print("Total Null Values: " + str(df[i].isnull().sum()))
        # round up to two decimal place: round(num, 2)
        print("Percentage of Nulls = " + str(np.round((df[i].isnull().sum() / len(df.index) * 100), 2)) + "%") 
        print("\n")
      
objects_w_null(df)


# Alley
# - not considered as missing value
# - the null means "No Alley"
# 
# Mas Vrn Type**
# - Some missing value 
# - maybe fill with the mode which is "None"
# 
# Bsmt Qual
# - not considered as missing value
# - the null for means "No Basement"
# 
# Basmt Cond
# - same as BsmtQual
# 
# Bsmt Exposure**
# - should be same as BastQual
# - but there is two extra value missing need to find out
# 
# BsmtFin Type 1
# - same as BsmtQual
# 
# BsmtFin Type 2**
# - same as BsmtQual
# - but extra one missing value
# 
# Electrical**
# - one missing need to fill it
# 
# Fireplace Qu
# - missing value means no fireplace
# 
# Garage Type
# - missing value means no garage
# 
# Garage Finish/Qual/Cond**
# - missing value means no garage
# - extra two missing need find out
# 
# Fence
# - missing value means no fence
# 
# Pool QC
# - missing value means no piil
# 
# Misc Feature
# - missing value means no extra feature

# In[13]:


# find out what are those real missing value first
def difference(df, base, compare):
    first = pd.isnull(df[base])
    second = pd.isnull(df[compare])
    arr = []
    for i in first.index:
        if first[i] != second[i]:
            arr.append(i)
    return arr


# ## For BsmtFin Type 2

# In[14]:


missing_index = difference(df, "BsmtFin Type 1", "BsmtFin Type 2")
print("Missing value index is", missing_index)


# In[15]:


# we can see that basement is available for this particular resident but bsmtfin type 2 is null
# we will fill it with the mode[0]
# index 0 only get the value without the type

def replace_specific(df, column, index):
    mode = df[column].mode()[0]
    for i in index:
        df[column].values[i] = mode 


# In[16]:


replace_specific(df, "BsmtFin Type 2", missing_index)


# ## For Bsmt Exposure

# In[17]:


missing_index = difference(df, "BsmtFin Type 1", "Bsmt Exposure")
print("Missing value index is", missing_index)
replace_specific(df, "Bsmt Exposure", missing_index)


# ## For Electrical

# In[18]:


# this one is different from the others, no cell to compare with, just missing value alone
# create another function
def seek_null(df, column):
    is_null = pd.isnull(df[column])
    arr = []
    for i in is_null.index:
        if is_null[i] == True:
            arr.append(i)
    return arr


# In[19]:


missing_index = seek_null(df, "Electrical")
print("Missing value index is", missing_index)
replace_specific(df, "Electrical", missing_index)


# ## For Garage Finish/Qual/Cond

# In[20]:


column = ["Garage Finish", "Garage Qual", "Garage Cond"]

for i in column:
    missing_index = difference(df, "Garage Type", i)
    print("Missing value index in", i, "is", missing_index)
    
# same two row got missing values
for i in column:
    replace_specific(df, i, missing_index)


# ## For Mas Vrn Type

# In[21]:


# new function need to be created because this is to be compare with a numerical column
def difference_w_num(df, num, cat):
    first = []
    
    for i in df[num]:
        if i <= 0 or pd.isnull(i): 
            first.append(False)
        else:
            first.append(True) # got Mas Vnr

    second = pd.isnull(df[cat])
    arr = []
    for i in second.index:
        if second[i] == True: 
            if first[i] == second[i]: # got area but not type means missing value
                arr.append(i)
    return arr

col_num = "Mas Vnr Area"
col_cat = "Mas Vnr Type"

missing_index = difference_w_num(df, col_num, col_cat)
print("Missing value index is", missing_index)


# # Replacing all NA (not null) in object

# In[22]:


objects_w_null(df) 
# now fill all the values that are NA with None, indicating that it is unavailable


# In[23]:


column = ["Alley", "Mas Vnr Type", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Fireplace Qu", "Garage Type",
         "Garage Finish", "Garage Qual", "Garage Cond", "Fence"]

df[column] = df[column].fillna("None")
    
# Now there are no null values in object


# # Null values for numerical data

# In[24]:


# np.number

def numeric_w_null(df):
  for i in df:
    if df[i].dtype != 'O': # not object
      if df[i].isnull().sum() > 0: 
        print(i)
        print("Total Null Values: " + str(df[i].isnull().sum()))
        # round up to two decimal place: round(num, 2)
        print("Percentage of Nulls = " + str(np.round((df[i].isnull().sum() / len(df.index) * 100), 2)) + "%") 
        print("\n")
      
numeric_w_null(df)


# ## For Lot Frontage

# In[25]:


column = "Lot Frontage"
missing_index = seek_null(df, column)
# print("Missing value index is", missing_index)


# In[26]:


# assume null in lot frontage means 0 (no frontage)
# change to median or mean if prediction agrees
df[column] = df[column].fillna(0)


# ## For Mas Vrn Area

# In[27]:


column = "Mas Vnr Area"
missing_index = seek_null(df, column)
# print("Missing value index is", missing_index)


# In[28]:


# assume null in Mas Vnr Area means 0 (no Mas Vrn)
# This matches with the None in type which are mostly 0 area
df[column] = df[column].fillna(0)


# ## BsmtFin SF 1 and 2, Bsmt Unf SF, Total Bsmt SF

# In[29]:


column = "BsmtFin SF 1"
missing_index = seek_null(df, column)
print("Missing value index is", missing_index)


# In[30]:


# the null should be 0 for square feet
# This matches with the None in type which
column = ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF"]
for i in column:
    df[i] = df[i].fillna(0)


# ## For Bsmt Full and Half Bath

# In[31]:


column = "Bsmt Full Bath"
missing_index = seek_null(df, column)
print("Missing value index is", missing_index)


# In[32]:


# the null should be 0 as according to other columns, this 2 rows has no basement
column = ["Bsmt Full Bath", "Bsmt Half Bath"]
for i in column:
    df[i] = df[i].fillna(0)


# ## For Garage Yr Blt

# In[33]:


column = "Garage Yr Blt"
missing_index = seek_null(df, column)
# print("Missing value index is", missing_index)


# In[34]:


# it is irrelavent to put zero, mean or median for year 
# and so we will fill the null with year build instead so that is will be more appropriate 
# BUT this will cause two column to have similar data which is redundant
# so we will delete this column
column = "Garage Yr Blt"
df = df.drop([column],axis = 1)


# ## For Garage Cars and Area

# In[35]:


column = "Garage Cars"
missing_index = seek_null(df, column)
print("Missing value index is", missing_index)


# In[36]:


# we can assume that an unfinished has no area and can fit no cars
# might use mean or median of Unf if allows 
column = ["Garage Cars", "Garage Area"]
for i in column:
    df[i] = df[i].fillna(0)


# In[37]:


# now there are no more null values in this dataset
print(df.isnull().sum().to_string())


# # Label Encoder

# In[38]:


# for categorical ordinal features 
# make it into numeric manually (cause wan those none to be 0) 
df['Street'].replace(to_replace = ['Grvl', 'Pave'], value = [0, 1], inplace = True)
df['Alley'].replace(to_replace = ['None', 'Grvl', 'Pave'], value = [0, 1, 2], inplace = True)
df['Lot Shape'].replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)
df['Land Contour'].replace(to_replace = ['Low', 'HLS', 'Bnk', 'Lvl'], value = [0, 1, 2, 3], inplace = True)
df['Utilities'].replace(to_replace = ['ELO', 'NoSeWa', 'AllPub', 'NoSewr'], value = [0, 1, 2, 3], inplace = True)
df['Lot Config'].replace(to_replace = ['FR3', 'FR2', 'CulDSac', 'Corner', 'Inside'], value = [0, 1, 2, 3, 4], inplace = True)
df['Land Slope'].replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)
df['Bldg Type'].replace(to_replace = ['Twnhs', 'TwnhsE', 'Duplex', '2fmCon', '1Fam'], value = [0, 1, 2, 3, 4], inplace = True)
df['Exter Qual'].replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df['Exter Cond'].replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df['Bsmt Qual'].replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df['Bsmt Cond'].replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df['Bsmt Exposure'].replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
df['BsmtFin Type 1'].replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
df['BsmtFin Type 2'].replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
df['Heating QC'].replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df['Central Air'].replace(to_replace = ['N', 'Y'], value = [0, 1], inplace = True)
df['Kitchen Qual'].replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df['Fireplace Qu'].replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df['Garage Finish'].replace(to_replace = ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)
df['Garage Qual'].replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df['Garage Cond'].replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df['Paved Drive'].replace(to_replace = ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)
df['Fence'].replace(to_replace = ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)
df['Functional'].replace(to_replace = ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 2, 3, 4, 4, 5], inplace = True)
df['Electrical'].replace(to_replace = ['FuseP', 'FuseF', 'FuseA', 'Mix', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)


# # Change Year Built to Age Sold (count since built) and Age Sold (Counting from remodel)

# In[39]:


df['Age Sold'] = df['Yr Sold'] - df['Year Built']
df['Age Sold (Remod)'] = abs(df['Yr Sold'] - df['Year Remod/Add'])
df = df.drop(['Year Built', 'Yr Sold', 'Year Remod/Add'], axis = 1)
DROPPED_VARIABLE += ['Year Built', 'Yr Sold', 'Year Remod/Add']

NEW_VARIABLE = ['Age Sold', 'Age Sold (Remod)']


# # Linearly dependent Variables

# In[40]:


# if any if the sf does not add up, print assertion error
assert not (df['Gr Liv Area'] != (df['1st Flr SF'] + df['2nd Flr SF'] + df['Low Qual Fin SF'])).any(), "Some SF do not add up"


# In[41]:


assert not (df['Total Bsmt SF'] != (df['BsmtFin SF 1'] + df['BsmtFin SF 2'] + df['Bsmt Unf SF'])).any(), "Some SF do not add up"


# In[42]:


# create a total SF and put them together
df['Total SF'] = df['Gr Liv Area'] + df['Total Bsmt SF']
NEW_VARIABLE.append('Total SF')


# In[43]:


df['Total Porch SF'] = df['Wood Deck SF'] + df['Open Porch SF'] + df['Enclosed Porch'] + df['3Ssn Porch'] + df['Screen Porch']
NEW_VARIABLE.append('Total Porch SF')


# In[44]:


df['Total Bath'] = df['Full Bath'] + df['Bsmt Full Bath'] + 0.5* df['Half Bath'] + 0.5 * df['Bsmt Half Bath']
NEW_VARIABLE.append('Total Bath')


# # Change Neighborhood Variable

# In[45]:


new_var = (df.groupby('Neighborhood')['SalePrice'].mean()  / 
           df.groupby('Neighborhood')['Total SF'].mean()).to_dict().items()
          
for j in new_var:
    df.loc[df['Neighborhood']== j[0], 'Mean Price per SF'] = j[1]

NEW_VARIABLE.append('Mean Price per SF')


# In[46]:


ordinal_cat = ['Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config', 'Land Slope', 'Bldg Type', 
              'Overall Qual', 'Overall Cond', 'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 
              'Heating QC', 'Central Air', 'Kitchen Qual', 'Fireplace Qu', 'Garage Finish', 'Garage Qual', 'Garage Cond', 
               'Paved Drive', 'Fence', 'Functional', 'Electrical']

df[ordinal_cat] = df[ordinal_cat].astype('int')

nominal_cat = ["MS SubClass", "MS Zoning",  "Condition 1", "Condition 2", "House Style", "Roof Style", "Roof Matl", 
               "Exterior 1st", "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", 
               "Garage Type", "Sale Type", "Sale Condition", "Neighborhood"]

discrete_var = ['Bedroom AbvGr', 'Bsmt Full Bath', 'Bsmt Half Bath',
                    'Fireplaces', 'Full Bath', 'Garage Cars', 'Half Bath',
                    'Kitchen AbvGr', 'TotRms AbvGrd', 'Age Sold', 'Age Sold (Remod)', 'Total Bath']


# In[47]:


to_drop = ordinal_cat + nominal_cat + discrete_var + ['SalePrice']
continuous_variable = df.drop(to_drop, axis = 1)

CONTINUOUS_VARIABLE = continuous_variable.columns.tolist()
DISCRETE_VARIABLE = discrete_var
NOMINAL_VARIABLE = nominal_cat
ORDINAL_VARIABLE = ordinal_cat
TARGET = ["SalePrice"]


# # Check Correlation

# In[48]:


def plot_correlation (corr, title): 
    fig, ax = plt.subplots(figsize = (16,12))
    ax.set_title(title, fontsize = 20)
    blue = sns.diverging_palette(240, 20, as_cmap=True)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    # darker colour means more correlation
    sns.heatmap(corr, vmin=-1, vmax=1, cmap = blue, mask = mask, xticklabels = corr.columns.values, 
                yticklabels = corr.columns.values, ax = ax)


# In[49]:


# Pearson for Nominal Data 
new_columns = pd.concat([df[CONTINUOUS_VARIABLE], df[TARGET]], axis = 1)
corr = new_columns.corr(method ='pearson')
plot_correlation (corr, "Pearson Correlation")


# In[50]:


strong = 0.65
weak = 0.35
uncorrelated = 0.1

pearson_strongly_correlated = set()
pearson_somehow_correlated = set()
pearson_weakly_correlated = set()
pearson_uncorrelated = set()

corr = new_columns.corr(method ='pearson')['SalePrice'].sort_values(ascending=False)
corr = corr.drop('SalePrice')
print(corr)
corr = corr.to_dict().items()


# In[51]:


for variable in corr: 
    if (abs(variable[1]) > strong): 
        pearson_strongly_correlated.add(variable[0])
        
    elif ((abs(variable[1]) <= strong) & (abs(variable[1]) > weak)): 
        pearson_somehow_correlated.add(variable[0])
        
    elif ((abs(variable[1]) <= weak) & (abs(variable[1]) > uncorrelated)): 
        pearson_weakly_correlated.add(variable[0])
        
    elif (abs(variable[1]) <= uncorrelated):
        pearson_uncorrelated.add(variable[0])
        


# In[52]:


print("Strong: ", pearson_strongly_correlated) # 0.65 - 1.0
print("\nSomehow: ", pearson_somehow_correlated) # 0.35 - 0.65
print("\nWeak: ", pearson_weakly_correlated) # 0.1 - 0.35
print("\nUncorrelated: ", pearson_uncorrelated) # 0.0 - 0.1


# In[53]:


# Spearman for Ordinal and Discrete Data (suitable as they have ordering higher number means more / higher number means better)
new_columns = pd.concat([df[ORDINAL_VARIABLE], df[DISCRETE_VARIABLE], df[TARGET]], axis = 1)
new_columns = new_columns.astype('int') # Change to int before doing correlation
corr = new_columns.corr(method ='spearman')
plot_correlation (corr, "Spearman Correlation")


# In[54]:


spearman_strongly_correlated = set()
spearman_somehow_correlated = set()
spearman_weakly_correlated = set()
spearman_uncorrelated = set()

corr = new_columns.corr(method ='spearman')['SalePrice'].sort_values(ascending=False)
corr = corr.drop('SalePrice')
print(corr)
corr = corr.to_dict().items()


# In[55]:


for variable in corr: 
    if (abs(variable[1]) > strong): 
        spearman_strongly_correlated.add(variable[0])
        
    elif ((abs(variable[1]) <= strong) & (abs(variable[1]) > weak)): 
        spearman_somehow_correlated.add(variable[0])
        
    elif ((abs(variable[1]) <= weak) & (abs(variable[1]) > uncorrelated)): 
        spearman_weakly_correlated.add(variable[0])
        
    elif (abs(variable[1]) <= uncorrelated):
        spearman_uncorrelated.add(variable[0])


# In[56]:


print("Strong: ", spearman_strongly_correlated) # 0.65 - 1.0
print("\nSomehow: ", spearman_somehow_correlated) # 0.35 - 0.65
print("\nWeak: ", spearman_weakly_correlated) # 0.1 - 0.35
print("\nUncorrelated: ", spearman_uncorrelated) # 0.0 - 0.1


# In[57]:


# add both together into one (can be use to test prediction later on)
strongly_correlated = pearson_strongly_correlated | spearman_strongly_correlated
somehow_correlated = pearson_somehow_correlated | spearman_somehow_correlated
weakly_correlated = pearson_weakly_correlated | spearman_weakly_correlated
uncorrelated = pearson_uncorrelated | spearman_uncorrelated


# # Binary Features

# Some variable such as pool and fireplace can be observe by looking at the the existance 0 or 1

# In[58]:


binary_variables = {"Got Alley": "Alley", 
                    "Got 2nd Flr": "2nd Flr SF",
                    "Got Frontage": "Lot Frontage",
                     "Got Bsmt": "Bsmt Qual", 
                     "Got Fireplace": "Fireplace Qu", 
                     "Got Garage": "Garage Area", 
                     "Got Porch": "Total Porch SF", 
                     "Got Pool": "Pool Area", 
                     "Got Misc": "Misc Val" }

for factor_column, column in binary_variables.items():
    df[factor_column] = df[column].apply(lambda x: 1 if x > 0 else 0)

BINARY_VARIABLES = list(binary_variables.keys())
NEW_DUMMY = []
NEW_DUMMY.extend(BINARY_VARIABLES)


# In[59]:


df[BINARY_VARIABLES].head()


# In[60]:


x = "Gr Liv Area"
y = "SalePrice"


# ## Got Alley*

# Mostly no alley but the one with alley are cheaper in price

# In[61]:


hue = 'Got Alley'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);

df = df.drop(columns=['Alley'])
DROPPED_VARIABLE.append('Alley')
ORDINAL_VARIABLE.remove('Alley')


# ## Got 2nd Floor

# Bigger Liv Area seems to have 2nd floor
# Seems not much effect on the price 

# In[62]:


hue = 'Got 2nd Flr'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);

df = df.drop(columns=['2nd Flr SF'])
DROPPED_VARIABLE.append('2nd Flr SF')
CONTINUOUS_VARIABLE.remove('2nd Flr SF')


# ## Got Frontage (DROP)

# Most house got frontage
# Only a few don't have and no make any different on price

# In[63]:


hue = 'Got Frontage'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);

df = df.drop(columns=['Got Frontage', 'Lot Frontage'])
DROPPED_VARIABLE.append('Lot Frontage')
NEW_DUMMY.remove('Got Frontage')
CONTINUOUS_VARIABLE.remove('Lot Frontage')


# ## Got Bsmt*

# Mostly got basement
# The one without basement seems to have cheaper price

# In[64]:


hue = 'Got Bsmt'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# ## Got Fireplace*

# Smaller house no fireplace
# No fireplace seems to be cheaper

# In[65]:


hue = 'Got Fireplace'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# ## Got Garage*

# Mostly got garage
# No garage cheaper

# In[66]:


hue = 'Got Garage'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# ## Got Porch

# Mostly got porch
# No porch cheaper (?)

# In[67]:


hue = 'Got Porch'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# ## Got Pool (DROP)

# Mostly no pool
# no difference in price

# In[68]:


hue = 'Got Pool'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Got Pool', 'Pool Area'])
DROPPED_VARIABLE.append('Pool Area')
NEW_DUMMY.remove('Got Pool')
CONTINUOUS_VARIABLE.remove('Pool Area')


# ## Got Misc (DROP)

# Mostly no misc
# no difference in price

# In[69]:


hue = 'Got Misc'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Got Misc', 'Misc Val'])
DROPPED_VARIABLE.append('Misc Val')
NEW_DUMMY.remove('Got Misc')
CONTINUOUS_VARIABLE.remove('Misc Val')


# # Nominal Feature
# Explore categorical nominal features (cannot be observe previously using correlation function)
# Since Gr Liv Area is the highest correlated continuous data (excluding self created ones)
# We will use Gr Liv Area vs Sale Price to observe nominal features
# Decide whether to keep them 

# ## MS SubClass (DROP)
# There are many different categories and are mostly mentioning the floors (relate to got second floor) and the age (relate to age sold) Safe to drop

# In[70]:


hue = 'MS SubClass'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['MS SubClass'])
DROPPED_VARIABLE.append('MS SubClass')
NOMINAL_VARIABLE.remove('MS SubClass')


# ## MS Zoning (DROP)
# dont really have a pattern here can delete

# In[71]:


hue = 'MS Zoning'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['MS Zoning'])
DROPPED_VARIABLE.append('MS Zoning')
NOMINAL_VARIABLE.remove('MS Zoning')


# ## Condition (TO DUMMY)
# Another factor that might me important as it means the proximity to varioua conditions
# but there are a lot of condition we will need to reduce/combine them
# we can see that offsite and railroad seems to have higher prices
# street seems to have lower prices 
# 
# condition 2 are mostly normal and so we will remove it

# In[72]:


df['Condition 1'].replace(to_replace = ['Artery', 'Feedr'], value = 'Street', inplace = True)
df['Condition 1'].replace(to_replace = ['RRNn', 'RRAn', 'RRNe', 'RRAe'], value = 'Railroad', inplace = True)
df['Condition 1'].replace(to_replace = ['PosN', 'PosA'], value = 'Park', inplace = True)

df['Condition 2'].replace(to_replace = ['Artery', 'Feedr'], value = 'Street', inplace = True)
df['Condition 2'].replace(to_replace = ['RRNn', 'RRAn', 'RRNe', 'RRAe'], value = 'Railroad', inplace = True)
df['Condition 2'].replace(to_replace = ['PosN', 'PosA'], value = 'Park', inplace = True)


# In[73]:


hue = 'Condition 1'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# In[74]:


# create dummy variables for condition 1, not needed for norm (drop norm)
new_dummy = pd.get_dummies(df["Condition 1"], prefix="Near")
new_dummy = new_dummy.drop(columns=['Near_Norm'], axis = 1)
df = pd.concat([df, new_dummy], axis=1)
df = df.drop(columns=['Condition 1'])


DROPPED_VARIABLE.append('Condition 1')
NOMINAL_VARIABLE.remove('Condition 1')

for n in new_dummy:
    NEW_DUMMY += [n]


# In[75]:


hue = 'Condition 2'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);

df = df.drop(columns=['Condition 2'])
DROPPED_VARIABLE.append('Condition 2')
NOMINAL_VARIABLE.remove('Condition 2')


# ## House Style (DROP)
# Similar to 'Has Second Floor'

# In[76]:


hue = 'House Style'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['House Style'])
DROPPED_VARIABLE.append('House Style')
NOMINAL_VARIABLE.remove('House Style')


# ## Roof (DROP)
# do not really have a pattern for roof style
# basically all same material for rood matl

# In[77]:


hue = 'Roof Style'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Roof Style'])
DROPPED_VARIABLE.append('Roof Style')
NOMINAL_VARIABLE.remove('Roof Style')


# In[78]:


hue = 'Roof Matl'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Roof Matl'])

DROPPED_VARIABLE.append('Roof Matl')
NOMINAL_VARIABLE.remove('Roof Matl')


# ## Exterior (DROP)
# Too many different variables and seems to stack together with no pattern

# In[79]:


hue = 'Exterior 1st'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Exterior 1st'])

DROPPED_VARIABLE.append('Exterior 1st')
NOMINAL_VARIABLE.remove('Exterior 1st')


# In[80]:


hue = 'Exterior 2nd'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Exterior 2nd'])

DROPPED_VARIABLE.append('Exterior 2nd')
NOMINAL_VARIABLE.remove('Exterior 2nd')


# ## Mas Vnr Type (TO DUMMY)
# No MasVnr seems to be cheaper

# In[81]:


df['Mas Vnr Type'].replace(to_replace = ['BrkFace', 'BrkCmn'], value = 'Brk', inplace = True)


# In[82]:


hue = 'Mas Vnr Type'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# In[83]:


new_dummy = pd.get_dummies(df["Mas Vnr Type"], prefix="MasVrn")
new_dummy = new_dummy.drop(columns=['MasVrn_None', 'MasVrn_CBlock'], axis = 1)
df = pd.concat([df, new_dummy], axis=1)

for n in new_dummy:
    NEW_DUMMY += [n]
df = df.drop(columns=['Mas Vnr Type'])

DROPPED_VARIABLE.append('Mas Vnr Type')
NOMINAL_VARIABLE.remove('Mas Vnr Type')


# ## Foundation (TO DUMMY)
# CBlock and BrkTil lower price,
# PConc higher price

# In[84]:


hue = 'Foundation'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# In[85]:


# make into dummy but drop slab, stone, wood cause less value
new_dummy = pd.get_dummies(df["Foundation"], prefix="Foundation")
new_dummy = new_dummy.drop(columns=['Foundation_Wood', 'Foundation_Slab', 'Foundation_Stone'], axis = 1)
df = pd.concat([df, new_dummy], axis=1)

for n in new_dummy:
    NEW_DUMMY += [n]
df = df.drop(columns=['Foundation'])

DROPPED_VARIABLE.append('Foundation')
NOMINAL_VARIABLE.remove('Foundation')


# ## Heating (DROP)
# mostly same and no pattern, can remove

# In[86]:


hue = 'Heating'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Heating'])

DROPPED_VARIABLE.append('Heating')
NOMINAL_VARIABLE.remove('Heating')


# ## Garage Type (DROP)
# No garage and detached tend to be cheaper price, this means it is the same as 'got garage'

# In[87]:


hue = 'Garage Type'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);
df = df.drop(columns=['Garage Type'])

DROPPED_VARIABLE.append('Garage Type')
NOMINAL_VARIABLE.remove('Garage Type')


# ## Sale Type (DROP but CREATE NEW)
# reduce variables since some variables quite related, 
# Reduce to 5 categories
# Warranty Deed (WD) - WD, CWD, VWD
# Just constructed - New
# Court Officer - COD
# Contract - Con, ConLW, ConLI, ConLD
# Other - Oth

# In[88]:


df['Sale Type'].replace(to_replace = ['WD ', 'CWD', 'VWD'], value = 'WD', inplace = True)
df['Sale Type'].replace(to_replace = ['Con', 'ConLw', 'ConLI', 'ConLD'], value = 'Contract', inplace = True)


# In[89]:


hue = 'Sale Type'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# New house seems to cost more than others, create a variable indicating the house is new

# In[90]:


df["Newly Constructed"] = df["Sale Type"].apply(lambda x: 1 if x == "New" else 0)
NEW_DUMMY.append('Newly Constructed')
df = df.drop(columns=['Sale Type'])

DROPPED_VARIABLE.append('Sale Type')
NOMINAL_VARIABLE.remove('Sale Type')


# ## Sale Condition (TO DUMMY)
# Partial, Normal and Abnormal got more values, 
# Partial seems to cost more
# Abnormal seems to cost less

# In[91]:


hue = 'Sale Condition'
fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x=x, y=y, hue=hue, style = hue, ax = ax);


# In[92]:


# make into dummy but drop slab, stone, wood cause less value
df["Sale Partial"] = df["Sale Condition"].apply(lambda x: 1 if x == "Partial" else 0)
df["Sale Abnormal"] = df["Sale Condition"].apply(lambda x: 1 if x == "Abnorml" else 0)

NEW_DUMMY += ['Sale Partial', 'Sale Abnormal']
df = df.drop(columns=['Sale Condition'])

DROPPED_VARIABLE.append('Sale Condition')
NOMINAL_VARIABLE.remove('Sale Condition')


# ## Neighborhood
# Obviously some neighborhood cost less while some cost more

# In[93]:


_, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x="Neighborhood", y="SalePrice", data=df, ax=ax)
ax.set_title("Neighborhood", fontsize=24)
ax.set_xlabel("Neighborhood", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_ylabel("House Price", fontsize=18);


# In[94]:


# too many columns if use dummy variables, use label encoder instead
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Neighborhood'] = labelencoder.fit_transform(df['Neighborhood'])


# # Outlier
# df with dummy
# df_findourlier no dummy

# In[95]:


from sklearn.ensemble import IsolationForest

# df_findourlier = df.drop(NEW_DUMMY, axis = 1) # no include dummy variables in outlier finding

contaminate = 0.005
iso = IsolationForest(contamination = contaminate)
iso.fit(df)
df["outlier"] = pd.Series(iso.predict(df))

# outlier will be label as 1
df["outlier"] = df["outlier"].apply(lambda x: 1 if x < 0 else 0)
print(df["outlier"].value_counts())

fig, ax = plt.subplots(figsize = (8,6))
sns.scatterplot(data=df, x="Gr Liv Area", y="SalePrice", hue="outlier", style = "outlier", ax = ax);


# In[96]:


df = df.drop(df[df['outlier'] == 1].index)
df = df.drop('outlier', axis = 1)
df = df.reset_index(drop=True)


# # Factor Analysis

# In[97]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler


# In[98]:


VARIABLE = CONTINUOUS_VARIABLE
VARIABLE.extend(DISCRETE_VARIABLE) # discrete variable should be scaled too
x_scale = df[VARIABLE]

TO_DROP = TARGET + VARIABLE
cat_variable = df.drop(TO_DROP, axis = 1)


# In[99]:


#scaler = StandardScaler()
scaler = QuantileTransformer()
x_scale = pd.DataFrame(scaler.fit_transform(x_scale.values), columns=x_scale.columns, index=x_scale.index)

x = pd.concat([x_scale, cat_variable], axis = 1)


# In[100]:


def univariant_selection(x, y, func):
    # k = 30, choose top 30 
    select_features = SelectKBest(score_func = func, k = 30)
    
    fit = select_features.fit(x, y)  # fit() is to run the score functions of x and y
    select_columns = pd.DataFrame(x.columns) # put column name in dataframe
    select_scores = pd.DataFrame(fit.scores_) # put score in dataframe
    features_scores = pd.concat([select_columns, select_scores],axis=1) # concatenate the columns
    features_scores.columns = ['Feature','Score']  # name of columns
    #print(features_scores.nlargest(30,'Score'))  # print out and show best features
    
    # Choose 30 top features and apply before doing prediction
    # top_features = features_scores.nlargest(30,'Score')
    # x = x[top_features["Feature"]]
    
    return features_scores.nlargest(30,'Score')


# In[101]:


x = pd.concat([x_scale, cat_variable], axis = 1)
# x = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# univariant selection
# f_regression works better with numerical data
score_df_num = univariant_selection(x_scale, y, f_regression)
# chi2 works better with categorical data
score_df_cat = univariant_selection(cat_variable, y, chi2)
# obtain both score and concat them
score_df = pd.concat([score_df_num, score_df_cat], axis = 0)


# In[102]:


# select the largest score from concat 
score_df.nlargest(30,'Score')
top_features = score_df.nlargest(30,'Score')
x = x[top_features["Feature"]]


# In[103]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[104]:


def prediction (data, target):
    for i, (train, test) in enumerate(KFold(3).split(data, target)):
        x_train, x_test = data.iloc[train].copy(), data.iloc[test].copy()
        y_train, y_test = target.iloc[train].copy(), target.iloc[test].copy()
    
        # Scale y value
        #y_scaler = StandardScaler()
        y_scaler = QuantileTransformer()
        #y_scaler = MinMaxScaler()
        # y_scaler = QuantileTransformer(output_distribution='uniform')
        y_train = y_train.values.reshape((-1, 1))
        y_test = y_test.values.reshape((-1, 1))
        y_train = y_scaler.fit_transform(y_train).ravel()
        y_test = y_scaler.transform(y_test).ravel()
        
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        name = f'Linear {i}'
        print(f'{name} rmse:', np.sqrt(mean_squared_error(y_test, pred)))
        
        reg = DecisionTreeRegressor()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        name = f'DecTree {i}'
        print(f'{name} rmse:', np.sqrt(mean_squared_error(y_test, pred)))
        
        reg = RandomForestRegressor()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        name = f'RandForest {i}'
        print(f'{name} rmse:', np.sqrt(mean_squared_error(y_test, pred)))
        
        reg = Ridge()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        name = f'Ridge {i}'
        print(f'{name} rmse:', np.sqrt(mean_squared_error(y_test, pred)))
        
        print()


# In[105]:


data = x
target = df['SalePrice']

prediction (data, target)

