## Importing the following libraries
## Basing this off of the Feature Engineering Demo and establishing this as best practices.
## Will uncomment sections of the following script as needed.

# 6/04 Before importing the below libraries, follow the README.py instructions to download the libraries into the environment.
# 6/04 If proceeding in a sequential manner, MUST uncomment AND KEEP uncommented as VSC will not retain the uncommented code \
# if it is commented out again.

## Run the following expression '# %%' to create a new cell in VSC, and 
## then run the cell to execute the code within it.

# %%

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# above libraries have been loaded.
## Load the Data, Examine and Explore

working_df = pd.read_csv(r"C:\Users\alter\Desktop\Coursera_Desktop\Final Projects\Final Project_EDA for ML_Ames Housing\data\Ames_Housing_Data.tsv", sep='\t')
## load path is as follows:"C:\Users\alter\Desktop\Coursera_Desktop\Final Projects\Final Project_EDA for ML_Ames Housing\data\Ames_Housing_Data.tsv"

# working_df.info()

## Removal of any outliers
"""
working_df = df_variable = Ames_Housing_Data.tsv
data_copy = df_variable_2 = working_df.copy()
one_hot_encode_cols = df_variable_3 = object-coded columns
mask_float = working_df.dtypes == float
float_cols = working_df.columns[mask_float]
skew_limit = 0.75
skew_vals = working_df[float_cols]
field = "BsmtFin SF 1"
"""

# print(working_df.shape)

working_df = working_df.loc[working_df['Gr Liv Area'] <= 4000,:]
print("Number of rows in the working_df data:", working_df.shape[0])
print("Number of columns in the working_df data:", working_df.shape[1])
data_copy = working_df.copy() # keeping a copy of the original data. 'data_copy' will be preserved as a copy and NOT used; 'df' will be the working copy.

## Look at the feature heads and data:
print("data_copy shape:", data_copy.shape)

## Identify the features for One-Hot Encoding. 
## Will identify and convert any categorical variables to numerical dummies, and then identify and make any skewed variables symmetric. 
## Then will identify any skew variables. 

## First, get a Pd.series showing all the string-defined categorical features.
# one_hot_encode_cols = working_df.dtypes[working_df.dtypes == object]
# one_hot_encode_cols = one_hot_encode_cols.index.tolist()

print("One-hot encode columns: None, as there are no object-coded columns in the dataset.")
# print(working_df[one_hot_encode_cols].head())
print("Working DataFrame description:")
print(working_df.describe())

# There are no object-coded columns within the dataset, so there are no columns to one-hot encode.

## Second, One-hot encode the dummy variables !!!MOVING ON TO THE SKEW CHECKS, AS THERE ARE NO OBJECT-CODED COLUMNS TO ONE-HOT ENCODE.

# working_df = pd.get_dummies(working_df, columns=one_hot_encode_cols, drop_first=True)
# working_df.describe().T

## Third, perform the Skew variables check

mask_float = working_df.dtypes == float
float_cols = working_df.columns[mask_float]
print("Float columns in the dataset:", float_cols)
print("First 5 rows of float columns:")
print(mask_float.head())

## Identified 39 float columns, proceeding to check for skewness.

skew_limit = 0.75 # any floats above this will require log transform
skew_vals = working_df[float_cols].skew()

# Fourth, show the skewed columns
skew_cols = (skew_vals.sort_values(ascending=False).to_frame().rename(columns={0: 'Skew'}).query('abs(Skew) > {}'.format(skew_limit)))
print(" ")
print("Skew values for all float columns:")
print(skew_cols)
print("Skewed columns (skew > {}):".format(skew_limit))
print(" ")
print("Skewed columns info:")
print(skew_cols.info())


## There are 6 columns with skewness above the limit of 0.75, and these are: 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 
## 'Low Qual Fin SF', and 'Wood Deck SF'.

## Applying np.log1p visually via histograms.

# Choose a field
# field = "BsmtFin SF 1" # Select a field here. 

# Create two "subplots" and a "figure" using matplotlib
# fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))

# Create a histogram on the "ax_before" subplot
# working_df[field].hist(ax=ax_before)

## Showing significant right skew in the histogram for "BsmtFin SF 1", 
# which is expected given the skew value of 1.25.

# Apply a log transformation (numpy syntax) to this column
# working_df[field].apply(np.log1p).hist(ax=ax_after)

# Formatting of titles etc. for each subplot
# ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
# ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
# fig.suptitle('Field "{}"'.format(field));

## Performing the Skew transformation

# for col in skew_cols.index.values:
#     if col == "SalePrice":
#         continue
#     working_df[col] = working_df[col].apply(np.log1p)

# Analyze the shape now to see the potentially-useful features.
# working_df.shape

## Sort out the variables that need to be removed.

"""
working_df = df_variable = Ames_Housing_Data.tsv
data_copy = df_variable_2 = working_df.copy()
one_hot_encode_cols = df_variable_3 = object-coded columns
mask_float = working_df.dtypes == float
float_cols = working_df.columns[mask_float]
skew_limit = 0.75
skew_vals = working_df[float_cols]
field = "BsmtFin SF 1"
smaller_working_df = df_variable_4 = working_df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Gr Liv Area', 'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 'Garage Cars', 'SalePrice']]
"""
# working_df.isnull().sum().sort_values()

## Go to a smaller dataset

smaller_working_df = working_df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Gr Liv Area', 'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 'Garage Cars', 'SalePrice']]

# Describe the smaller dataset. 

smaller_working_df.describe().T

print(smaller_working_df.info())

## Fill in any missing values with 0

smaller_working_df = smaller_working_df.fillna(0)

# Check the result

print("Smaller working DataFrame with no missing values description:")
print(smaller_working_df.info())

## Work out pair plots to see which feature-target relationships have noticeable distributions.

sns.pairplot(smaller_working_df, plot_kws=dict(alpha=.1, edgecolor='none'))

# Determine which of the features has noticeable distributions that could use further data exploration.




