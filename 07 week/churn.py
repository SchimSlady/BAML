import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_excel('e-commerce-dataset.xlsx', sheet_name='E_Comm')
df.to_csv('e-commerce_churn.csv', index=False)
df = pd.read_csv('e-commerce_churn.csv')

##################################################################
######################## Data preparation ########################
##################################################################

print("\n---------------------------------------\n\
---------- Data Preparation -----------\n\
----------------------------------------\n")

print(df.info())
df = df.dropna()

df['Churn'] = df['Churn'].astype('category')

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

##################################################################
########################       a.)        ########################
##################################################################
print("\n---------------------------------------\n\
-------------- Part a.) -----------------\n\
----------------------------------------\n")

# Train a random forest model

# Calculate accuracy

##################################################################
########################       c.)        ########################
##################################################################
print("\n---------------------------------------\n\
-------------- Part c.) -----------------\n\
----------------------------------------\n")

# Train-test split

# View proportions of Churn

# Define Model

# Cross-validation

# Train the final model

# Variable Importance Plot

# Apply on test set

# Confusion Matrix

# Precision, accuracy, recall
