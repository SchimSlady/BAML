import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# File reading/writing
df1 = pd.read_csv("train.csv") 
df2 = pd.read_csv('test.csv')
df = pd.concat([df1, df2])
df2

#Drop unnecessary columns
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)

#Fill NAs for cabin because otherwise too much data would be lost
df['Cabin'] = df['Cabin'].fillna('NoInfo')
df['Cabin'] = df['Cabin'].str[0]

#Replace Gender with 1 and 0
df ['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df.info()

value_count = df['Cabin'].value_counts()
value_count

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_cols, drop_first=True)

training_data = df.dropna(subset=['Survived']).copy()
private_data = df[df['Survived'].isna()].copy()
training_data.info()

#Drop NAs for Age and Embarked because those are only few values
training_data = training_data.dropna(subset=["Age"])
training_data.info()

# Define Model
train_model = RandomForestClassifier(n_estimators=500, max_features=3, random_state=0)

# Train-test split
train_df, test_df = train_test_split(training_data, test_size=0.20, stratify=training_data['Survived'], random_state=2023+2024)

# Train a random forest model
X_train = train_df.drop(columns=['Survived'])
#Alternativ: X = train_df[features] mit features = ['feature1', 'feature2', etc.]
y_train = train_df['Survived']

# Test Set
X_test = test_df.drop(columns=['Survived'])
#Alternativ: X = test_df[features] mit features = ['feature1', 'feature2', etc.]
y_test = test_df['Survived']

# Cross-validation
cv_fits_accuracy = cross_val_score(train_model, X_train, y_train, cv=4, scoring='accuracy')
cv_fits_precision = cross_val_score(train_model, X_train, y_train, cv=4, scoring='precision')
cv_fits_recall = cross_val_score(train_model, X_train, y_train, cv=4, scoring='recall')
cv_fits_BAC = cross_val_score(train_model, X_train, y_train, cv=4, scoring='balanced_accuracy')


print("\nCV-Accuracy:", np.mean(cv_fits_accuracy))
print("CV-Precision:", np.mean(cv_fits_precision))
print("CV-Recall:", np.mean(cv_fits_recall))
print("CV-BAC:", np.mean(cv_fits_BAC))

# Train the final model
train_model.fit(X_train, y_train) # oder variablen nutzen x_train und y_train

# Variable Importance Plot
importance_values = train_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance_values})
imp_plot = importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
imp_plot.plot()
plt.show()

# Apply on test set (private data)
test_predictions = train_model.predict(private_data.drop(columns=['Survived'])).astype(int)
print(test_predictions)

# Write the test predictions into the private data df into the column 'label'
private_data ['Survived'] = test_predictions

# Keep only the 'ID' and 'label' columns
result = private_data[['PassengerId', 'Survived']]

# Save the result to a CSV file
result.to_csv('prediction2.csv', index=False)