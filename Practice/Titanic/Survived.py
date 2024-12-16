import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

test = pd.read_csv('test.csv')

#Replace Gender with 1 and 0
test ['Sex'] = test['Sex'].map({'male': 1, 'female': 0})

#Drop unnecessary columns
test = test.drop('Name', axis=1)
test = test.drop('Ticket', axis=1)

#Ersetze die Cabin Nummer nur mit dem zugehörigen Deck 
test['Cabin'] = test['Cabin'].str[0]

# Identify non-numeric columns
non_numeric_cols = test.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
test = pd.get_dummies(test, columns=non_numeric_cols, drop_first=True)

test.info()

#read Training Data set
training = pd.read_csv('train.csv')

#Drop NAs for Age and Embarked because those are only few values
training = training.dropna(subset=["Age", 'Embarked'])

#Fill NAs for cabin because otherwise too much data would be lost
training['Cabin'] = training['Cabin'].fillna('NoInfo')

#Replace Gender with 1 and 0
training ['Sex'] = training['Sex'].map({'male': 1, 'female': 0})

#Transform Objects into categories
training['Survived'] = training['Survived'].astype('category')

#Ersetze die Cabin Nummer nur mit dem zugehörigen Deck 
training['Cabin'] = training['Cabin'].str[0]

#Drop unnecessary columns
training = training.drop('Name', axis=1)
training = training.drop('Ticket', axis=1)

#Info about training set
training.info()

# Identify non-numeric columns
non_numeric_cols = training.select_dtypes(include=['object']).columns

# Apply one-hot encoding to non-numeric columns
training = pd.get_dummies(training, columns=non_numeric_cols, drop_first=True)

training = training.drop('Cabin_T', axis=1)
training = training.drop('Cabin_N', axis=1)
training.info()

#Check for imbalance
anzahl = training['Survived'].value_counts(normalize=False)
print(anzahl)

# Define Model
train_model = RandomForestClassifier(n_estimators=500, max_features=3, random_state=0)

# Train a random forest model
X = training.drop(columns=['Survived'])
y = training['Survived']

# Cross-validation
cv_fits_accuracy = cross_val_score(train_model, X, y, cv=4, scoring='accuracy')
cv_fits_precision = cross_val_score(train_model, X, y, cv=4, scoring='precision')
cv_fits_recall = cross_val_score(train_model, X, y, cv=4, scoring='recall')

print("\nCV-Accuracy:", np.mean(cv_fits_accuracy))
print("CV-Precision:", np.mean(cv_fits_precision))
print("CV-Recall:", np.mean(cv_fits_recall))

# Train the final model
train_model.fit(training.drop(columns=['Survived']), training['Survived'])

# Apply on test set
test_predictions = train_model.predict(test)
print(test_predictions)

# Erstelle ein DataFrame mit den Vorhersagen und Wahrscheinlichkeiten
test_predictions_df = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # Identifiziere Passagiere  
    'Survived': test_predictions,  # Modellvorhersagen
})
test_predictions_df.to_csv('prediction_file.csv', index=False)