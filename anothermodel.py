import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from xgboost import XGBClassifier
import xgboost as xgb
import pickle

dataset = r"C:\Users\alisa\Downloads\DATASET-balanced.csv"

data = pd.read_csv(dataset)
data['LABEL'] = data['LABEL'].replace({'FAKE': 0, 'REAL': 1})  # Replace label strings with numerical values

# Extract features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training set and transform it
X_test = scaler.transform(X_test)  # Transform the test set using the same scaler

from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1]
}

# Create an instance of XGBRegressor
xgb_model = xgb.XGBClassifier()

# Create the grid search object
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Get the best combination of hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_xgb_model = xgb.XGBClassifier(**best_params)
best_xgb_model.fit(X_train, y_train, early_stopping_rounds = 10, eval_set=[(X_test,y_test)])

# Make predictions on the test data
y_pred = best_xgb_model.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score


    # Predict labels for test data
y_pred = best_xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

model_file = "xgb_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(best_xgb_model, f)



