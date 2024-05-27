import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
dataset_path = '/Users/admin/Downloads/archive/KAGGLE/DATASET-balanced.csv'  # Replace with the correct path
df = pd.read_csv(dataset_path)

# Separate features and labels
y = df['LABEL']
X = df.drop(columns=['LABEL'])

# Binarise the Labels for Binary Classification
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare the model and K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X_scaled, y)
best_params = grid_search.best_params_

# Train with the best parameters
model = RandomForestClassifier(**best_params, random_state=42)

acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

# Train Each Fold
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))

    print(f"Fold completed in {train_time:.2f} seconds")

# Calculate the Mean Results and Standard Deviation
print("Mean results and (std.):\n")
print("Accuracy: " + str(round(np.mean(acc_scores) * 100, 3)) + "% (" + str(round(np.std(acc_scores) * 100, 3)) + "%)")
print("Precision: " + str(round(np.mean(precision_scores) * 100, 3)) + "% (" + str(round(np.std(precision_scores) * 100, 3)) + "%)")
print("Recall: " + str(round(np.mean(recall_scores) * 100, 3)) + "% (" + str(round(np.std(recall_scores) * 100, 3)) + "%)")
print("F1-Score: " + str(round(np.mean(f1_scores) * 100, 3)) + "% (" + str(round(np.std(f1_scores) * 100, 3)) + "%)")
print("ROC AUC Score: " + str(round(np.mean(roc_auc_scores) * 100, 3)) + "% (" + str(round(np.std(roc_auc_scores) * 100, 3)) + "%)")
