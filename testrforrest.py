import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV
import warnings
import joblib
warnings.filterwarnings('ignore')

# Load the dataset

dataset_path = r'C:\Users\alisa\Downloads\DATASET-balanced.csv'  # Replace with the correct path
df = pd.read_csv(dataset_path)
df['LABEL'] = df['LABEL'].replace({'FAKE': 0, 'REAL': 1})  # Replace label strings with numerical values



#########
df_train = df[:-2000]
#########
df_validate = df[-2000:]

# Load the saved model
model_path = 'rforrest.joblib'
model = joblib.load(model_path)



# Separate features and labels (if you have labels for evaluation)
y_test = df_validate['LABEL']  # Assuming the test dataset has a 'LABEL' column
X_test = df_validate.drop(columns=['LABEL'])

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')
