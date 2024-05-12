import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings

import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf

def extract_features1(wav_file, sr=22050):
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=sr)

    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Aggregate features
    features = [np.mean(chroma_stft), np.mean(rms), np.mean(spectral_centroid),
                np.mean(spectral_bandwidth), np.mean(rolloff), np.mean(zero_crossing_rate)]
    for mfcc in mfccs:
        features.append(np.mean(mfcc))

    return features

dataset = r"C:\Users\alisa\Downloads\DATASET-balanced.csv"


data = pd.read_csv(dataset)
data['LABEL'] = data['LABEL'].replace({'FAKE': '0', 'REAL': 1})



def split_dataset(csv_file):
    # Read the CSV file


    # X contains all columns except the last one (features)
    X = data.iloc[:, :-1]

    # y contains the last column (target variable)
    y = data.iloc[:, -1]

    return X, y

# Example usage:
X, y = split_dataset(data)

import numpy as np
import librosa
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.utils import to_categorical


model =Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(26, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the dataset
X=X.values
y=y.values
# Preprocess the dataset
X = X.reshape((X.shape[0], X.shape[1], 1))
y = to_categorical(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))


from keras.models import load_model

model.save('detectionmodel.keras')

