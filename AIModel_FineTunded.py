import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to extract audio features
def extract_features(wav_file, sr=22050):
    y, sr = librosa.load(wav_file, sr=sr)  # Load the audio file
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)  # Chroma STFT
    rms = librosa.feature.rms(y=y)  # Root Mean Square (RMS) Energy
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Spectral Centroid
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # Spectral Bandwidth
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # Spectral Rolloff
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)  # Zero Crossing Rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Mel-frequency cepstral coefficients (MFCCs)

    # Aggregate features by computing the mean of each feature
    features = [np.mean(chroma_stft), np.mean(rms), np.mean(spectral_centroid),
                np.mean(spectral_bandwidth), np.mean(rolloff), np.mean(zero_crossing_rate)]
    for mfcc in mfccs:
        features.append(np.mean(mfcc))

    return features

# Function to augment audio data
def augment_audio(y, sr):
    y_noise = y + 0.005 * np.random.randn(len(y))  # Add noise
    y_stretch = librosa.effects.time_stretch(y, rate=np.random.uniform(0.75, 1.25))  # Time stretching
    y_pitch = librosa.effects.pitch_shift(y, sr, n_steps=np.random.randint(0,3))  # Pitch shifting
    return [y, y_noise, y_stretch, y_pitch]

# Load dataset
dataset = r"C:\Users\alisa\Downloads\DATASET-balanced.csv"

data = pd.read_csv(dataset)
data['LABEL'] = data['LABEL'].replace({'FAKE': '0', 'REAL': 1})  # Replace label strings with numerical values

# Extract features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training set and transform it
X_test = scaler.transform(X_test)  # Transform the test set using the same scaler

# Reshape X to fit the model's expected input shape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_train = to_categorical(y_train)  # Convert labels to one-hot encoded format
y_test = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))  # First Conv1D layer
model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
model.add(Dropout(0.3))  # Dropout layer to prevent overfitting
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))  # Second Conv1D layer with more filters
model.add(MaxPooling1D(pool_size=2))  # Another max pooling layer
model.add(Dropout(0.3))  # Dropout layer
model.add(Flatten())  # Flatten layer to convert 2D to 1D
model.add(Dense(units=128, activation='relu'))  # Dense layer with 128 units
model.add(Dropout(0.5))  # Dropout layer
model.add(Dense(units=2, activation='softmax'))  # Output layer with softmax activation

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Save the trained model
model.save('detectionmodel.keras')

# Evaluate the model using confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(confusion_matrix(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes))
