import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os



target_sr = 22050


def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y = librosa.effects.resample(y, sr, target_sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=128)
        mel_spect_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spect_db
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def augment_audio(y, sr):
    augmented_samples = []
    y_stretched = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    augmented_samples.append(y_stretched)
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=np.random.randint(-5, 5))
    augmented_samples.append(y_shifted)
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise
    augmented_samples.append(y_noisy)
    return augmented_samples


def segment_mel_spectrogram(mel_spect, segment_length=128, hop_length=64):
    segments = []
    num_segments = (mel_spect.shape[1] - segment_length) // hop_length + 1
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        segment = mel_spect[:, start:end]
        segments.append(segment)
    return segments


real_dir = r'C:\Users\alisa\Downloads\REAL2'
fake_dir = r'C:\Users\alisa\Downloads\FAKE\target generated'

audio = []
labels = []

# Label: 0 for real, 1 for fake
for filename in os.listdir(real_dir):
    filepath = os.path.join(real_dir, filename)
    if os.path.isfile(filepath):
        audio.append(filepath)
        labels.append(0)

for filename in os.listdir(fake_dir):
    filepath = os.path.join(fake_dir, filename)
    if os.path.isfile(filepath):
        audio.append(filepath)
        labels.append(1)

X = []
y = []
segment_length = 128
hop_length = 64

for file_path, label in zip(audio, labels):
    try:
        y_audio, sr = librosa.load(file_path, sr=None)
        augmented_samples = augment_audio(y_audio, sr)
        for sample in augmented_samples:
            mel_spect = librosa.feature.melspectrogram(sample, sr=sr, n_mels=128)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
            segments = segment_mel_spectrogram(mel_spect, segment_length=segment_length, hop_length=hop_length)
            X.extend(segments)
            y.extend([label] * len(segments))
    except:
        pass

X = np.array(X)
X = X[..., np.newaxis]  
y = np.array(y)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(segment_length, segment_length, 1), padding='same', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)


history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])


model.save('spectromodel1.keras')


audio_train = audio[:45000]+audio[57151:142000]
label_train = labels[:45000]+labels[57151:142000]

audio_validate = audio[45001:57150]+audio[142001:]
label_validate = labels[45001:57150]+labels[142001:]

A = []
b = []
for file_path, label in zip(audio_validate, label_validate):
    try:
        mel_spect = preprocess_audio(file_path)
        segments = segment_mel_spectrogram(mel_spect, segment_length=segment_length, hop_length=hop_length)
        A.extend(segments)
        b.extend([label] * len(segments))
    except:
        pass

A = np.array(A)
A = A[..., np.newaxis]
b = np.array(b)
predictions = model.predict(A)
predicted_labels = (predictions > 0.5).astype(int).flatten()


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(b, predicted_labels)
print(f'Validation Accuracy: {accuracy:.4f}')
print(classification_report(b, predicted_labels))


def predict_audio(file_path, model, segment_length, hop_length):
    mel_spect = preprocess_audio(file_path)
    segments = segment_mel_spectrogram(mel_spect, segment_length=segment_length, hop_length=hop_length)
    segments = np.array(segments)
    segments = segments[..., np.newaxis]
    predictions = model.predict(segments)
    final_prediction = (np.mean(predictions) > 0.5).astype(int)
    return final_prediction

test = r'C:\Users\alisa\Downloads\ElevenLabs_2024-02-21T05_28_34_Me_ivc_s50_sb75_se0_b_m2.wav'
final_label = predict_audio(test, model, segment_length, hop_length)
print(f'Predicted label: {final_label}')