import os
import shutil
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def save_waveform_image(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        plt.figure(figsize=(14, 5))
        plt.plot(y)
        plt.title('Waveform')
        plt.savefig(save_path)
        plt.close('all')
    except:
        print('Failed to process:', audio_path)
        pass


def preprocess_audio_files(audio_dir, image_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for subdir, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav','.flac')):
                audio_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(audio_path, audio_dir)
                if file.endswith('.wav'):
                    save_path = os.path.join(image_dir, relative_path.replace('.wav', '.png'))
                if file.endswith('.flac'):
                    save_path = os.path.join(image_dir, relative_path.replace('.flac', '.png'))
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_waveform_image(audio_path, save_path)


def organize_images(image_dir, train_dir, test_dir, test_size=0.2):
    categories = ['real2000', 'fake2000']
    for category in categories:
        category_dir = os.path.join(image_dir, category)
        images = [os.path.join(category_dir, img) for img in os.listdir(category_dir) if img.endswith('.png')]
        train_images, test_images = train_test_split(images, test_size=test_size)
        
        for img_set, set_dir in zip([train_images, test_images], [train_dir, test_dir]):
            category_set_dir = os.path.join(set_dir, category)
            if not os.path.exists(category_set_dir):
                os.makedirs(category_set_dir)
            for img_path in img_set:
                shutil.copy(img_path, category_set_dir)


real_audio_dir = r'C:\Users\alisa\Downloads\2000realaudio'
fake_audio_dir = r'C:\Users\alisa\Downloads\2000fakeaudio'
real_image_dir = 'allimages/real2000'
fake_image_dir = 'allimages/fake2000'
image_dir = 'allimages'
train_dir = 'allimages/train'
test_dir = 'allimages/test'


preprocess_audio_files(real_audio_dir, real_image_dir)
preprocess_audio_files(fake_audio_dir, fake_image_dir)
organize_images(image_dir, train_dir, test_dir)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    fill_mode='reflect'
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = LearningRateScheduler(scheduler)


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[callback]
)


for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy:.2f}')

model.save('waveform-image-CNN-EfficientNet.h5')


def predict_audio(audio_path, model):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(14, 5))
    plt.plot(y)
    plt.title('Waveform')
    waveform_image_path = 'temp_waveform.png'
    plt.savefig(waveform_image_path)
    plt.close()

    img = load_img(waveform_image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction


test_audio_path = r'C:\Users\alisa\Downloads\ali.wav'
prediction = predict_audio(test_audio_path, model)
print(f'Prediction for {test_audio_path}: {prediction}')


y_true = test_generator.classes
y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
roc_auc = roc_auc_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred.round())
precision = precision_score(y_true, y_pred.round())
recall = recall_score(y_true, y_pred.round())
print(f"ROC-AUC: {roc_auc:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
