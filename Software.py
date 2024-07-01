import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from tensorflow.keras.models import load_model




# Load the model
### Quick Software file to test it out locally
model = load_model(r'C:\Users\alisa\OneDrive\Desktop\GenGuard\AI-Generated-Speech-Detection\bestmodel.keras')
class AppDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Audio Detection')
        self.setGeometry(100, 100, 300, 200)
        
        layout = QVBoxLayout()

        self.label = QLabel('Listening...')
        layout.addWidget(self.label)

        self.button = QPushButton('Start Detection')
        self.button.clicked.connect(self.start_detection)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        audio_data = np.array(indata)
        # Preprocess audio_data as required by your model
        # For example, reshaping or normalizing
        prediction = model.predict(audio_data)
        # Update the label based on the prediction
        if prediction[0] > 0.5:
            self.label.setText('AI-generated audio detected')
        else:
            self.label.setText('Human audio detected')

    def start_detection(self):
        # Start audio capture and model prediction here
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()

app = QApplication([])
window = AppDemo()
window.show()
sys.exit(app.exec_())
