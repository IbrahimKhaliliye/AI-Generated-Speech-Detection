# Audio and Image Classification Models
This is the work of Ali Saffarini and Ibrahim Khaliliye

This repository contains various implementations of models for audio and image classification tasks. The models include convolutional neural networks (CNNs) and other machine learning techniques. Each implementation is provided in separate files and notebooks for ease of understanding and experimentation.

## Repository Structure

Here is a brief description of each file and notebook in this repository:

- **Audio Classification:**
These classifications are reliant on the data of the frequencies of the audio as well as some other important features, like rms, chroma_stft, rolloff, etc.
  - `Audio_Classification_CNN_Notebook.ipynb`: Notebook for training a CNN model on audio data.
  - `Wav_MFCCS.ipynb`: Notebook for extracting Mel-Frequency Cepstral Coefficients (MFCCs) from WAV files for audio classification.
  - `AIModel_FineTunded.py`: Python script for fine-tuning an AI model.
  - `anothermodel.py`: Python script for another image classification model.
 - `AImodel.py`: Python script for an AI model for classification tasks.
- **Image Classification:**
These models were inspired by the idea of classifying the audio files based on the images of their waveforms.
  - `imagemodel.ipynb`: Notebook for a basic image classification model.
  - `largerimagemodel.ipynb`: Notebook for a more complex image classification model designed for larger datasets.
  - `fineTunedImage.py`: Python script for fine-tuning an image classification model.
 

- **Model Testing and Trials:**

  - `testingallmodels.ipynb`: Notebook for testing different models on the same dataset.
  - `testforotherdataset.ipynb`: Notebook for testing models on a different dataset.
  - `trial.ipynb`: Notebook for trial runs of various models.
  - `test.py`: Python script for testing models.
  - `test.ipynb`: Notebook for experimental models and tests by Ibrahim.

- **Data Processing and Fixes:**

  - `hardreaddata.ipynb`: Notebook focusing on data preprocessing and reading techniques for large datasets.
  - `fix.ipynb`: Notebook for addressing issues and fixes in existing models.

- **Miscellaneous:**
  - `testrforrest.py`: Python script for testing a random forest model for classification tasks.
  - `Software.py`: Python script for software-related functionality.
  - `settings.json`: Configuration file for terminal settings to disable OneDNN optimizations for TensorFlow.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Link to pretrained models: https://drive.google.com/drive/folders/1u_RbokM4mtJtv4jY8p5LBbef-oFxgDqO?usp=sharing
- Python 3.x
- Jupyter Notebook
- Required libraries: TensorFlow, Keras, scikit-learn, pandas, numpy, matplotlib, etc.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name
   pip install -r requirements.txt

   ```
