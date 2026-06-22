# Speech Emotion Recognition

A Machine Learning project that detects emotions from speech audio using Python, Librosa and TensorFlow.

## Features
- Audio preprocessing
- MFCC feature extraction
- Emotion prediction using trained model
- Supports WAV audio files

## Technologies Used
- Python
- TensorFlow
- Librosa
- NumPy

## Project Structure

├── dataset_loader.py
├── feature_extraction.py
├── train_model.py
├── predict.py
├── emotion_model.h5
├── requirements.txt
└── test_audio.wav

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run prediction

python predict.py

3. Enter audio file path

Example:
test_audio.wav

## Output

Predicted Emotion: calm
