import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")

emotion_labels = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

audio_path = input("Enter audio file path: ")

audio, sample_rate = librosa.load(audio_path, sr=None)

mfccs = np.mean(
    librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
    axis=0
)

mfccs = np.expand_dims(mfccs, axis=0)

prediction = model.predict(mfccs)

emotion = emotion_labels[np.argmax(prediction)]

print("\nPredicted Emotion:", emotion)