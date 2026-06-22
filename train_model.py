import os
import numpy as np
from feature_extraction import extract_features
dataset_path = "dataset"
x = []
y = []
emotion_map ={
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
    }
for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)
    if os.path.isdir(actor_path):  
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                emotion_code = file.split("-")[2]
                emotion = emotion_map[emotion_code]
                features = extract_features(file_path)
                x.append(features)
                y.append(emotion)
x = np.array(x)
y = np.array(y)
print("Feature extracted:",len(x))
print("Labels extracted:",len(y))
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Emotion classes:",)
print(encoder.classes_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(40,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(8, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)  
model.save("emotion_model.h5") 
print("Model saved successfully!")