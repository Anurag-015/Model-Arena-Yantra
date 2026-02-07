import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Deepfake Detector - Single File Solution")
print("Structure: train/real/, train/fake/, test/")

def extract_frames(video_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 5 == 0:  # Sample every 5th frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        frame_count += 1
    cap.release()
    if len(frames) == 0: return None
    return np.mean(frames, axis=0) / 255.0  # Average frame as image

def load_dataset(data_dir, label=None):
    X, ids = [], []
    subdirs = ['fake', 'real'] if label is not None else ['']
    if label is None: subdirs = ['']  # For test
    for subdir in subdirs:
        subpath = os.path.join(data_dir, subdir) if subdir else data_dir
        if not os.path.exists(subpath): continue
        for video_file in tqdm(os.listdir(subpath), desc=f"Loading {subdir or 'test'}"):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(subpath, video_file)
                frame = extract_frames(video_path)
                if frame is not None:
                    X.append(frame)
                    ids.append(video_file)
        if label is not None:
            labels = [label] * len(X)
            return np.array(X), np.array(labels), ids
    return np.array(X), ids

print("\n1. Loading training data...")
train_X_fake, _ = load_dataset(os.path.join('train', 'fake'))
train_y_fake = np.zeros(len(train_X_fake))
train_X_real, _ = load_dataset(os.path.join('train', 'real'))
train_y_real = np.ones(len(train_X_real))
train_X = np.concatenate([train_X_fake, train_X_real])
train_y = np.concatenate([train_y_fake, train_y_real])

print(f"Train samples: {len(train_X)}")

train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

print("\n2. Building EfficientNet model...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')

print("3. Training model...")
history = model.fit(train_X, train_y, epochs=15, batch_size=16,
                    validation_data=(val_X, val_y), callbacks=[early_stop,
                    tf.keras.callbacks.ReduceLROnPlateau(patience=3)],
                    verbose=1)

val_acc = max(history.history['val_accuracy'])
print(f"Best validation accuracy: {val_acc:.3f}")

print("\n4. Loading test data...")
test_X, test_ids = load_dataset('test', label=None)
print(f"Test samples: {len(test_X)}")

print("5. Making predictions...")
predictions = model.predict(test_X, verbose=0).flatten()

df = pd.DataFrame({
    'filename': test_ids,
    'fake_probability': predictions,
    'is_real': (predictions < 0.5).astype(int)  # 1=fake if prob>0.5, adjust threshold
})
df = df.sort_values('filename')
df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to predictions.csv")
print(df.head())
print(f"\nFake ratio in test: {(predictions > 0.5).mean():.1%}")
print("Done! Model ready for deployment.")
