Harshit, [07-02-2026 03:02 PM]
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
FRAMES_PER_VIDEO = 25
BATCH_SIZE = 16
EPOCHS = 12

TRAIN_DIR = "train"
TEST_DIR = "test"
MODEL_PATH = "deepfake_face_model.h5"
CSV_PATH = "results.csv"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

detector = MTCNN()

def extract_face(frame):
    try:
        # OpenCV -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]["box"]

        if w < 20 or h < 20:
            return None

        x, y = max(0, x), max(0, y)
        face = rgb[y:y+h, x:x+w]

        if face.size == 0:
            return None

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        return face

    except Exception:
        return None

def get_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.array([])

    step = max(total_frames // FRAMES_PER_VIDEO, 1)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            face = extract_face(frame)
            if face is not None:
                frames.append(face / 255.0)

            if len(frames) == FRAMES_PER_VIDEO:
                break

        count += 1

    cap.release()
    return np.array(frames)

def load_training_data():
    X, y = [], []

    for label, cls in enumerate(["real", "fake"]):
        folder = os.path.join(TRAIN_DIR, cls)

        for video in tqdm(os.listdir(folder), desc=f"Processing {cls}"):
            video_path = os.path.join(folder, video)
            faces = get_faces_from_video(video_path)

            if len(faces) == FRAMES_PER_VIDEO:
                X.append(faces)
                y.append(label)

    if len(X) == 0:
        raise RuntimeError("f**")

    X = np.array(X)
    y = np.array(y)

    # Convert video-level → frame-level
    X = X.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
    y = np.repeat(y, FRAMES_PER_VIDEO)

    return X, y

def build_model():
    base = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(base.input, output)
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

print("\nLoading training data...")
X_train, y_train = load_training_data()

print(f"Training samples: {X_train.shape}")
model = build_model()
model.summary()

print("\nTraining model...")
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    shuffle=True
)

model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
3
def predict_video(video_path, model):
    faces = get_faces_from_video(video_path)

    if len(faces) == 0:
        return "UNKNOWN", 0.0

    preds = model.predict(faces, verbose=0)
    score = float(np.mean(preds))
    label = "FAKE" if score > 0.5 else "REAL"
    return label, score

print("\nRunning predictions on test videos...")
results = []

for video in tqdm(os.listdir(TEST_DIR)):
    video_path = os.path.join(TEST_DIR, video)
    label, score = predict_video(video_path, model)
    results.append([video, label, round(score, 4)])

df = pd.DataFrame(results, columns=["video_name", "prediction", "confidence"]

Harshit, [07-02-2026 03:02 PM]
)
df.to_csv(CSV_PATH, index=False)

print(f"\ndone{CSV_PATH}")
