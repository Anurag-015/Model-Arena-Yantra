import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from glob import glob
import random

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 4  # Number of videos per batch
FRAMES_PER_VIDEO = 15
EPOCHS = 5
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(SEED)

# Check GPU Availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("⚠️ No GPU detected. Running on CPU. (Note: TensorFlow >2.10 on Windows Native does not support GPU. Use WSL2 or older Python/TF versions if GPU is mandatory)")

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    """
    Extracts a fixed number of frames from a video, uniformly sampled.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Return zeros if video cannot be opened
        return np.zeros((num_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video has fewer frames than required, we'll take all and pad or duplicate
    # If video has 0 frames (corrupt), return zeros
    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    if total_frames < num_frames:
        indices = np.arange(total_frames)
        # Pad by repeating the last frame or cycling? Let's cycle.
        indices = np.pad(indices, (0, num_frames - total_frames), mode='wrap')
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames = []
    for i in range(num_frames):
        # Set frame position
        # Optimization: if indices are sorted, we can just read sequentially and skip
        # But setting pos is safer for random access logic
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[i])
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            # If read fails, append a black frame
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
            
    cap.release()
    return np.array(frames)

class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels=None, batch_size=BATCH_SIZE, 
                 frames_per_video=FRAMES_PER_VIDEO, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.frames_per_video = frames_per_video
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.video_paths[k] for k in indexes]
        
        X = []
        y = []
        
        for i, path in enumerate(batch_paths):
            frames = extract_frames(path, self.frames_per_video)
            X.append(frames)
            if self.labels is not None:
                # Label for each frame is the label of the video
                label = self.labels[indexes[i]]
                y.extend([label] * self.frames_per_video)
        
        X = np.concatenate(X, axis=0) # (Batch * Frames, H, W, C)
        
        if self.labels is not None:
            y = np.array(y)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Finding data...")
    # Assume current directory structure
    real_videos = glob("train/real/*.mp4")
    fake_videos = glob("train/fake/*.mp4")
    
    print(f"Found {len(real_videos)} REAL videos")
    print(f"Found {len(fake_videos)} FAKE videos")
    
    # Create labels
    # REAL = 0, FAKE = 1 (Since prompt says: FAKE if > 0.5 else REAL)
    # Wait, usually Deepfake detection: Fake=1, Real=0.
    # Prompt: "Assign labels using threshold 0.5 (FAKE if >0.5 else REAL)"
    # This confirms FAKE is the positive class (1).
    
    all_paths = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
    )
    
    print(f"Training on {len(train_paths)} videos, Validating on {len(val_paths)} videos")
    
    # Generators
    train_gen = VideoDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = VideoDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build and Train Model
    print("Building model...")
    model = build_model()
    
    print("Training head...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    
    # Fine-tuning
    print("Fine-tuning...")
    # Unfreeze the base model (which is layer 1)
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=3, validation_data=val_gen)
    
    # Inference
    print("Starting inference on test videos...")
    test_videos = glob("test/*.mp4")
    # Sort to match order? No, glob order is arbitrary but we need filenames.
    
    submission_data = []
    
    # Process test videos one by one (or in small batches) to avoid memory issues
    # Since we need video-level prediction, easier to do one by one
    
    for i, video_path in enumerate(test_videos):
        if i % 10 == 0:
            print(f"Processing {i}/{len(test_videos)}")
            
        filename = os.path.basename(video_path)
        
        # Extract frames
        frames = extract_frames(video_path, num_frames=FRAMES_PER_VIDEO) # (F, 224, 224, 3)
        
        # Predict
        if frames.shape[0] > 0:
            preds = model.predict(frames, verbose=0) # (F, 1)
            # Average probability
            video_prob = np.mean(preds)
        else:
            video_prob = 0.5 # Default if read fails
            
        # Assign label
        label = "FAKE" if video_prob > 0.5 else "REAL"
        
        submission_data.append({
            "filename": filename,
            "label": label,
            "probability": float(video_prob)
        })
        
    # Generate CSV
    df = pd.DataFrame(submission_data)
    # Ensure column order
    df = df[['filename', 'label', 'probability']]
    
    # Verify strict 0-1 range (sigmoid does this, but floating point might be weird)
    df['probability'] = df['probability'].clip(0.0, 1.0)
    
    output_file = "submission.csv"
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print(df.head())

if __name__ == "__main__":
    main()
