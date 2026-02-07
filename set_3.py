# Anurag Gupta
import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image

# Configuration
CONFIG = {
    'input_size': (224, 224),
    'batch_size': 8,  # Small batch size for limited GPU memory
    'epochs': 10,
    'frame_sample_rate': 12,
    'train_dir': 'train',
    'test_dir': 'test',
    'processed_train_dir': 'processed_dataset/train',
    'processed_test_dir': 'processed_dataset/test',
    'models_dir': 'models',
    'submission_file': 'submission.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def setup_environment():
    print(f"[INFO] Using device: {CONFIG['device']}")
    if CONFIG['device'] == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print("[INFO] Mixed precision enabled (via torch.cuda.amp).")
    else:
        print("[WARNING] No GPU detected. Training will be slow.")

    os.makedirs(CONFIG['processed_train_dir'], exist_ok=True)
    os.makedirs(CONFIG['processed_test_dir'], exist_ok=True)
    os.makedirs(CONFIG['models_dir'], exist_ok=True)

# --- Preprocessing ---

def extract_faces_from_video(video_path, mtcnn, save_dir=None, required_size=(224, 224), sample_rate=12):
    """
    Extracts faces from video using MTCNN.
    If save_dir is provided, saves face images to disk.
    Returns list of face images (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    faces_list = []
    frame_count = 0
    filename_base = os.path.basename(video_path).split('.')[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detect faces
            # mtcnn.detect returns boxes, probs. We use forward() to get cropped tensors or save manually
            # Using detect() allows us to crop and resize manually to ensure consistency
            try:
                boxes, _ = mtcnn.detect(frame_pil)
            except Exception:
                boxes = None

            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    # Clamp coordinates
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_rgb.shape[1], x2)
                    y2 = min(frame_rgb.shape[0], y2)
                    
                    if x2 - x1 < 20 or y2 - y1 < 20: continue # Skip tiny faces

                    face = frame_rgb[y1:y2, x1:x2]
                    try:
                        face = cv2.resize(face, required_size)
                        faces_list.append(face)
                        
                        if save_dir:
                            save_path = os.path.join(save_dir, f"{filename_base}_frame{frame_count}_face{i}.jpg")
                            # Save as BGR for opencv
                            cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    except Exception:
                        continue
        frame_count += 1
    
    cap.release()
    return faces_list

def prepare_dataset():
    print("[INFO] Preparing dataset...")
    
    # Initialize MTCNN on device
    # keep_all=True to detect all faces, select_largest=False
    mtcnn = MTCNN(keep_all=True, device=CONFIG['device'], margin=0, min_face_size=40)
    
    # Process Train Data
    for label in ['REAL', 'FAKE']:
        src_dir = os.path.join(CONFIG['train_dir'], label)
        dst_dir = os.path.join(CONFIG['processed_train_dir'], label)
        os.makedirs(dst_dir, exist_ok=True)
        
        if not os.path.exists(src_dir):
            print(f"[WARNING] Directory {src_dir} not found. Skipping.")
            continue
            
        videos = glob.glob(os.path.join(src_dir, "*.mp4"))
        print(f"[INFO] Processing {len(videos)} videos in {label}...")
        
        for video_path in tqdm(videos):
            extract_faces_from_video(video_path, mtcnn, save_dir=dst_dir, 
                                     required_size=CONFIG['input_size'], 
                                     sample_rate=CONFIG['frame_sample_rate'])

# --- Dataset & DataLoader ---

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(['REAL', 'FAKE']): # 0: REAL, 1: FAKE
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label) # 0 for REAL, 1 for FAKE (target: is_fake)
                        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(CONFIG['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# --- Model ---

class DeepfakeNet(nn.Module):
    def __init__(self, model_name='B0'):
        super(DeepfakeNet, self).__init__()
        if model_name == 'B0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
        elif model_name == 'B3':
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features, 1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(model_name, train_loader, epochs=10):
    print(f"\n[INFO] Training {model_name}...")
    model = DeepfakeNet(model_name).to(CONFIG['device'])
    
    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    criterion = nn.BCEWithLogitsLoss() # Combine sigmoid + BCELoss
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    patience = 3
    counter = 0
    
    # First phase: Train head only
    print("Phase 1: Training Head...")
    for epoch in range(3): # Train head for 3 epochs
        model.train()
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/3"):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device']).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        print(f"Loss: {epoch_loss/len(train_loader):.4f}")

    # Second phase: Fine-tune
    print("Phase 2: Fine-tuning...")
    for param in model.backbone.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device']).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['models_dir'], f'best_{model_name}.pth'))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    return model

# --- Inference ---

def inference_pipeline():
    print("\n[INFO] Starting Inference...")
    
    # Load Models
    model_b0 = DeepfakeNet('B0').to(CONFIG['device'])
    model_b3 = DeepfakeNet('B3').to(CONFIG['device'])
    
    try:
        model_b0.load_state_dict(torch.load(os.path.join(CONFIG['models_dir'], 'best_B0.pth')))
        model_b3.load_state_dict(torch.load(os.path.join(CONFIG['models_dir'], 'best_B3.pth')))
    except FileNotFoundError:
        print("[WARNING] Trained models not found. Skipping inference.")
        return

    model_b0.eval()
    model_b3.eval()
    
    test_videos = glob.glob(os.path.join(CONFIG['test_dir'], "*.mp4"))
    mtcnn = MTCNN(keep_all=True, device=CONFIG['device'], margin=0, min_face_size=40)
    transform = get_transforms('val')
    
    results = []
    
    with torch.no_grad():
        for video_path in tqdm(test_videos):
            filename = os.path.basename(video_path)
            
            # Extract faces on the fly
            faces_np = extract_faces_from_video(video_path, mtcnn, required_size=CONFIG['input_size'], sample_rate=CONFIG['frame_sample_rate'])
            
            if len(faces_np) == 0:
                final_prob = 0.5
            else:
                # Prepare batch
                face_tensors = []
                for face in faces_np:
                    face_pil = Image.fromarray(face)
                    face_tensors.append(transform(face_pil))
                
                batch = torch.stack(face_tensors).to(CONFIG['device'])
                
                # Predict
                logits_b0 = model_b0(batch)
                logits_b3 = model_b3(batch)
                
                probs_b0 = torch.sigmoid(logits_b0).cpu().numpy()
                probs_b3 = torch.sigmoid(logits_b3).cpu().numpy()
                
                mean_prob_b0 = np.mean(probs_b0)
                mean_prob_b3 = np.mean(probs_b3)
                
                final_prob = 0.7 * mean_prob_b3 + 0.3 * mean_prob_b0
            
            final_prob = float(np.clip(final_prob, 0.0, 1.0))
            label = "FAKE" if final_prob > 0.5 else "REAL"
            
            results.append({
                "filename": filename,
                "label": label,
                "probability": final_prob
            })
            
    df = pd.DataFrame(results)
    df.to_csv(CONFIG['submission_file'], index=False)
    print(f"[INFO] Submission saved to {CONFIG['submission_file']}")

# --- Main ---

if __name__ == "__main__":
    setup_environment()
    
    # Check if data needs processing
    if not os.path.exists(CONFIG['processed_train_dir']) or len(os.listdir(CONFIG['processed_train_dir'])) == 0:
        prepare_dataset()
    else:
        print("[INFO] Processed dataset found. Skipping extraction.")
        
    # Dataset & Loader
    full_dataset = DeepfakeDataset(CONFIG['processed_train_dir'], transform=get_transforms('train'))
    if len(full_dataset) > 0:
        train_loader = DataLoader(full_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0) # num_workers=0 for Windows
        
        # Train Models
        train_model('B0', train_loader, epochs=CONFIG['epochs'])
        train_model('B3', train_loader, epochs=CONFIG['epochs'])
    else:
        print("[WARNING] No training data found.")
        
    # Inference
    inference_pipeline()
