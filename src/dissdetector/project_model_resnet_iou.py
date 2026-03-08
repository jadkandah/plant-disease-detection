import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time
import copy
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score # <--- NEW IMPORTS

# --- Configuration ---
# *** CRITICAL: Set the absolute path to your project root ***
ROOT = Path("/Users/sanadmadani/plant-disease-detection/plant-disease-detection")

DATASET_PATH = ROOT / 'jordan_dataset' / 'images'
NUM_CLASSES = 52
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
IMAGE_SIZE = 512
MODEL_OUTPUT_PATH = ROOT / 'resnet_50_plant_disease.pth'

# Check for GPU availability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Import OpenCV for Albumentations' border_mode.
try:
    import cv2
except ImportError:
    print("Warning: OpenCV (cv2) not found. Albumentations may fail.")
    # Define a dummy variable if cv2 is not available to prevent crash
    cv2 = type('DummyCV2', (object,), {'BORDER_CONSTANT': 0})
    pass
# --- End Configuration ---


# --- NEW: Metric Calculation Utility ---
def calculate_metrics(y_true, y_pred, average='macro'):
    """Calculates F1-score and Jaccard Index (IoU) for multi-class classification."""
    
    # F1-score is typically the gold standard for classification
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Jaccard Index (IoU) is equivalent to IoU in classification
    jaccard = jaccard_score(y_true, y_pred, average=average, zero_division=0)
    
    return f1, jaccard

# Helper class to combine Albumentations with PyTorch Transforms (as defined previously)
class AlbumentationsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, transform=None, **kwargs)
        self.albumentation_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        
        if self.albumentation_transform:
            augmented = self.albumentation_transform(image=img_np)
            img_tensor = augmented['image']
        else:
            img_tensor = transforms.ToTensor()(img)
        
        return img_tensor, target

# --- Data Transformations (Online Augmentation) ---

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomCrop(int(IMAGE_SIZE * 0.8), int(IMAGE_SIZE * 0.8), p=0.5),
    A.HorizontalFlip(p=0.5), 
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

val_test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

# --- Data Loading Function (No changes needed) ---
def load_data(data_dir):
    image_datasets = {
        'train': AlbumentationsImageFolder(os.path.join(data_dir, 'train'), train_transforms),
        'val': AlbumentationsImageFolder(os.path.join(data_dir, 'val'), val_test_transforms),
        'test': AlbumentationsImageFolder(os.path.join(data_dir, 'test'), val_test_transforms)
    }
    
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=(x == 'train'), 
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 4
        )
        for x in ['train', 'val', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    print("\n--- Model Label Mapping (Text to Integer) ---")
    print(image_datasets['train'].class_to_idx)
    print("---------------------------------------------")

    return dataloaders, dataset_sizes, image_datasets['train'].class_to_idx


# --- Model Loading Function (No changes needed) ---
def load_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(DEVICE)
    print(f"\nLoaded ResNet-50 model with final layer adapted for {num_classes} classes.")
    return model


# --- Training Function (UPDATED) ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    """The main training loop with F1 and Jaccard calculation."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # --- NEW: Lists to collect all true and predicted labels for metrics ---
            all_labels = []
            all_preds = []
            
            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} phase'):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad() 

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # --- NEW: Collect labels and predictions ---
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # --- NEW: Calculate F1 and Jaccard (IoU) ---
            f1, jaccard = calculate_metrics(all_labels, all_preds)

            # --- UPDATED PRINT STATEMENT ---
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {f1:.4f} Jaccard (IoU): {jaccard:.4f}')

            # deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


# --- Main Execution (UPDATED) ---
if __name__ == '__main__':
    
    # 1. Load Data
    data_dir = DATASET_PATH
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found at {data_dir}. Please check the hardcoded ROOT path.")
        sys.exit(1)

    dataloaders, dataset_sizes, class_to_idx = load_data(DATASET_PATH)

    # 2. Load Model
    model_ft = load_model(NUM_CLASSES)

    # 3. Define Loss Function, Optimizer, and Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 4. Train
    print("\nStarting Training...")
    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=NUM_EPOCHS
    )

    # 5. Save Model
    torch.save(model_ft.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nModel saved successfully to {MODEL_OUTPUT_PATH}")

    # 6. Test/Final Evaluation (Optional, but recommended)
    print("\n--- Final Test Set Evaluation ---")
    model_ft.eval()
    
    test_labels = []
    test_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc='Test phase'):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    # Calculate final metrics on the test set
    test_f1, test_jaccard = calculate_metrics(test_labels, test_preds)
    test_acc = np.sum(np.array(test_preds) == np.array(test_labels)) / len(test_labels)
    
    print(f'Test Accuracy: {test_acc:.4f} (on {dataset_sizes["test"]} images)')
    print(f'Test F1-score (Macro): {test_f1:.4f}')
    print(f'Test Jaccard Index (IoU): {test_jaccard:.4f}')