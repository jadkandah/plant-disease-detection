import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import os
from pathlib import Path
from tqdm import tqdm
import time
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# --- Configuration ---
# *** CRITICAL: Set the absolute path to your project root ***
# This path is based on the location confirmed in previous steps.
ROOT = Path("/Users/sanadmadani/plant-disease-detection/plant-disease-detection")

DATASET_PATH = ROOT / 'jordan_dataset' / 'images'
NUM_CLASSES = 52  # Based on the 52 classes processed previously
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 512
MODEL_OUTPUT_PATH = ROOT /'src'/'dissdetector'/'resnet_50_plant_disease.pth'

# Check for GPU availability'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# --- End Configuration ---


# Helper class to combine Albumentations with PyTorch Transforms
class AlbumentationsImageFolder(datasets.ImageFolder):
    """
    Wrapper for ImageFolder to apply Albumentations transforms.
    The transform parameter passed to the constructor will be an A.Compose object.
    """
    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, transform=None, **kwargs)
        self.albumentation_transform = transform

    def __getitem__(self, index):
        # 1. Load image and target (label)
        path, target = self.samples[index]
        
        # We use PIL to load, then convert to numpy array (required by Albumentations)
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        
        # 2. Apply Albumentations transform
        if self.albumentation_transform:
            augmented = self.albumentation_transform(image=img_np)
            # The output of ToTensorV2 is a PyTorch Tensor
            img_tensor = augmented['image']
        else:
            # Fallback for simple PIL/Tensor conversion if no Albumentations used
            img_tensor = transforms.ToTensor()(img)
        
        return img_tensor, target

# --- Data Transformations (Online Augmentation) ---

# Normalization values for ImageNet pre-trained models
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# 1. Training Augmentations (Online Augmentation: runs every time an image is loaded)
# This prevents overfitting and forces the model to generalize.
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomCrop(int(IMAGE_SIZE * 0.8), int(IMAGE_SIZE * 0.8), p=0.5), # Randomly crop 80%
    A.HorizontalFlip(p=0.5), # Fixed from A.Flip()
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD), # Standard normalization
    ToTensorV2(), # Converts the numpy array to a PyTorch Tensor
])

# 2. Validation/Test Transforms (Only resize and normalize)
# No random transformations here, as we need consistent evaluation.
val_test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

# Import OpenCV for Albumentations' border_mode. This must be imported
# for the code to run correctly when Albumentations is called.
try:
    import cv2
except ImportError:
    print("Warning: OpenCV (cv2) not found. Albumentations may fail.")
    pass

# --- Data Loading Function ---
def load_data(data_dir):
    """Sets up PyTorch Datasets and DataLoaders."""
    
    # The structure of the 'images' folder is perfect for ImageFolder/AlbumentationsImageFolder
    # /images/
    #   /Apple/
    #     /Apple_scab/
    #     /healthy/
    #   /Wheat/
    #     /healthy/
    #     /Aphid/
    # ...
    
    # We load the data separately for train, val, and test splits
    image_datasets = {
        'train': AlbumentationsImageFolder(
            os.path.join(data_dir, 'train'),
            train_transforms
        ),
        'val': AlbumentationsImageFolder(
            os.path.join(data_dir, 'val'),
            val_test_transforms
        ),
        'test': AlbumentationsImageFolder(
            os.path.join(data_dir, 'test'),
            val_test_transforms
        )
    }
    
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=(x == 'train'), # Only shuffle the training data
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 4 # Use half of available CPU cores
        )
        for x in ['train', 'val', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    # Print the class mapping generated automatically by PyTorch:
    # This dictionary maps the textual folder name (e.g., 'Apple_scab') to an integer (e.g., 0)
    print("\n--- Model Label Mapping (Text to Integer) ---")
    print(image_datasets['train'].class_to_idx)
    print("---------------------------------------------")

    return dataloaders, dataset_sizes, image_datasets['train'].class_to_idx


# --- Model Loading Function ---
def load_model(num_classes):
    """Loads pre-trained ResNet-50 and modifies the final layer."""
    
    # Load ResNet-50 pre-trained on ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze all the base layers (we only want to train the final classification layer)
    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of features in the last fully connected layer
    num_ftrs = model.fc.in_features
    
    # Replace the last layer with a new one that matches our number of classes (52)
    # The model will learn a new classification based on the features learned by ImageNet
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(DEVICE)
    print(f"\nLoaded ResNet-50 model with final layer adapted for {num_classes} classes.")
    return model


# --- Training Function ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    """The main training loop."""
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

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} phase'):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad() # Zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Step the scheduler *after* the epoch statistics are calculated
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# --- Main Execution ---
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
    
    # Observe that all parameters are being optimized
    # Since we froze the base layers, only the final 'model.fc' layer parameters will update
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

    # Decay LR by a factor of 0.1 every 7 epochs
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
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc='Test phase'):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Accuracy: {test_acc:.4f} (on {dataset_sizes["test"]} images)')