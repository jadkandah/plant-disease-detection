# jordan_plant_dataset_builder.py
# Purpose: Download, extract, harmonize, and combine multiple plant disease datasets
# into a Jordan-optimized dataset folder structure and a metadata CSV for training.

# --- Imports and Setup ---
import os
import sys
import json
import shutil
import zipfile
import tarfile
import urllib.request
import hashlib
import random
from pathlib import Path

# External library imports
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np 
import albumentations as A

# --- Initial Installs ---
# NOTE: This part is executed outside the standard Python environment. 
# You would typically run this line in your terminal before running the script, 
# or uncomment the 'os.system' line below if you want the script to try and install them.
# The user needs 'kaggle' installed and configured with kaggle.json.
# os.system('pip install --quiet kaggle tqdm pandas scikit-learn pillow albumentations opencv-python numpy') 

# --- Configurations (Cell 3) ---
ROOT = Path.cwd()
DATA_DIR = ROOT / 'data_raw'
OUT_DIR = ROOT / 'jordan_dataset'
METADATA_CSV = OUT_DIR / 'metadata.csv'
IMAGE_SIZE = (512, 512)
RANDOM_SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# --- Helper functions (Cell 4, 5, 6) ---

def run(cmd):
    """Execute shell command and raise error on failure."""
    print("$", cmd)
    r = os.system(cmd)
    if r != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def unzip(path, dest):
    """Unzip or untar archive files."""
    path = Path(path)
    dest = Path(dest)
    print(f"Extracting {path.name}...")
    if path.suffix in ['.zip']:
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(dest)
    elif path.suffix in ['.tgz', '.gz', '.tar'] or path.name.endswith('.tar.gz'):
        with tarfile.open(path, 'r:*') as t:
            t.extractall(dest)
    else:
        print(f"Assuming {path.name} is already a folder.")

def download_kaggle(dataset_slug, dest_folder):
    """Download dataset from Kaggle into dest_folder using kaggle CLI."""
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    cmd = f"kaggle datasets download -d {dataset_slug} -p {dest_folder} --unzip"
    run(cmd)

def download_url(url, dest_path):
    """Download file from a direct URL."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Already downloaded: {dest_path.name}")
        return dest_path
    print(f"Downloading {url.split('/')[-1]} -> {dest_path.name}")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path

def normalize_and_save(src_path, dest_path, size=IMAGE_SIZE):
    """Resize, convert to RGB, and save image."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src_path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        img.save(dest_path, format='JPEG', quality=90)
    except Exception as e:
        print('Failed processing', src_path, e)

def file_hash_name(path):
    """Generate a unique filename based on file content hash."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest() + '.jpg'

# --- Define dataset sources (Cell 7) ---
KAGGLE_DATASETS = [
    'vipoooool/new-plant-diseases-dataset', 
    'habibulbasher01644/olive-leaf-image-dataset',
    'serhathoca/zeytin', 
    'kushagra3204/wheat-plant-diseases', 
]

OTHER_URLS = [] # No direct URL examples are used here.

# --- Download all sources (Cell 8) ---
print('--- Downloading Datasets ---')
''''
for slug in KAGGLE_DATASETS:
    print('Downloading Kaggle:', slug)
    download_kaggle(slug, DATA_DIR)

for url in OTHER_URLS:
    filename = url.split('/')[-1]
    dest = DATA_DIR / filename
    download_url(url, dest)
    try:
        unzip(dest, DATA_DIR)
    except Exception as e:
        print('Could not unzip:', dest, e)
'''
print('Download & extraction complete. Inspect:', DATA_DIR)

# --- Class Mapping (Cell 10) ---
IMG_OUT = OUT_DIR / 'images'
IMG_OUT.mkdir(parents=True, exist_ok=True)

# Full CLASS_MAPPING from the fixed code
CLASS_MAPPING = {
    'Apple___Apple_scab': ('Apple', 'Apple_scab'), 'Apple___Black_rot': ('Apple', 'Black_rot'),
    'Apple___Cedar_apple_rust': ('Apple', 'Cedar_apple_rust'), 'Apple___healthy': ('Apple', 'healthy'),
    'Peach___Bacterial_spot': ('Peach', 'Bacterial_spot'), 'Peach___healthy': ('Peach', 'healthy'),
    'Pepper,_bell___Bacterial_spot': ('Pepper', 'Bacterial_spot'), 'Pepper,_bell___healthy': ('Pepper', 'healthy'),
    'Potato___Early_blight': ('Potato', 'Early_blight'), 'Potato___Late_blight': ('Potato', 'Late_blight'),
    'Potato___healthy': ('Potato', 'healthy'), 'Tomato___Bacterial_spot': ('Tomato', 'Bacterial_spot'),
    'Tomato___Early_blight': ('Tomato', 'Early_blight'), 'Tomato___Late_blight': ('Tomato', 'Late_blight'),
    'Tomato___Leaf_Mold': ('Tomato', 'Leaf_Mold'), 'Tomato___Septoria_leaf_spot': ('Tomato', 'Septoria_leaf_spot'),
    'Tomato___Spider_mites Two-spotted_spider_mite': ('Tomato', 'Spider_mites'),
    'Tomato___Target_Spot': ('Tomato', 'Target_Spot'), 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ('Tomato', 'Yellow_Leaf_Curl_Virus'),
    'Tomato___Tomato_mosaic_virus': ('Tomato', 'Mosaic_virus'), 'Tomato___healthy': ('Tomato', 'healthy'),
    'Grape___Black_rot': ('Grape', 'Black_rot'), 'Grape___Esca_(Black_Measles)': ('Grape', 'Esca'),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ('Grape', 'Leaf_blight'), 'Grape___healthy': ('Grape', 'healthy'),
    'Orange___Haunglongbing_(Citrus_greening)': ('Orange', 'Citrus_greening'),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ('Maize', 'Cercospora_leaf_spot'),
    'Corn_(maize)___Common_rust_': ('Maize', 'Common_rust'), 'Corn_(maize)___Northern_Leaf_Blight': ('Maize', 'Northern_Leaf_Blight'), 
    'Corn_(maize)___healthy': ('Maize', 'healthy'), 'Strawberry___Leaf_scorch': ('Strawberry', 'Leaf_scorch'),
    'Strawberry___healthy': ('Strawberry', 'healthy'), 'Cherry___Powdery_mildew': ('Cherry', 'Powdery_mildew'),
    'Cherry___healthy': ('Cherry', 'healthy'), 'Soybean___healthy': ('Soybean', 'healthy'),
    'Squash___Powdery_mildew': ('Squash', 'Powdery_mildew'), 'Raspberry___healthy': ('Raspberry', 'healthy'),
    'Blueberry___healthy': ('Blueberry', 'healthy'),
    # Olive/Zeytin/Wheat mappings
    'Olive_healthy': ('Olive', 'healthy'), 'Olive_peacock_spot': ('Olive', 'peacock_spot'), 
    'Olive_Verticillium_wilt': ('Olive', 'Verticillium_wilt'), 'Pas': ('Olive', 'Rust'),
    'HalkaliLeke': ('Olive', 'peacock_spot'), 'Saglikli': ('Olive', 'healthy'),
    'Leke': ('Olive', 'unknown_spot'), 'Healthy_wheat': ('Wheat', 'healthy'),
    'Wheat_rust': ('Wheat', 'Rust'), 'Wheat_septoria': ('Wheat', 'Septoria'),
}

# --- Gather images into final structure (Cell 12 - Fixed Logic) ---
print('--- Standardizing and Gathering Images ---')

METADATA = []

COMMON_ROOTS = [
    DATA_DIR,
    
    # VIPOOOOL Plant Village - NEW AND CORRECTED PATHS
    DATA_DIR / 'New Plant Diseases Dataset(Augmented)', # Checks for classes directly here
    DATA_DIR / 'New Plant Diseases Dataset(Augmented)' / 'train', 
    DATA_DIR / 'New Plant Diseases Dataset(Augmented)' / 'New Plant Diseases Dataset(Augmented)' / 'train', 
    
    # SERHATHOCA Zeytin (Olive) - NEW PATHS (These contain folders like Pas, Saglikli)
    DATA_DIR / 'data' / 'hastalikli',  # For diseased classes
    DATA_DIR / 'data' / 'sağlam',     # For healthy classes
    DATA_DIR / 'dataset' / 'hastalikli', # Common alternative from a different unzipping
    DATA_DIR / 'dataset' / 'sağlam', 
    
    # HABIBULBASHER Olive Leaves (Paths retained from before, might need adjustment)
    DATA_DIR / 'olive-leaf-image-dataset', 
    DATA_DIR / 'olive-leaf-image-dataset' / 'olive-leaf-image-dataset',
    
    # KUSHAGRA3204 Wheat Diseases (Paths retained from before, might need adjustment)
    DATA_DIR / 'wheat-plant-diseases',
    DATA_DIR / 'wheat-plant-diseases' / 'train', 
]

for src_folder_name, (crop, disease) in tqdm(CLASS_MAPPING.items(), desc="Processing Classes"):
    src = None
    for root in COMMON_ROOTS:
        potential_src = root / src_folder_name
        if potential_src.is_dir():
            src = potential_src
            break

    if src is None:
        print(f'Warning: Source folder missing for CLASS_MAPPING key: {src_folder_name}. Skipping.')
        continue

    dst_folder = IMG_OUT / crop / disease
    
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                src_file = Path(root) / f
                
                try:
                    new_name = file_hash_name(src_file)
                    dst_file = dst_folder / new_name
                    
                    if dst_file.exists():
                        continue 
                        
                    normalize_and_save(src_file, dst_file)
                    
                    METADATA.append({
                        'image_path': str(dst_file.relative_to(OUT_DIR)),
                        'crop': crop,
                        'disease': disease,
                        'source': src_folder_name,
                    })
                except Exception as e:
                    print(f"Error processing {src_file}: {e}")


pd.DataFrame(METADATA).to_csv(METADATA_CSV, index=False)
print('Collected images:', len(METADATA))

# --- Add local Jordan images (Cell 13) ---
LOCAL_IMAGES_DIR = ROOT / 'local_images'
if LOCAL_IMAGES_DIR.exists():
    print(f"Processing local images from: {LOCAL_IMAGES_DIR.name}")
    for root, dirs, files in os.walk(LOCAL_IMAGES_DIR):
        parts = Path(root).parts
        try:
            crop_index = parts.index(LOCAL_IMAGES_DIR.name) + 1
            crop = parts[crop_index]
            disease = parts[crop_index + 1]
        except (ValueError, IndexError):
            crop = 'Unknown'
            disease = 'Unknown'
            
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                src_file = Path(root) / f
                dst_folder = IMG_OUT / crop / disease
                
                try:
                    new_name = file_hash_name(src_file)
                    dst_file = dst_folder / new_name
                    normalize_and_save(src_file, dst_file)
                    
                    METADATA.append({
                        'image_path': str(dst_file.relative_to(OUT_DIR)),
                        'crop': crop,
                        'disease': disease,
                        'source': 'local_images',
                    })
                except Exception as e:
                    print(f"Error processing local file {src_file}: {e}")

pd.DataFrame(METADATA).to_csv(METADATA_CSV, index=False)
print('After local images, total images:', len(METADATA))

# --- Train / Val / Test split (Cell 14) ---
print('--- Splitting Data ---')
meta = pd.read_csv(METADATA_CSV)
meta['label'] = meta['crop'] + '___' + meta['disease']

if len(meta) > 1 and len(meta['label'].unique()) > 1:
    train, temp = train_test_split(meta, stratify=meta['label'], test_size=0.3, random_state=RANDOM_SEED)
    val, test = train_test_split(temp, stratify=temp['label'], test_size=0.5, random_state=RANDOM_SEED)

    for df, name in [(train,'train'), (val,'val'), (test,'test')]:
        df.to_csv(OUT_DIR / f'metadata_{name}.csv', index=False)
        print(f'{name}:', len(df))
else:
    print("Warning: Not enough data or classes to perform stratified Train/Val/Test split.")

# --- Balance report (Cell 15) ---
if len(meta) > 0:
    counts = meta['label'].value_counts()
    print("\nTop 10 Classes (Balance Report):")
    print(counts.head(10).to_string())
    counts.to_frame('n_images').to_csv(OUT_DIR / 'class_counts.csv')
    print('Saved class counts to', OUT_DIR / 'class_counts.csv')
else:
    print("Cannot generate balance report: Metadata is empty.")

# --- Simple augmentation (Cell 16 - Fixed) ---
print('--- Augmenting Small Classes ---')
MIN_SAMPLES = 500
augmenter = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
])

augmented_count = 0
if 'counts' in locals() and len(counts) > 0:
    for label, n in counts.items():
        if n < MIN_SAMPLES:
            need = MIN_SAMPLES - n
            print(f'Augmenting {label}, need {need} copies.')
            crop, disease = label.split('___')
            folder = IMG_OUT / crop / disease
            images = list(folder.glob('*.jpg'))
            
            i = 0
            while need > 0 and images:
                src = random.choice(images)
                try:
                    img = Image.open(src).convert('RGB')
                    arr = np.array(img)
                    aug = augmenter(image=arr)['image']
                    
                    new_name = src.stem + f'_aug_{i}.jpg'
                    outp = folder / new_name
                    Image.fromarray(aug).save(outp, format='JPEG', quality=90)
                    
                    augmented_count += 1
                    need -= 1
                    i += 1
                except Exception as e:
                    print(f'Augment failed for {src}: {e}')

print('Augmented images created:', augmented_count)

# --- Final checks and readiness (Cell 17) ---
print('\n✨ Dataset Generation Complete! ✨')
print('Final dataset folder:', OUT_DIR)
print('Images root:', IMG_OUT)
print(f'Total final images (including augmentation): {len(list(IMG_OUT.rglob("*.jpg")))}')

# --- Execution Command ---
# To run this script, save it as jordan_plant_dataset_builder.py and execute:
# python jordan_plant_dataset_builder.py