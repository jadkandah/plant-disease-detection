1/11/2025
######
Research & Dataset Selection
We researched publicly available datasets of plant diseases. The chosen datasets were:
Dataset	Reason for Use
vipoooool/new-plant-diseases-dataset:
Base dataset covering multiple crops and disease types. Useful for general plant disease classification.
habibulbasher01644/olive-leaf-image-dataset:
Contains images of olive leaves, including common diseases in Jordan (like Peacock Spot).
serhathoca/zeytin	Additional:
olive leaf images for better representation of olive diseases.
kushagra3204/wheat-plant-diseases:
Focused wheat plant disease dataset to include key crops cultivated in Jordan.
Why classification and not object detection:
Object detection identifies and localizes multiple objects in an image, which is more complex and resource-intensive.
Our goal is to classify images of leaves by disease type, which is simpler and sufficient for early detection and decision-making in agriculture.
Classification allows faster model training and easier integration into mobile/field apps for farmers.
######
Dataset Download
Kaggle datasets were downloaded using the Kaggle CLI.
Any other datasets (URLs) were downloaded using urllib.
Archives (.zip, .tar, .tgz) were extracted automatically.
Olive Leaf Image Dataset — https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset
kaggle.com

Olive Leaf Disease Dataset (Zeytin) — https://www.kaggle.com/datasets/serhathoca/zeytin
kaggle.com

Wheat Plant Diseases — https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases
kaggle.com

20k+ Multi‑Class Crop Disease Images — https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images

Main dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?utm_source=chatgpt.com
#####
GOAL STRUCTURE OF THE DATASET 
jordan_dataset/
├── train/
│   ├── Apple/
│   │   ├── Apple_scab/
│   │   └── healthy/
│   ├── Olive/
│   │   ├── Peacock_spot/
│   │   └── healthy/
│   ├── Wheat/
│   │   ├── Leaf_rust/
│   │   └── healthy/
├── test/
│   ├── Apple/
│   │   ├── Apple_scab/
│   │   └── healthy/
│   ├── Olive/
│   │   ├── Peacock_spot/
│   │   └── healthy/
│   ├── Wheat/
│   │   ├── Leaf_rust/
│   │   └── healthy/
├── valid/
│   ├── Apple/
│   │   ├── Apple_scab/
│   │   └── healthy/
│   ├── Olive/
│   │   ├── Peacock_spot/
│   │   └── healthy/
│   ├── Wheat/
│   │   ├── Leaf_rust/
│   │   └── healthy/
####3 these were used in the code to help us merge the datasets and seprate them for training testing and validation 
├── metadata.csv
├── metadata_train.csv
├── metadata_val.csv
├── metadata_test.csv
└── class_counts.csv

#######
in the code we added data Augmentation (manipulation) for classes who got less than 500 images.
we used RandomRotate90(), Flip(), Transpose(), RandomBrightnessContrast(), ShiftScaleRotate()