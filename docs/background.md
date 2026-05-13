# Background

This page summarizes the dataset, preprocessing, metadata generation, and model
experiments used by the system.

## Project Context

The goal is to classify plant leaf images into healthy or diseased categories.

The system focuses on classification rather than object detection because:

- A single prediction per image is enough for the current user workflow.
- Classification is simpler to train and deploy than bounding-box detection.
- The mobile app needs fast feedback for farmers and field users.
- The system can still apply preprocessing before classification.

## Dataset

The active dataset is stored under `jordan_dataset/`.

The current dataset contains:

- 11 crop groups.
- 55 total classes.
- `train`, `val`, and `test` splits.
- A background-removed dataset variant used by the current training configs.
- An `Original` folder for the raw-image variant.

Current image counts:

| Split | Images |
| --- | ---: |
| Train | 48,213 |
| Validation | 8,536 |
| Test | 8,532 |
| Total | 65,281 |

The class names follow this format:

```text
Crop___Disease
```

Example:

```text
Tomato___Early_blight
Olive___Peacock_spot
Wheat___Yellow_Rust
Apple___healthy
```

## Dataset Sources

The dataset was built by combining public plant disease image datasets, including:

- `vipoooool/new-plant-diseases-dataset`
- `habibulbasher01644/olive-leaf-image-dataset`
- `serhathoca/zeytin`
- `kushagra3204/wheat-plant-diseases`
- `jawadali1045/20k-multi-class-crop-disease-images`

The final repository dataset is organized by split, crop, and disease class:

```text
jordan_dataset/
  train/
    Crop/
      Disease/
        image.png
  val/
    Crop/
      Disease/
        image.png
  test/
    Crop/
      Disease/
        image.png
```

## Preprocessing

The project uses three preprocessing areas:

- Dataset preprocessing for training.
- Common frontend preprocessing before prediction.
- Backend-only preprocessing for online inference.

Training preprocessing includes:

- Reading RGB image files.
- Resizing or cropping to the configured image size.
- Normalization using ImageNet mean and standard deviation.
- Albumentations transforms such as crop, flip, affine transform, brightness/contrast changes, blur, and shadow augmentation.
- Safe handling of corrupted or unreadable images.

Runtime frontend preprocessing includes:

- Rejecting corrupted, empty, black, blurry, too dark, too bright, or low-contrast images.
- Returning a processed image URI for upload or local inference.
- Resizing the image to the model input size in the web implementation.

Backend-specific preprocessing includes:

- Decoding the uploaded image with OpenCV.
- Applying MobileSAM leaf extraction in online mode when available.
- Skipping SAM in offline backend mode.
- Leaving final resize and normalization to the inference transform.

## Background Removal

The repository includes MobileSAM-based background-removal scripts.

Important files:

- `src/dissdetector/preprocessing/remove_background_sam.py`
- `src/dissdetector/preprocessing/sam_background.py`
- `backend/prediction/preprocessing/sam_utils.py`

The training-side SAM script:

- Loads the `mobile_sam.pt` checkpoint.
- Prompts SAM using a central box.
- Selects the best mask.
- Cleans the mask with OpenCV morphology.
- Places the leaf on a white background.
- Crops around the detected leaf area.

The backend SAM utility is optional. If MobileSAM or the checkpoint is missing,
the backend returns the original image instead of failing the whole request.

## Weather Metadata

The multimodal training pipeline uses generated weather metadata stored at:

```text
jordan_dataset/metadata_weather.csv
```

The metadata includes:

- `image_rel_path`
- `temp_c`
- `humidity_pct`
- `wind_m_s`
- `precip_mm`
- `soil_moisture_pct`
- `random_datetime`
- `crop_name`
- `disease_name`

The metadata generator assigns plausible timestamps to images and joins them
with Open-Meteo historical weather for Amman, Jordan. Because the image datasets
do not contain real capture timestamps, this metadata is synthetic and should be
treated as an experiment input, not field-measured sensor data.

## Experiment Tracking

MLflow is used to track model experiments.

The project stores:

- Experiment parameters.
- Training and validation metrics.
- Test accuracy and mIoU.
- Model size and parameter counts.
- Saved PyTorch checkpoints.
- JSON manifests with class mappings and configuration.

Experiment configuration files:

- `src/dissdetector/config/online_models.yaml`
- `src/dissdetector/config/offline_models.yaml`

The active MLflow tracking URI is:

```text
sqlite:///mlflow.db
```
