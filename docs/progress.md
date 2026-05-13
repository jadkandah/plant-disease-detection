# Progress

This page records the main project progress in a cleaned format.

## Dataset Work

The project started by researching public plant disease datasets and selecting
datasets that cover common crop diseases.

Datasets were downloaded from Kaggle and other public sources, then merged into
a unified structure:

```text
jordan_dataset/
  train/
  val/
  test/
```

The final active dataset contains:

- 11 crop groups.
- 55 classes.
- 65,281 images.
- Train, validation, and test splits.

## Class Coverage

Current crop groups:

- Apple
- Cauliflower
- Eggplant
- Grape
- Maize
- Olive
- Orange
- Peach
- Potato
- Tomato
- Wheat

The database seed data and model class list use the same `Crop___Disease` class-key format.

## Preprocessing Progress

Implemented preprocessing steps:

- Image loading and RGB conversion.
- Image validation and corrupted-file handling.
- Train-time augmentation using Albumentations.
- ImageNet normalization.
- Background removal experiments using MobileSAM.
- Frontend common preprocessing before upload or local inference.
- Backend-only SAM leaf extraction for online mode.
- Backend quality-check code kept as a fallback, but the active common path is frontend-side.

## Early Experiments

Initial experiments compared online and offline candidate models with smaller
image size and fewer epochs.

Models tested:

- ResNet50
- EfficientNet-B0
- MobileNetV3-Small
- MobileNetV3-Large

Metrics tracked:

- Test accuracy
- Test mIoU
- Model size
- Number of parameters
- Training time

Early results showed that EfficientNet-B0 and MobileNetV3-Small were strong
candidates, but later full-dataset 512px experiments became more important.

## Current Training Configuration

Current experiment configs are stored in:

- `src/dissdetector/config/online_models.yaml`
- `src/dissdetector/config/offline_models.yaml`

Online experiment runs include:

- `multimodal_resnet50_background_removed_512_epochs40`
- `image_only_resnet50_background_removed_512_epochs40`
- `metadata_only_mlp_background_removed_512_epochs40`
- `late_fusion_multimodal_resnet50_background_removed_512_epochs40`

Offline experiment runs include:

- `mobilenet_v3_small_512_background_removed_epochs25`
- `mobilenet_v3_large_512_background_removed_epochs30`
- `efficientnet_b0_512_background_removed_epochs30`

## Saved Model Results

Important saved model manifests show these results:

| Model | Scope | Test Accuracy | Test mIoU | Size |
| --- | --- | ---: | ---: | ---: |
| `image_only_resnet50_background_removed_512_epochs40` | Backend online image-only | 0.9531 | 0.8168 | 90.42 MB |
| `late_fusion_multimodal_resnet50_background_removed_512_epochs40` | Multimodal experiment | 0.9538 | 0.8141 | 94.73 MB |
| `metadata_only_mlp_background_removed_512_epochs40` | Metadata-only experiment | 0.0679 | 0.0109 | 0.09 MB |
| `mobilenet_v3_small_512_background_removed_epochs25` | Offline-oriented image-only | saved checkpoint | used for backend offline mode | 6.2 MB |

The metadata-only model performed poorly by itself, which confirms that weather
metadata alone is not enough for disease classification in this dataset.

## Backend Progress

Implemented backend features:

- Custom email-based user model.
- JWT registration, login, refresh, profile, and password change.
- Disease metadata API.
- Prediction endpoint with multipart image upload.
- Online and offline backend inference modes.
- Prediction history model.
- Offline history sync endpoint.
- Admin-only users, predictions, disease CRUD, and dashboard statistics.

## Mobile App Progress

Implemented mobile features:

- Login and sign-up screens.
- Camera image capture.
- Gallery image selection.
- Common frontend preprocessing through `imagePreprocessing.ts`.
- Online image upload to the backend.
- Result display with crop and disease information.
- English and Arabic text support.
- Model mode context for online/offline selection.
- Network-state checks.
- Offline queue using AsyncStorage.
- Automatic sync when connectivity returns.
- Local disease metadata for offline result display.

## Export Progress

The export utility converts image-only PyTorch checkpoints to ONNX.

Current exported files:

```text
mobile-app/assets/mobile_models/offline_model.onnx
mobile-app/assets/mobile_models/offline_model_manifest.json
```

The manifest records:

- Model name.
- Class mapping.
- Image size.
- Input and output names.
- Normalization values.
- Runtime notes.

Multimodal checkpoints are intentionally blocked from offline export because
they require metadata features and backend-only preprocessing.

## Documentation Progress

The MkDocs documentation has been cleaned and organized for Appendix C.

The documentation now covers:

- Project overview.
- Dataset and preprocessing background.
- Architecture and API design.
- Code-level module responsibilities.
- Training, inference, export, and experiment tracking.
- Project plan and final progress summary.
