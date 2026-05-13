# Code Documentation

This page explains the main code modules and their responsibilities.

The project uses MkDocs for documentation. The documentation is based on the
actual repository structure and focuses on modules, pipelines, and components
that are present in the codebase.

## Documentation Structure

MkDocs reads Markdown files from `docs/`.

The documentation is organized to support Appendix C:

- `index.md`: project overview.
- `background.md`: dataset, preprocessing, metadata, and experiments.
- `design.md`: architecture, API, backend, mobile app, and export flow.
- `code-documentation.md`: module-level code documentation.
- `plan.md`: scope, milestones, risks, and deliverables.
- `progress.md`: implementation progress and experiment summary.

## ML Package

Main package:

```text
src/dissdetector/
```

### `config/runtime.py`

Defines shared runtime settings for ML code.

Responsibilities:

- Project root and dataset paths.
- Default dataset variant.
- Default seed, patience, selection metric, and MLflow tracking URI.
- Dataset split resolution for `background_removed` and `original` variants.
- Selection metric validation.
- Reproducibility helpers.

### `models/model_factory.py`

Creates PyTorch models by name.

Supported model families:

- `resnet50`
- `image_only_resnet50`
- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `multimodal_resnet50`
- `metadata_only_mlp`
- `late_fusion_multimodal_resnet50`

The file also defines the multimodal architectures:

- `MultiModalResNet50`: combines ResNet50 image features with metadata features before classification.
- `MetadataOnlyMLP`: uses only metadata features for comparison.
- `LateFusionMultiModalResNet50`: creates image logits and metadata logits, then fuses them for the final prediction.

### `training/train_core.py`

Handles the image-only training path.

Responsibilities:

- Build train and validation/test transforms.
- Build a shared class mapping across train, validation, and test splits.
- Load image datasets with safe handling for unreadable files.
- Create PyTorch dataloaders.
- Train image-only models.
- Use early stopping.
- Save and load training checkpoints.
- Evaluate the model on the test split.
- Compute accuracy and mIoU.

This module is used for standard image classification models such as ResNet50,
EfficientNet-B0, and MobileNetV3.

### `training/multimodal_core.py`

Handles the metadata-aware training path.

Responsibilities:

- Load `metadata_weather.csv`.
- Validate metadata columns.
- Normalize metadata features using training split statistics.
- Build multimodal datasets that return image tensors, metadata tensors, and labels.
- Support metadata-only experiments.
- Compute class weights for imbalanced classes.
- Train multimodal models with early stopping.
- Use a two-phase optimizer strategy for ResNet-based multimodal models.
- Evaluate loss, accuracy, mIoU, and macro-F1.

Metadata features:

- `temp_c`
- `humidity_pct`
- `wind_m_s`
- `precip_mm`
- `soil_moisture_pct`

### `training/metrics.py`

Contains shared metric helpers.

Responsibilities:

- Build a confusion matrix from predictions and labels.
- Compute mean Intersection over Union from the confusion matrix.

### `training/early_stopping.py`

Provides an early stopping helper used by the training loops.

It tracks the selected validation metric and stops training when there is no
improvement for the configured patience.

## Experiment Code

Main folder:

```text
src/dissdetector/experiments/
```

### `run_mlflow_experiment.py`

Runs one configured experiment.

Responsibilities:

- Set the MLflow tracking URI.
- Select image-only or multimodal training based on model name.
- Log parameters and metrics.
- Save PyTorch model weights.
- Save a JSON manifest with class mapping, image size, config, and metrics.
- Log model artifacts to MLflow.

### `run_online_models.py`

Loads `online_models.yaml` and runs each online experiment configuration.

### `run_offline_models.py`

Loads `offline_models.yaml` and runs each offline experiment configuration.

### `rank_models.py`

Reads MLflow runs and ranks models using a weighted score.

Ranking considers:

- Test accuracy.
- Test mIoU.
- Validation accuracy.
- Validation loss.
- Training speed.
- Parameter count.
- Model size.

## Preprocessing Code

Main folder:

```text
src/dissdetector/preprocessing/
```

### `generate_metadata_weather.py`

Generates synthetic weather metadata for images.

Responsibilities:

- List dataset images.
- Assign plausible timestamps by disease type.
- Fetch historical weather from Open-Meteo.
- Use fallback weather values if the API request fails.
- Write `jordan_dataset/metadata_weather.csv`.

### `remove_background_sam.py`

Removes image backgrounds using MobileSAM.

Responsibilities:

- Load MobileSAM.
- Generate a prompt box.
- Select and clean the best leaf mask.
- Replace the background with white.
- Crop around the detected leaf.
- Save processed images.

## Export Code

Main file:

```text
src/dissdetector/export/export_mobile_model.py
```

Responsibilities:

- Export image-only PyTorch checkpoints to ONNX.
- Resolve model metadata from manifests or command-line arguments.
- Rebuild class mappings when needed.
- Write an ONNX model file.
- Write an offline model manifest.
- Validate ONNX Runtime output against PyTorch output when possible.
- Reject multimodal models for offline export.

The export output is consumed by the mobile app:

```text
mobile-app/assets/mobile_models/offline_model.onnx
mobile-app/assets/mobile_models/offline_model_manifest.json
```

## Backend Code

Main folder:

```text
backend/
```

### `backend/config/`

Django project configuration.

Responsibilities:

- App registration.
- URL routing.
- JWT authentication settings.
- SQLite database configuration.
- CORS settings.
- Media file configuration.

### `backend/authentication/`

User authentication app.

Responsibilities:

- Custom email-based user model.
- Registration.
- Login.
- Token refresh.
- Profile endpoint.
- Password change endpoint.

### `backend/diseases/`

Disease metadata app.

Responsibilities:

- Store class-key disease records.
- Store English and Arabic crop/disease names.
- Mark records as healthy or diseased.
- Provide read-only disease metadata API.
- Seed disease data using `seed_data.py`.

### `backend/prediction/`

Prediction app.

Important files:

- `views.py`
- `serializers.py`
- `inference.py`
- `preprocessing/pipeline.py`
- `preprocessing/quality.py`
- `preprocessing/sam_utils.py`

Responsibilities:

- Validate uploaded prediction images.
- Decode images using OpenCV.
- Apply backend-only preprocessing.
- Apply optional SAM preprocessing in online mode.
- Keep backend quality checks available as a fallback, while the active common preprocessing path runs in the frontend.
- Load PyTorch models as reusable singletons.
- Run inference.
- Convert output index to class key.
- Return confidence and disease metadata.
- Save synced prediction history records.

### `backend/history/`

Prediction history app.

Responsibilities:

- Store prediction records.
- Filter history by current user.
- Filter by crop or date.
- Receive offline prediction records through `/api/history/sync/`.
- Preserve offline prediction timestamps when syncing.

### `backend/adminpanel/`

Admin API app.

Responsibilities:

- Admin-only user list.
- Admin-only prediction list.
- Admin dashboard statistics.
- Admin disease metadata CRUD.

## Mobile App Code

Main folder:

```text
mobile-app/
```

### `App.tsx`

Defines the app providers, navigation stack, bottom tabs, and sync status banner.

Providers:

- `LanguageProvider`
- `ModelModeProvider`
- `AuthProvider`

### `src/services/auth/apiClient.ts`

Configures the Axios API client.

Responsibilities:

- Select base API URL for web or native.
- Add JWT bearer token to requests.
- Refresh expired access tokens.
- Clear stored tokens when refresh fails.

### `src/services/preprocessing/imagePreprocessing.ts`

Runs the common frontend preprocessing step before upload or local inference.

Responsibilities:

- Provide the shared `preprocessImage` entry point used by camera and gallery screens.
- Check image quality in the web implementation using canvas pixel data.
- Reject near-black, blurry, too dark, too bright, or low-contrast images.
- Resize web images to the model input size before upload or local inference.
- Return the original URI on native when browser canvas APIs are unavailable.

### `src/store/ModelModeContext.tsx`

Tracks the selected prediction mode.

Responsibilities:

- Store selected model mode in AsyncStorage.
- Detect whether online mode can be used.
- Force offline mode when network access is unavailable.

### `src/services/offline/localInference.ts`

Runs local ONNX inference for the Expo web implementation.

Responsibilities:

- Load ONNX Runtime Web.
- Load the exported ONNX model and manifest.
- Convert browser images to normalized tensors.
- Run model inference.
- Apply softmax.
- Map predicted index to class key.
- Build local disease metadata.

### `src/services/offline/offlineQueue.ts`

Stores and syncs offline predictions.

Responsibilities:

- Add offline prediction results to AsyncStorage.
- Read pending queue items.
- Remove synced items.
- Send records to `/api/history/sync/`.

### `src/services/sync/useAutoSync.ts`

Automatically syncs queued offline predictions.

Sync triggers:

- Connectivity becomes available.
- App returns to foreground.
- Periodic polling while online.
- Queue updates.

### `src/screens/Home/CameraScreen.tsx`

Captures images with Expo Camera and starts prediction.

### `src/screens/Home/GalleryScreen.tsx`

Selects images from the gallery and starts prediction.

### `src/screens/Home/ResultScreen.tsx`

Displays the prediction result.

### `src/screens/History/HistoryScreen.tsx`

Displays server history in online mode and queued offline results in offline mode.

## Data and Model Artifacts

Important artifact locations:

| Path | Purpose |
| --- | --- |
| `jordan_dataset/` | Active dataset |
| `jordan_dataset/metadata_weather.csv` | Generated weather metadata |
| `saved_models/` | PyTorch checkpoints and manifests |
| `mlruns/` | MLflow run artifacts |
| `mlflow.db` | MLflow SQLite tracking database |
| `mobile_sam.pt` | MobileSAM checkpoint |
| `mobile-app/assets/mobile_models/` | ONNX model and manifest |
