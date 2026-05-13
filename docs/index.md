# Plant Disease Detection System

This documentation describes the Plant Disease Detection System used for the
university project appendix.

The system combines:

- A machine learning pipeline for plant disease classification.
- A Django REST backend for authentication, prediction, disease metadata, history, and admin access.
- A React Native / Expo mobile application for image capture, gallery upload, prediction display, offline queueing, and synchronization.
- MkDocs documentation that explains the code structure, pipelines, and implementation decisions.

## System Summary

The project detects plant disease from leaf images.

The current implementation supports:

- 11 crop groups: Apple, Cauliflower, Eggplant, Grape, Maize, Olive, Orange, Peach, Potato, Tomato, and Wheat.
- 55 model classes using the `Crop___Disease` class-key format.
- Image-only models for backend and offline/mobile-oriented inference.
- Metadata-aware training experiments using image features plus generated weather metadata.
- Background-removed and original dataset variants.
- MLflow experiment tracking with saved model checkpoints and manifests.

## Main Runtime Flow

For a normal prediction:

1. The user captures or selects a plant image in the mobile app.
2. The mobile app runs the common frontend preprocessing step before inference.
3. The app sends the processed image to the backend when online, or uses the local offline path where available.
4. In online backend mode, the backend applies extra backend-only preprocessing such as MobileSAM leaf extraction.
5. The selected model returns a class key and confidence score.
6. The backend or local app maps the class key to disease information.
7. The app displays the crop, disease, health status, and confidence.
8. Prediction history is stored locally or on the server depending on connectivity.

Simple pipeline:

```text
image -> frontend common preprocessing -> optional online backend preprocessing -> model -> class key -> disease info -> result
```

## Documentation Pages

- [Background](background.md): dataset sources, preprocessing, metadata, and experiment context.
- [Design](design.md): architecture, backend API, mobile interaction, inference modes, and export flow.
- [Code Documentation](code-documentation.md): code modules and responsibilities.
- [Plan](plan.md): project objectives, milestones, risks, and deliverables.
- [Progress](progress.md): cleaned project timeline and experiment results.

## Repository Areas

| Area | Purpose |
| --- | --- |
| `src/dissdetector/` | ML preprocessing, training, models, experiments, and export utilities |
| `backend/` | Django REST API, prediction service, disease metadata, user auth, and history |
| `mobile-app/` | React Native / Expo application |
| `jordan_dataset/` | Train, validation, and test image splits |
| `saved_models/` | Trained PyTorch checkpoints and manifest files |
| `mobile-app/assets/mobile_models/` | Exported ONNX model and mobile/web manifest |
| `docs/` | MkDocs documentation pages |
