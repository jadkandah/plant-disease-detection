# Plan

This page summarizes the project plan and deliverables.

## Objectives

The project objectives are:

- Build a plant disease classification dataset focused on relevant crops.
- Train and compare multiple image classification models.
- Evaluate models using accuracy, mIoU, model size, and parameter count.
- Provide backend prediction through a Django REST API.
- Provide a mobile interface for camera and gallery image input.
- Support online and offline-oriented prediction workflows.
- Store prediction history and sync offline records when connectivity returns.
- Document the system clearly for Appendix C.

## Scope

Included:

- Image classification for plant disease detection.
- Dataset preprocessing and background removal experiments.
- PyTorch model training and evaluation.
- MLflow experiment tracking.
- Django backend APIs.
- React Native / Expo mobile app.
- ONNX export for an image-only offline model.
- MkDocs project documentation.

Not included in the current implementation:

- Object detection with bounding boxes.
- Real field sensor integration.
- Real per-image capture-time weather metadata.
- Production deployment hardening.
- Native mobile ONNX runtime integration beyond the current Expo web-oriented local inference path.

## Milestones

| Milestone | Status | Output |
| --- | --- | --- |
| Dataset selection | Done | Public datasets selected and merged |
| Dataset organization | Done | `jordan_dataset/train`, `val`, and `test` |
| Background removal | Done | MobileSAM preprocessing scripts and background-removed dataset variant |
| Baseline training | Done | ResNet50, EfficientNet-B0, MobileNetV3 experiments |
| Full dataset training | Done | 512px background-removed models |
| Multimodal experiments | Done | Image + synthetic weather metadata training path |
| Backend API | Done | Auth, prediction, disease info, history, admin routes |
| Mobile app | Done | Auth, camera, gallery, result, history, settings |
| Offline queue and sync | Done | AsyncStorage queue and sync endpoint |
| ONNX export | Done | Exported image-only ONNX model and manifest |
| Documentation cleanup | Done | MkDocs appendix-ready documentation |

## Implementation Strategy

The system was developed in layers:

1. Prepare and normalize the dataset.
2. Train image-only baseline models.
3. Add experiment tracking and model ranking.
4. Add background removal and larger 512px experiments.
5. Add metadata-aware training experiments.
6. Build backend prediction and history APIs.
7. Build mobile workflows for image input and result display.
8. Add offline queueing and model export.
9. Clean and organize documentation.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Class imbalance | Use augmentation and class-aware evaluation metrics |
| Corrupted or invalid images | Skip bad samples during training and reject invalid uploads at runtime |
| Large model size | Keep MobileNetV3-Small as lightweight offline-oriented model |
| Backend model load time | Load models as singletons and reuse them |
| Offline prediction limitations | Export only image-only models and keep multimodal models backend-side |
| Synthetic metadata bias | Clearly mark generated weather metadata as synthetic |
| Connectivity loss | Queue offline records and sync when online access returns |

## Deliverables

Final deliverables include:

- Trained PyTorch checkpoints in `saved_models/`.
- Model manifests with class mappings and metrics.
- MLflow tracking database and runs.
- Django backend source code.
- React Native / Expo mobile app source code.
- Exported ONNX model and manifest for offline-oriented inference.
- MkDocs documentation under `docs/`.
