# Design

This page documents the system architecture and how the backend, ML models, and
mobile app work together.

## High-Level Architecture

The system has three main layers:

- ML layer: preprocessing, training, experiment tracking, model export, and saved checkpoints.
- Backend layer: Django REST API for users, predictions, disease metadata, history, and admin views.
- Mobile layer: React Native / Expo app for authentication, image input, prediction display, settings, and history.

Main flow:

```text
Mobile app -> frontend preprocessing -> Django API -> optional online backend preprocessing -> model inference -> disease lookup -> response -> mobile result screen
```

Offline-oriented flow:

```text
Mobile app -> frontend preprocessing -> local ONNX image model -> local disease metadata -> offline queue -> sync when online
```

## Online and Offline Modes

The code distinguishes between `online` and `offline` prediction modes.

| Mode | Where it runs | Model path | Notes |
| --- | --- | --- | --- |
| Online backend | Django backend | `saved_models/image_only_resnet50_background_removed_512_epochs40.pth` | Receives the frontend-preprocessed image, then can apply backend-only MobileSAM leaf extraction |
| Offline backend | Django backend | `saved_models/mobilenet_v3_small_512_background_removed_epochs25.pth` | Receives the frontend-preprocessed image and skips SAM |
| Local app offline | Expo web implementation | `mobile-app/assets/mobile_models/offline_model.onnx` | Uses the frontend-preprocessed image with ONNX Runtime Web |

The export utility intentionally rejects multimodal checkpoints for offline
mobile export because those models require metadata features and backend-side
preprocessing.

## Backend Architecture

The backend is a Django REST Framework project under `backend/`.

Main Django apps:

| App | Responsibility |
| --- | --- |
| `authentication` | Custom email-based user model, registration, login, profile, password change, JWT refresh |
| `prediction` | Image upload validation, backend-only preprocessing, model inference, disease lookup, history creation |
| `diseases` | Disease metadata records keyed by model class keys |
| `history` | Per-user prediction history and offline sync endpoint |
| `adminpanel` | Admin-only users, predictions, disease CRUD, and dashboard statistics |

## API Endpoints

The real routes are defined in `backend/config/urls.py` and app-level `urls.py`
files.

| Endpoint | Method | Purpose | Auth |
| --- | --- | --- | --- |
| `/api/health/` | `GET` | Health check | No |
| `/api/auth/register/` | `POST` | Create account and return JWT tokens | No |
| `/api/auth/login/` | `POST` | Login and return JWT tokens | No |
| `/api/auth/refresh/` | `POST` | Refresh access token | No |
| `/api/auth/profile/` | `GET` | Current user profile | Yes |
| `/api/auth/change-password/` | `PUT/PATCH` | Change current user password | Yes |
| `/api/diseases/` | `GET` | List disease metadata | No |
| `/api/diseases/{id}/` | `GET` | Retrieve one disease metadata record | No |
| `/api/predict/` | `POST` | Upload image and receive prediction | Yes |
| `/api/history/` | `GET/POST` | User prediction history | Yes |
| `/api/history/sync/` | `POST` | Sync queued offline predictions | Yes |
| `/api/admin/users/` | `GET` | Admin user list | Admin |
| `/api/admin/predictions/` | `GET` | Admin prediction list | Admin |
| `/api/admin/predictions/stats/` | `GET` | Admin dashboard statistics | Admin |
| `/api/admin/diseases/` | `GET/POST/PUT/PATCH/DELETE` | Admin disease metadata management | Admin |

## Prediction Request

`/api/predict/` accepts multipart form data:

| Field | Required | Values |
| --- | --- | --- |
| `image` | Yes | Uploaded image file |
| `source_type` | No | `camera` or `gallery` |
| `mode` | No | `online` or `offline` |

The response includes:

- `success`
- `mode`
- `prediction_key`
- `confidence`
- `is_healthy`
- `disease_info`

If the predicted class key is not present in the disease table, the backend
returns the raw prediction with `disease_info: null`.

## Backend Prediction Flow

1. Validate multipart input with `PredictionRequestSerializer`.
2. Decode the uploaded file as an OpenCV image.
3. Skip common quality checks because they are handled by the frontend preprocessing step.
4. In online mode, apply backend-only SAM leaf extraction when available.
5. In offline backend mode, skip SAM preprocessing.
6. Run PyTorch inference using the selected model.
7. Convert the predicted index to a class key.
8. Load matching `DiseaseInfo` from the database.
9. Save a `PredictionRecord`.
10. Return prediction details to the mobile app.

## Mobile App Architecture

The mobile app is an Expo React Native application under `mobile-app/`.

Main areas:

| Area | Responsibility |
| --- | --- |
| `App.tsx` | Navigation, providers, tabs, and sync banner |
| `src/screens/Auth/` | Login and sign-up |
| `src/screens/Home/` | Home, camera, gallery, and result screens |
| `src/screens/History/` | Online history and pending offline history |
| `src/screens/Profile/` | User profile |
| `src/screens/Settings/` | Language and model-mode settings |
| `src/services/auth/apiClient.ts` | Axios client, base URL, JWT injection, token refresh |
| `src/services/preprocessing/imagePreprocessing.ts` | Common frontend preprocessing before upload or local inference |
| `src/services/offline/` | Local ONNX inference, disease metadata, offline queue |
| `src/services/sync/useAutoSync.ts` | Automatic sync when connectivity returns |
| `src/store/` | Auth, language, and model mode contexts |

## Mobile Interaction Flow

Camera and gallery screens follow the same logic:

1. User captures or selects an image.
2. App runs `preprocessImage` as the common preprocessing step.
3. App checks the selected model mode and network state.
4. If online mode is available, the processed image is uploaded to `/api/predict/`.
5. If offline mode is selected or connectivity is unavailable, the app uses the processed image for local inference.
6. Offline results are added to an AsyncStorage queue.
7. `useAutoSync` sends queued records to `/api/history/sync/` when online access returns.
8. The result screen shows crop, disease, and health status.

## Disease Metadata

The ML model returns a class key such as:

```text
Tomato___Late_blight
```

The backend stores matching disease metadata in `DiseaseInfo`.

The mobile offline path has a local metadata builder in:

```text
mobile-app/src/services/offline/diseaseMetadata.ts
```

This keeps offline result display possible without a backend lookup.

## Model Export

The export flow uses:

```text
src/dissdetector/export/export_mobile_model.py
```

The utility:

- Loads a PyTorch image-only checkpoint.
- Resolves the model name, image size, and class mapping.
- Exports the model to ONNX.
- Writes a manifest with preprocessing details.
- Optionally validates ONNX output against PyTorch output.
- Refuses multimodal model export for offline inference.

Current exported assets:

```text
mobile-app/assets/mobile_models/offline_model.onnx
mobile-app/assets/mobile_models/offline_model_manifest.json
```

## Security Notes

The backend uses:

- JWT authentication through SimpleJWT.
- Email-based custom user accounts.
- Admin-only permission checks for admin panel routes.
- Per-user filtering for prediction history.

Development settings currently include:

- `DEBUG = True`
- SQLite database
- Broad CORS allowance
- Local network hosts in `ALLOWED_HOSTS`

These settings are acceptable for development but should be changed before a
production deployment.
