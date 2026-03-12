# Plant Disease Detection Mobile App — AI Agent Build Specification

## 1. Project Summary
Build a **bilingual mobile application** for a graduation project (GP) that allows users to capture or upload a plant image, predict the disease using an AI model, display the diagnosis with confidence, provide treatment guidance, store diagnosis history, and support online/offline behavior.

This product is primarily a **demo-quality GP application** for supervisors/examiners, although the intended end-user persona is **farmers**.

**Important clarification:** this is a **mobile app**, not a web drag-and-drop website.

---

## 2. Product Goals
The app must:
- Upload or capture a plant image and predict disease.
- Show crop name, disease name, confidence, and healthy/diseased status.
- Provide treatment advice.
- Support multiple crops.
- Show simple **weather-based disease risk**.
- Save prediction history per user account.
- Support **Arabic and English** across the full app.
- Support **online/offline sync behavior**.
- Look modern, clean, and farmer-friendly.

This is a **GP demo**, so polish, clarity, and reliability matter more than production-scale complexity.

---

## 3. Target Users
### Primary intended persona
- Farmers

### Actual demo users
- GP supervisors / examiners
- Student developers / admins

---

## 4. Core Product Scope
### In scope
- Authentication
- Camera capture
- Gallery upload
- AI classification flow
- Diagnosis result view
- Treatment and disease info
- Prediction history
- Offline local save + online auto-sync
- Profile and settings
- Admin dashboard for developers
- Bilingual Arabic/English support
- Weather risk indicator

### Out of scope for now
- Disease-region segmentation
- Chatbot treatment assistant (future feature only)
- Full offline on-device inference implementation
- Drag-and-drop desktop UI
- Multi-image prediction in one request

---

## 5. Technical Stack
### Frontend
- **React Native** mobile app
- Use the provided sample screens as **inspiration only**, not as strict copies
- Light mode first
- Green + white primary visual style

### Backend
- **Django backend**
- Prefer **Django REST Framework** for APIs
- **JWT authentication**
- **SQLite** database for the GP project

### AI Integration
- Online mode: call backend model inference endpoint
- Offline mode: **prepare app structure only** and clearly mark offline local inference as **future work / placeholder**

### Localization
- Full Arabic and English translation support
- UI text, disease labels, and treatment content must all be translatable

---

## 6. Design Direction
### Style
- Modern
- Farmer-friendly
- Clean and simple
- Mobile-first
- Light mode
- Green + white color palette

### Design guidance
- Use the provided mockups/screens as inspiration for layout flow and visual tone
- Do **not** copy them exactly
- Improve spacing, readability, and navigation consistency
- Focus on clear, presentation-ready UI for GP demonstration

---

## 7. Main User Flows
### Flow A — Sign up and login
1. User opens app.
2. User sees login screen.
3. User can navigate to sign up.
4. User creates account with full name, email, password, and phone number.
5. User logs in using email + password.
6. User lands on home dashboard.

### Flow B — Diagnose from camera
1. User opens diagnosis flow from home dashboard.
2. User chooses camera.
3. App opens camera screen with guide overlay.
4. User captures one image.
5. App sends image for prediction when online.
6. App shows result screen.
7. Result is saved to local storage and synced to backend.

### Flow C — Diagnose from gallery
1. User opens diagnosis flow.
2. User chooses gallery.
3. User selects one image.
4. App uploads image for prediction.
5. App shows result screen.
6. Result is stored in history.

### Flow D — Offline usage
1. User captures/selects image while offline.
2. If offline inference is not available yet, app should handle this gracefully.
3. App stores pending record metadata locally when appropriate.
4. App shows sync state.
5. When internet returns, app auto-syncs pending records.

### Flow E — Review history
1. User opens history.
2. User can search records.
3. User can filter by crop and date.
4. User opens a historical diagnosis.
5. User sees full result details.

### Flow F — Weather risk
1. App gets phone location automatically (with permission).
2. App fetches current weather data from backend or weather service.
3. App shows simple disease risk level: **Low / Medium / High**.
4. Weather risk feature should be built in a modular way because it is not 100% final yet.

---

## 8. Required Screens
Build the following screens:

1. **Login Screen**
2. **Sign Up Screen**
3. **Home Dashboard Screen**
4. **Camera Screen**
5. **Gallery / Select Photo Screen**
6. **Diagnosis Result Screen**
7. **Diagnosis History Screen**
8. **Settings Screen**
9. **Profile Screen**
10. **Supported Crops & Diseases Screen**
11. **Admin Dashboard Screen**

### Optional / conditional screen
12. **Weather Risk Screen** (optional module; structure should support it even if not fully enabled)

---

## 9. Screen-by-Screen Requirements

### 9.1 Login Screen
Purpose: authenticate existing users.

Fields:
- Email
- Password

Actions:
- Login
- Navigate to Sign Up

Requirements:
- Input validation
- Clear error states
- JWT token handling
- Arabic/English switch support

---

### 9.2 Sign Up Screen
Fields:
- Full name
- Email
- Password
- Phone number

Actions:
- Create account
- Navigate to Login

Requirements:
- Validation for email/password/phone
- Friendly success and error messages
- Bilingual support

---

### 9.3 Home Dashboard Screen
Purpose: central navigation and app summary.

Suggested sections:
- Welcome message with user name
- Quick actions:
  - Capture image
  - Upload from gallery
  - View history
  - View supported crops
- Small weather risk card
- Recent diagnosis preview
- Sync status indicator

Requirements:
- Clean dashboard layout
- Fast navigation to core tasks
- Mobile-first usability

---

### 9.4 Camera Screen
Requirements:
- One image at a time
- No cropping required
- Camera preview with framing guide overlay
- Capture button
- Back navigation
- Clear instruction text (for lighting and alignment)

Notes:
- Use camera screen inspiration from sample UI
- Keep UI minimal and clean

---

### 9.5 Gallery Screen
Requirements:
- Select one image from device gallery
- Preview selected image before submit if needed
- No multi-select
- No drag-and-drop

---

### 9.6 Diagnosis Result Screen
Must display:
- Uploaded image preview
- Crop name
- Disease name in **Arabic and English**
- Confidence percentage
- Confidence bar
- Healthy / Diseased badge
- Treatment advice in bullet points
- Disease description
- Causes

Nice-to-have structure:
- Section 1: Image and prediction summary
- Section 2: Disease information
- Section 3: Treatment advice
- Section 4: Weather risk indicator
- Section 5: Save/sync status

Behavior:
- Save result to history
- Allow opening same result later from history
- Clearly show whether record is synced or pending sync

---

### 9.7 Diagnosis History Screen
Requirements:
- Display all past predictions for current user
- Search by disease/crop
- Filter by crop
- Filter by date
- Open full old result
- Show sync state
- Automatically sync offline records when internet is available

Each history card should show:
- Thumbnail image
- Date
- Crop
- Disease
- Confidence
- Sync state

---

### 9.8 Settings Screen
Include:
- Language switcher (Arabic / English)
- Logout
- Account info entry point
- Change password

Optional additions:
- Sync info/status
- Help/About
- Privacy policy

---

### 9.9 Profile Screen
Include:
- Full name
- Email
- Phone number
- Account metadata if useful
- Edit profile action (optional)

---

### 9.10 Supported Crops & Diseases Screen
Purpose: clearly demonstrate model coverage.

Requirements:
- Show supported crops
- Show diseases under each crop
- Use expandable list/cards
- Bilingual labels where possible
- Mark healthy classes clearly

This should be visible in the app and not hidden in backend only.

---

### 9.11 Admin Dashboard Screen
Audience: student developers/admins only.

Admin features:
- View all user records
- View diagnosis counts
- View sync statistics if possible
- Manage disease information content
- Manage prediction records

Important:
- Admin can still use all normal farmer/user features
- Admin dashboard should be clearly separated from normal user flow

---

## 10. AI Prediction Requirements
### Model behavior
Current supported feature:
- **Classification only**

Not included now:
- Segmentation
- Detection boxes
- Chatbot diagnosis assistant

### Inference modes
#### Online mode
- Send selected image to Django backend API
- Receive crop, disease, confidence, and info payload

#### Offline mode
- Do **not** implement full on-device inference now unless a ready mobile model exists
- Build the app architecture so offline inference can be added later
- Clearly document this as future work

### Response payload should support
- Crop name
- Disease label (internal)
- Disease English name
- Disease Arabic name
- Confidence
- Healthy/diseased status
- Disease description
- Causes
- Treatment advice
- Weather risk if available

---

## 11. Weather Risk Module
This module is desired but still somewhat tentative, so implement it in a modular way.

### Requirements
- App requests phone location automatically with permission
- Backend calculates or fetches a simple disease-weather risk indicator
- Output is only:
  - Low
  - Medium
  - High

### UI usage
- Can appear as a compact card on home screen
- Can also appear inside result screen for the predicted crop/disease

### Notes
- Keep implementation simple and demo-friendly
- No need for complex forecasting dashboards unless time allows

---

## 12. Authentication Requirements
### Account fields
- Full name
- Email
- Password
- Phone number

### Login method
- Email + password only

### Auth implementation
- JWT authentication
- Store access/refresh tokens securely
- Handle expired sessions cleanly

---

## 13. History and Sync Requirements
The app must support mixed online/offline behavior.

### Offline/local behavior
- Save diagnosis history locally when needed
- Mark records as pending sync if not yet uploaded to backend

### Online behavior
- Auto-sync offline records when internet returns
- Avoid duplicate uploads
- Preserve image/date/prediction/confidence/user linkage

### Stored fields per record
- Image
- Date/time
- Prediction
- Confidence
- User name / user reference
- Crop
- Sync status

### Suggested mobile storage
- AsyncStorage or local database solution such as SQLite on device

---

## 14. Treatment Content Strategy
For now, treatment content should be:
- **Hardcoded static content**
- Stored in backend database and/or JSON seed data

Future upgrade:
- Chatbot-generated treatment guidance

### Requirement
The system must be designed so static disease info can later be replaced or extended by a chatbot module.

---

## 15. Localization Requirements
The app must support full bilingual behavior.

### Languages
- Arabic
- English

### Everything to localize
- All screen labels
- Buttons
- Errors and alerts
- Disease names
- Treatment advice
- Causes
- Descriptions
- Navigation text
- Settings text

### UX notes
- Support RTL layout for Arabic where appropriate
- Ensure clean typography and spacing in both languages

---

## 16. Data Models (Suggested)

### User
- id
- full_name
- email
- password_hash
- phone_number
- is_admin
- created_at

### DiseaseInfo
- id
- crop_name_en
- crop_name_ar
- disease_name_en
- disease_name_ar
- class_key
- health_status
- description_en
- description_ar
- causes_en
- causes_ar
- treatment_en
- treatment_ar

### PredictionRecord
- id
- user_id
- image_url or image_path
- crop_name
- disease_name_en
- disease_name_ar
- confidence
- is_healthy
- predicted_at
- location_lat (optional)
- location_lng (optional)
- weather_risk_level (optional)
- sync_status
- source_type (camera/gallery)

---

## 17. API Contract (Suggested)
Use Django REST Framework.

### Auth
- `POST /api/auth/register/`
- `POST /api/auth/login/`
- `POST /api/auth/refresh/`
- `POST /api/auth/change-password/`
- `GET /api/auth/profile/`

### Prediction
- `POST /api/predict/`
  - Input: image file, language, optional location
  - Output: crop, disease, confidence, status, info, weather risk

### History
- `GET /api/history/`
- `GET /api/history/{id}/`
- `POST /api/history/sync/`

### Disease Info
- `GET /api/diseases/`
- `GET /api/diseases/{id}/`

### Weather
- `GET /api/weather-risk/?lat=...&lng=...&crop=...`

### Admin
- `GET /api/admin/stats/`
- `GET /api/admin/records/`
- `PUT /api/admin/disease-info/{id}/`

---

## 18. Supported Classes
The app and backend should support the following current classes.

```python
{
    "Healthy Wheat": ("Wheat", "healthy"),
    "Wheat aphid": ("Wheat", "Aphid"),
    "Wheat black rust": ("Wheat", "Black_rust"),
    "Wheat Brown leaf Rust": ("Wheat", "Brown_leaf_Rust"),
    "Wheat leaf blight": ("Wheat", "Leaf_blight"),
    "Wheat mite": ("Wheat", "Mite"),
    "Wheat powdery mildew": ("Wheat", "Powdery_mildew"),
    "Wheat scab": ("Wheat", "Scab"),
    "Wheat Stem fly": ("Wheat", "Stem_fly"),
    "Wheat___Yellow_Rust": ("Wheat", "Yellow_Rust"),

    "Cauliflower_Bacterial_spot_rot": ("Cauliflower", "Bacterial_spot_rot"),
    "Cauliflower_Black_Rot": ("Cauliflower", "Black_Rot"),
    "Cauliflower_Downy_Mildew": ("Cauliflower", "Downy_Mildew"),
    "Cauliflower_Healthy": ("Cauliflower", "healthy"),
    "EggPlant_Healthy_Leaf": ("Eggplant", "healthy"),
    "EggPlant_Insect_Pest_Disease": ("Eggplant", "Insect_Pest_Disease"),
    "EggPlant_Leaf_Spot_Disease": ("Eggplant", "Leaf_Spot_Disease"),
    "EggPlant_Mosaic_Virus_Disease": ("Eggplant", "Mosaic_Virus_Disease"),
    "EggPlant_Small_Leaf_Disease": ("Eggplant", "Small_Leaf_Disease"),
    "EggPlant_White_Mold_Disease": ("Eggplant", "White_Mold_Disease"),
    "EggPlant_Wilt_Disease": ("Eggplant", "Wilt_Disease"),

    "Apple___Apple_scab": ("Apple", "Apple_scab"),
    "Apple___Black_rot": ("Apple", "Black_rot"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar_apple_rust"),
    "Apple___healthy": ("Apple", "healthy"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Maize", "Cercospora_leaf_spot_Gray_leaf_spot"),
    "Corn_(maize)___Common_rust_": ("Maize", "Common_rust"),
    "Corn_(maize)___healthy": ("Maize", "healthy"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Maize", "Northern_Leaf_Blight"),
    "Grape___Black_rot": ("Grape", "Black_rot"),
    "Grape___Esca_(Black_Measles)": ("Grape", "Esca_Black_Measles"),
    "Grape___healthy": ("Grape", "healthy"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Grape", "Leaf_blight_Isariopsis_Leaf_Spot"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Orange", "Citrus_greening"),
    "Peach___Bacterial_spot": ("Peach", "Bacterial_spot"),
    "Peach___healthy": ("Peach", "healthy"),
    "Potato___Early_blight": ("Potato", "Early_blight"),
    "Potato___healthy": ("Potato", "healthy"),
    "Potato___Late_blight": ("Potato", "Late_blight"),
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial_spot"),
    "Tomato___Early_blight": ("Tomato", "Early_blight"),
    "Tomato___healthy": ("Tomato", "healthy"),
    "Tomato___Late_blight": ("Tomato", "Late_blight"),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf_Mold"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Spider_mites"),
    "Tomato___Target_Spot": ("Tomato", "Target_Spot"),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic_virus"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow_Leaf_Curl_Virus"),

    "aculus_olearius": ("Olive", "Aculus_olearius_mite"),
    "Healthy": ("Olive", "healthy"),
    "olive_peacock_spot": ("Olive", "Peacock_spot"),
}
```

### Implementation note
Normalize these into cleaner display labels for the UI while keeping original class keys for model/backend compatibility.

---

## 19. UX and Product Priorities
The most important priorities for this GP build are:
1. **Offline/online syncing**
2. **History and authentication**
3. **Weather integration**
4. Clean AI prediction flow presentation

### Clarification on “showing AI prediction flow”
This means the app should make it obvious to a GP examiner how the AI works in practice:
- user chooses image
- app processes image
- result appears clearly
- confidence and disease information are visible
- record is saved in history

---

## 20. Error Handling
The app must handle:
- Invalid image upload
- Camera/gallery permission denied
- No internet connection
- Prediction request failure
- Sync failure
- Authentication failure
- Missing location permission
- Empty history state

Error messages must be available in Arabic and English.

---

## 21. Non-Functional Requirements
- Mobile-first performance
- Clean demo-ready UI
- Clear navigation
- Consistent bilingual experience
- Modular architecture for future chatbot and offline inference
- Secure JWT handling
- Graceful offline behavior

---

## 22. Suggested Folder Structure

### React Native app
```text
mobile-app/
  src/
    api/
    assets/
    components/
    constants/
    hooks/
    i18n/
    navigation/
    screens/
      Auth/
      Home/
      Camera/
      Gallery/
      Result/
      History/
      Settings/
      Profile/
      SupportedCrops/
      Admin/
    services/
      auth/
      prediction/
      sync/
      weather/
      storage/
    store/
    types/
    utils/
```

### Django backend
```text
backend/
  config/
  apps/
    authentication/
    prediction/
    diseases/
    history/
    weather/
    adminpanel/
  media/
  static/
  requirements.txt
```

---

## 23. Build Instructions for the AI Agent
The AI agent should:
1. Build a **React Native mobile app** matching the requirements above.
2. Build a **Django REST backend** with JWT auth and SQLite.
3. Create clean, bilingual screens in Arabic and English.
4. Use the shared UI images as **design inspiration only**.
5. Implement authentication, history, profile, settings, and admin flow.
6. Implement online image prediction integration through backend API.
7. Implement local storage + auto-sync architecture.
8. Add a modular weather risk feature using automatic phone location.
9. Keep offline on-device inference as a documented future-ready placeholder.
10. Seed disease/treatment data using static database/JSON content.
11. Ensure the result screen is rich and presentation-friendly for GP demo.
12. Keep the code modular, readable, and easy to present academically.

---

## 24. Acceptance Criteria
The build is successful if:
- User can register and login.
- User can capture or upload one plant image.
- App returns a diagnosis with crop, disease, confidence, and treatment.
- App supports Arabic and English.
- App saves prediction history per account.
- App can search/filter history by crop and date.
- App can show supported crops/diseases.
- App has a developer/admin dashboard.
- App stores offline/pending records and syncs them automatically later.
- App includes weather risk as a simple Low/Medium/High indicator or at least a modular placeholder.
- UI looks polished enough for GP presentation.

---

## 25. Future Enhancements
- On-device offline model inference
- Chatbot for treatment advice
- Segmentation / infected-area highlighting
- Push notifications
- More advanced weather forecasting
- Expert consultation flow
- Better admin analytics

