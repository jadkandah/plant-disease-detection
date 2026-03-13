import os
import random
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from tqdm import tqdm

# =========================
# Configuration
# =========================
ROOT = Path("/Users/sanadmadani/plant-disease-detection/plant-disease-detection")
DATASET_PATH = ROOT / "jordan_dataset2"
OUTPUT_CSV = DATASET_PATH / "metadata_weather.csv"

LATITUDE = 31.9539
LONGITUDE = 35.9106
TIMEZONE = "Asia/Amman"

YEAR_START = 2023
YEAR_END = 2025

FEATURE_COLS = [
    "temp_c",
    "humidity_pct",
    "wind_m_s",
    "precip_mm",
    "soil_moisture_pct",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# Map your raw class names
# =========================
CLASS_NAME_MAP = {
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

# =========================
# Disease -> likely months
# These only guide date choice.
# Weather values still come from API.
# =========================
PROFILE_MONTHS = {
    "healthy": [3, 4, 5],
    "rust": [2, 3, 4, 11, 12],
    "blight": [2, 3, 4, 10, 11, 12],
    "mildew": [2, 3, 4, 10, 11],
    "leaf_spot": [2, 3, 4, 10, 11],
    "spot": [2, 3, 4, 10, 11],
    "rot": [1, 2, 3, 11, 12],
    "mite": [5, 6, 7, 8, 9],
    "aphid": [3, 4, 5],
    "stem_fly": [4, 5, 6],
    "virus": [4, 5, 6, 7, 8],
    "wilt": [5, 6, 7, 8],
    "scab": [2, 3, 4],
    "default": [3, 4, 5, 10],
}

# =========================
# Helpers
# =========================
def list_images(base_dir: Path):
    image_paths = []
    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            continue

        for root, _, files in os.walk(split_dir):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in IMG_EXTS:
                    rel_path = fpath.relative_to(base_dir).as_posix()
                    image_paths.append(rel_path)

    return sorted(image_paths)


def normalize_label_from_rel_path(rel_path: str):
    parts = rel_path.split("/")
    if len(parts) < 4:
        return None, None, None

    split_name, parent, leaf = parts[0], parts[1], parts[2]

    if leaf in CLASS_NAME_MAP:
        crop, disease = CLASS_NAME_MAP[leaf]
        return split_name, crop, disease

    if parent in CLASS_NAME_MAP:
        crop, disease = CLASS_NAME_MAP[parent]
        return split_name, crop, disease

    if "___" in leaf:
        crop, disease = leaf.split("___", 1)
        return split_name, crop, disease

    return split_name, parent, leaf


def disease_to_months(disease_name: str):
    d = disease_name.lower().replace("-", "_").replace(" ", "_")

    if "healthy" in d:
        return PROFILE_MONTHS["healthy"]
    if "rust" in d:
        return PROFILE_MONTHS["rust"]
    if "blight" in d:
        return PROFILE_MONTHS["blight"]
    if "mildew" in d or "mold" in d:
        return PROFILE_MONTHS["mildew"]
    if "leaf_spot" in d:
        return PROFILE_MONTHS["leaf_spot"]
    if "spot" in d:
        return PROFILE_MONTHS["spot"]
    if "rot" in d:
        return PROFILE_MONTHS["rot"]
    if "mite" in d:
        return PROFILE_MONTHS["mite"]
    if "aphid" in d:
        return PROFILE_MONTHS["aphid"]
    if "stem_fly" in d or "fly" in d:
        return PROFILE_MONTHS["stem_fly"]
    if "virus" in d:
        return PROFILE_MONTHS["virus"]
    if "wilt" in d:
        return PROFILE_MONTHS["wilt"]
    if "scab" in d:
        return PROFILE_MONTHS["scab"]

    return PROFILE_MONTHS["default"]


def choose_realistic_datetime(disease_name: str):
    months = disease_to_months(disease_name)
    year = random.randint(YEAR_START, YEAR_END)
    month = random.choice(months)
    day = random.randint(1, 28)
    hour = random.randint(6, 17)   # daytime images usually make more sense
    return datetime(year, month, day, hour, 0, 0)


def fetch_hourly_weather_for_day(lat, lon, day_str, timezone):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day_str,
        "end_date": day_str,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,soil_moisture_0_to_7cm",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": timezone,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def build_hourly_lookup(api_json):
    hourly = api_json.get("hourly", {})
    times = hourly.get("time", [])
    temp = hourly.get("temperature_2m", [])
    humidity = hourly.get("relative_humidity_2m", [])
    wind = hourly.get("wind_speed_10m", [])
    precip = hourly.get("precipitation", [])
    soil = hourly.get("soil_moisture_0_to_7cm", [])

    lookup = {}
    for i, t in enumerate(times):
        lookup[t] = {
            "temp_c": temp[i] if i < len(temp) else None,
            "humidity_pct": humidity[i] if i < len(humidity) else None,
            "wind_m_s": wind[i] if i < len(wind) else None,
            "precip_mm": precip[i] if i < len(precip) else None,
            "soil_moisture_pct": soil[i] * 100.0 if i < len(soil) and soil[i] is not None else None,
        }
    return lookup


def fallback_weather_from_month(month):
    # only used if API fails for a row
    if month in [12, 1, 2]:
        return {"temp_c": 12, "humidity_pct": 75, "wind_m_s": 2.2, "precip_mm": 0.8, "soil_moisture_pct": 30}
    if month in [3, 4, 5]:
        return {"temp_c": 21, "humidity_pct": 60, "wind_m_s": 2.0, "precip_mm": 0.2, "soil_moisture_pct": 22}
    if month in [6, 7, 8]:
        return {"temp_c": 31, "humidity_pct": 35, "wind_m_s": 2.5, "precip_mm": 0.0, "soil_moisture_pct": 14}
    return {"temp_c": 24, "humidity_pct": 50, "wind_m_s": 2.1, "precip_mm": 0.0, "soil_moisture_pct": 18}


# =========================
# Main
# =========================
def generate_metadata():
    if not DATASET_PATH.is_dir():
        raise RuntimeError(f"Dataset path not found: {DATASET_PATH}")

    image_rel_paths = list_images(DATASET_PATH)
    if not image_rel_paths:
        raise RuntimeError(f"No images found under: {DATASET_PATH}")

    print(f"Found {len(image_rel_paths)} images.")

    image_datetimes = {}
    unique_days = set()

    for rel_path in image_rel_paths:
        _, crop, disease = normalize_label_from_rel_path(rel_path)
        disease = disease if disease is not None else "default"
        dt = choose_realistic_datetime(disease)
        image_datetimes[rel_path] = {
            "crop": crop,
            "disease": disease,
            "dt": dt,
        }
        unique_days.add(dt.strftime("%Y-%m-%d"))

    print(f"Need API calls for {len(unique_days)} unique days.")

    daily_cache = {}
    for day_str in tqdm(sorted(unique_days), desc="Fetching Open-Meteo weather"):
        try:
            api_json = fetch_hourly_weather_for_day(LATITUDE, LONGITUDE, day_str, TIMEZONE)
            daily_cache[day_str] = build_hourly_lookup(api_json)
        except Exception as e:
            print(f"Failed API fetch for {day_str}: {e}")
            daily_cache[day_str] = {}

    rows = []
    api_missing = 0

    for rel_path in tqdm(image_rel_paths, desc="Building CSV"):
        info = image_datetimes[rel_path]
        dt = info["dt"]
        day_str = dt.strftime("%Y-%m-%d")
        hour_key = dt.strftime("%Y-%m-%dT%H:00")

        weather = daily_cache.get(day_str, {}).get(hour_key)
        if weather is None:
            api_missing += 1
            weather = fallback_weather_from_month(dt.month)

        rows.append({
            "image_rel_path": rel_path,
            "temp_c": weather["temp_c"],
            "humidity_pct": weather["humidity_pct"],
            "wind_m_s": weather["wind_m_s"],
            "precip_mm": weather["precip_mm"],
            "soil_moisture_pct": weather["soil_moisture_pct"],
            "random_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "crop_name": info["crop"],
            "disease_name": info["disease"],
        })

    df = pd.DataFrame(rows)
    df = df[
        ["image_rel_path"] + FEATURE_COLS + ["random_datetime", "crop_name", "disease_name"]
    ]

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    print(f"Rows where exact API hour was missing and fallback was used: {api_missing}")
    print(df.head())


if __name__ == "__main__":
    generate_metadata()