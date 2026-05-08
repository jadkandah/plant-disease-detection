import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


ROOT = Path("/home/jad/plant-disease-detection")
DATASET_PATH = ROOT / "jordan_dataset"
OUTPUT_CSV = DATASET_PATH / "metadata_weather.csv"

LATITUDE = 31.9539
LONGITUDE = 35.9106
TIMEZONE = "Asia/Amman"

YEAR_START = 2023
YEAR_END = 2025
RANDOM_SEED = 42

FEATURE_COLS = [
    "temp_c",
    "humidity_pct",
    "wind_m_s",
    "precip_mm",
    "soil_moisture_pct",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# This dataset has no per-image capture timestamps, so the metadata generator
# synthesizes plausible datetimes and joins them to historical Open-Meteo values.
SYNTHETIC_METADATA_SOURCE = "synthetic_open_meteo"
SYNTHETIC_METADATA_NOTE = (
    "Synthetic timestamp sampled for each image and joined to historical "
    "Open-Meteo weather for Amman, Jordan."
)

random.seed(RANDOM_SEED)

CLASS_NAME_MAP = {
    "Apple___Apple_scab": ("Apple", "Apple_scab"),
    "Apple___Black_rot": ("Apple", "Black_rot"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar_apple_rust"),
    "Apple___healthy": ("Apple", "healthy"),
    "Corn_(maize)___Common_rust": ("Maize", "Common_rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Maize", "Northern_Leaf_Blight"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Maize",
        "Cercospora_leaf_spot Gray_leaf_spot",
    ),
    "Corn_(maize)___healthy": ("Maize", "healthy"),
    "Grape___Black_rot": ("Grape", "Black_rot"),
    "Grape___Esca_(Black_Measles)": ("Grape", "Esca_(Black_Measles)"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape",
        "Leaf_blight_(Isariopsis_Leaf_Spot)",
    ),
    "Grape___healthy": ("Grape", "healthy"),
    "Orange___Black_spot": ("Orange", "Black_spot"),
    "Orange___Canker": ("Orange", "Canker"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Orange", "Citrus_greening"),
    "Orange___healthy": ("Orange", "healthy"),
    "Peach___Bacterial_spot": ("Peach", "Bacterial_spot"),
    "Peach___healthy": ("Peach", "healthy"),
    "Potato___Early_blight": ("Potato", "Early_blight"),
    "Potato___Late_blight": ("Potato", "Late_blight"),
    "Potato___healthy": ("Potato", "healthy"),
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial_spot"),
    "Tomato___Early_blight": ("Tomato", "Early_blight"),
    "Tomato___Late_blight": ("Tomato", "Late_blight"),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf_Mold"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato___Target_Spot": ("Tomato", "Target_Spot"),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic_virus"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Spider_mites"),
    "Tomato___healthy": ("Tomato", "healthy"),
}

CLASS_MONTHS = {
    "Apple_scab": [3, 4, 5],
    "Black_rot": [5, 6, 7, 8],
    "Cedar_apple_rust": [4, 5, 6],
    "healthy": list(range(1, 13)),
    "Common_rust": [6, 7, 8],
    "Northern_Leaf_Blight": [6, 7, 8],
    "Cercospora_leaf_spot Gray_leaf_spot": [7, 8, 9],
    "Esca_(Black_Measles)": [6, 7, 8],
    "Leaf_blight_(Isariopsis_Leaf_Spot)": [6, 7, 8],
    "Citrus_greening": list(range(1, 10)),
    "Canker": [4, 5, 6, 7],
    "Black_spot": [5, 6, 7, 8],
    "Bacterial_spot": [4, 5, 6, 7],
    "Early_blight": [5, 6, 7, 8],
    "Late_blight": [4, 5, 6],
    "Leaf_Mold": [5, 6, 7],
    "Septoria_leaf_spot": [6, 7, 8],
    "Target_Spot": [6, 7, 8],
    "Mosaic_virus": [5, 6, 7, 8],
    "Yellow_Leaf_Curl_Virus": [5, 6, 7, 8],
    "Spider_mites": [6, 7, 8, 9],
}
DEFAULT_MONTHS = list(range(1, 13))


def list_images(base_dir: Path):
    image_paths: list[str] = []
    for split in ["train", "val", "test"]:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue

        for root, dirs, files in os.walk(split_dir):
            dirs.sort()
            files.sort()
            for filename in files:
                path = Path(root) / filename
                if path.suffix.lower() in IMG_EXTS:
                    image_paths.append(path.relative_to(base_dir).as_posix())

    return image_paths


def parse_path(rel_path: str):
    split, crop, disease, *_ = rel_path.split("/")
    raw_class = f"{crop}___{disease}"
    if raw_class in CLASS_NAME_MAP:
        crop, disease = CLASS_NAME_MAP[raw_class]
    return split, crop, disease


def choose_datetime(disease: str):
    months = CLASS_MONTHS.get(disease, DEFAULT_MONTHS)
    return datetime(
        random.randint(YEAR_START, YEAR_END),
        random.choice(months),
        random.randint(1, 28),
        random.randint(6, 17),
    )


def fetch_weather(day: str):
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": day,
            "end_date": day,
            "hourly": (
                "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                "precipitation,soil_moisture_0_to_7cm"
            ),
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def build_lookup(data):
    hourly = data["hourly"]
    lookup = {}
    for index, timestamp in enumerate(hourly["time"]):
        soil_value = hourly["soil_moisture_0_to_7cm"][index]
        lookup[timestamp] = {
            "temp_c": hourly["temperature_2m"][index],
            "humidity_pct": hourly["relative_humidity_2m"][index],
            "wind_m_s": hourly["wind_speed_10m"][index],
            "precip_mm": hourly["precipitation"][index],
            "soil_moisture_pct": (soil_value or 0) * 100,
        }
    return lookup


def fallback(month: int):
    if month in [12, 1, 2]:
        return dict(temp_c=12, humidity_pct=75, wind_m_s=2.2, precip_mm=0.8, soil_moisture_pct=30)
    if month in [3, 4, 5]:
        return dict(temp_c=21, humidity_pct=60, wind_m_s=2.0, precip_mm=0.2, soil_moisture_pct=22)
    if month in [6, 7, 8]:
        return dict(temp_c=31, humidity_pct=35, wind_m_s=2.5, precip_mm=0.0, soil_moisture_pct=14)
    return dict(temp_c=24, humidity_pct=50, wind_m_s=2.1, precip_mm=0.0, soil_moisture_pct=18)


def generate_metadata(output_csv: Path = OUTPUT_CSV):
    image_paths = list_images(DATASET_PATH)
    print(f"Found {len(image_paths)} images")

    image_metadata = {}
    days = set()

    for rel_path in image_paths:
        _, crop, disease = parse_path(rel_path)
        dt = choose_datetime(disease)
        image_metadata[rel_path] = {
            "crop": crop,
            "disease": disease,
            "datetime": dt,
        }
        days.add(dt.strftime("%Y-%m-%d"))

    weather_cache = {}
    for day in tqdm(sorted(days), desc="Fetching weather"):
        try:
            weather_cache[day] = build_lookup(fetch_weather(day))
        except Exception:
            weather_cache[day] = {}

    rows = []
    for rel_path in tqdm(image_paths, desc="Building metadata rows"):
        info = image_metadata[rel_path]
        dt = info["datetime"]

        timestamp_key = dt.strftime("%Y-%m-%dT%H:00")
        day_key = dt.strftime("%Y-%m-%d")
        weather = weather_cache.get(day_key, {}).get(timestamp_key) or fallback(dt.month)

        rows.append(
            {
                "image_rel_path": rel_path,
                **weather,
                "random_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "crop_name": info["crop"],
                "disease_name": info["disease"],
                "metadata_source": SYNTHETIC_METADATA_SOURCE,
                "metadata_note": SYNTHETIC_METADATA_NOTE,
                "is_synthetic_metadata": True,
            }
        )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote metadata to {output_csv}")


def main():
    generate_metadata()


if __name__ == "__main__":
    main()
