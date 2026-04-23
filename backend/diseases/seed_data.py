import os
import django
import sys

# Setup django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from diseases.models import DiseaseInfo

# ──────────────────────────────────────────────
# Class keys MUST match the model's CLASS_NAMES exactly.
# These use the Parent___Leaf format from the dataset folders.
# The model was trained on 45 classes (no Eggplant).
# ──────────────────────────────────────────────
CLASSES = {
    # Apple (4)
    "Apple___Apple_scab": ("Apple", "Apple Scab"),
    "Apple___Black_rot": ("Apple", "Black Rot"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar Apple Rust"),
    "Apple___healthy": ("Apple", "Healthy"),

    # Cauliflower (4)
    "Cauliflower___Bacterial_spot_rot": ("Cauliflower", "Bacterial Spot Rot"),
    "Cauliflower___Black_Rot": ("Cauliflower", "Black Rot"),
    "Cauliflower___Downy_Mildew": ("Cauliflower", "Downy Mildew"),
    "Cauliflower___healthy": ("Cauliflower", "Healthy"),

    # Grape (4)
    "Grape___Black_rot": ("Grape", "Black Rot"),
    "Grape___Esca_Black_Measles": ("Grape", "Esca (Black Measles)"),
    "Grape___Leaf_blight_Isariopsis_Leaf_Spot": ("Grape", "Leaf Blight (Isariopsis)"),
    "Grape___healthy": ("Grape", "Healthy"),

    # Maize (4)
    "Maize___Cercospora_leaf_spot_Gray_leaf_spot": ("Maize", "Cercospora / Gray Leaf Spot"),
    "Maize___Common_rust": ("Maize", "Common Rust"),
    "Maize___Northern_Leaf_Blight": ("Maize", "Northern Leaf Blight"),
    "Maize___healthy": ("Maize", "Healthy"),

    # Olive (3)
    "Olive___Aculus_olearius_mite": ("Olive", "Aculus Olearius Mite"),
    "Olive___Peacock_spot": ("Olive", "Peacock Spot"),
    "Olive___healthy": ("Olive", "Healthy"),

    # Orange (1)
    "Orange___Citrus_greening": ("Orange", "Citrus Greening (Huanglongbing)"),

    # Peach (2)
    "Peach___Bacterial_spot": ("Peach", "Bacterial Spot"),
    "Peach___healthy": ("Peach", "Healthy"),

    # Potato (3)
    "Potato___Early_blight": ("Potato", "Early Blight"),
    "Potato___Late_blight": ("Potato", "Late Blight"),
    "Potato___healthy": ("Potato", "Healthy"),

    # Tomato (10)
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial Spot"),
    "Tomato___Early_blight": ("Tomato", "Early Blight"),
    "Tomato___Late_blight": ("Tomato", "Late Blight"),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf Mold"),
    "Tomato___Mosaic_virus": ("Tomato", "Mosaic Virus"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria Leaf Spot"),
    "Tomato___Spider_mites": ("Tomato", "Spider Mites"),
    "Tomato___Target_Spot": ("Tomato", "Target Spot"),
    "Tomato___Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow Leaf Curl Virus"),
    "Tomato___healthy": ("Tomato", "Healthy"),

    # Wheat (10)
    "Wheat___Aphid": ("Wheat", "Aphid"),
    "Wheat___Black_rust": ("Wheat", "Black Rust"),
    "Wheat___Brown_leaf_Rust": ("Wheat", "Brown Leaf Rust"),
    "Wheat___Leaf_blight": ("Wheat", "Leaf Blight"),
    "Wheat___Mite": ("Wheat", "Mite"),
    "Wheat___Powdery_mildew": ("Wheat", "Powdery Mildew"),
    "Wheat___Scab": ("Wheat", "Scab"),
    "Wheat___Stem_fly": ("Wheat", "Stem Fly"),
    "Wheat___Yellow_Rust": ("Wheat", "Yellow Rust"),
    "Wheat___healthy": ("Wheat", "Healthy"),

    # Eggplant (7)
    "Eggplant___healthy": ("Eggplant", "Healthy"),
    "Eggplant___Insect_Pest_Disease": ("Eggplant", "Insect Pest Disease"),
    "Eggplant___Leaf_Spot_Disease": ("Eggplant", "Leaf Spot Disease"),
    "Eggplant___Mosaic_Virus_Disease": ("Eggplant", "Mosaic Virus Disease"),
    "Eggplant___Small_Leaf_Disease": ("Eggplant", "Small Leaf Disease"),
    "Eggplant___White_Mold_Disease": ("Eggplant", "White Mold Disease"),
    "Eggplant___Wilt_Disease": ("Eggplant", "Wilt Disease"),

    # Orange extended (3 additional)
    "Orange___healthy": ("Orange", "Healthy"),
    "Orange___Black_spot": ("Orange", "Black Spot"),
    "Orange___Canker": ("Orange", "Canker"),
}

# Arabic crop name mapping
CROP_NAME_AR = {
    "Apple": "تفاح",
    "Cauliflower": "قرنبيط",
    "Eggplant": "باذنجان",
    "Grape": "عنب",
    "Maize": "ذرة",
    "Olive": "زيتون",
    "Orange": "برتقال",
    "Peach": "خوخ",
    "Potato": "بطاطا",
    "Tomato": "طماطم",
    "Wheat": "قمح",
}

# Arabic disease name mapping
DISEASE_NAME_AR = {
    "Apple Scab": "جرب التفاح",
    "Black Rot": "العفن الأسود",
    "Cedar Apple Rust": "صدأ أرز التفاح",
    "Healthy": "سليم",
    "Bacterial Spot Rot": "تعفن البقعة البكتيرية",
    "Downy Mildew": "البياض الزغبي",
    "Esca (Black Measles)": "إسكا (الحصبة السوداء)",
    "Leaf Blight (Isariopsis)": "لفحة الأوراق",
    "Cercospora / Gray Leaf Spot": "بقعة الأوراق الرمادية",
    "Common Rust": "الصدأ الشائع",
    "Northern Leaf Blight": "لفحة الأوراق الشمالية",
    "Aculus Olearius Mite": "عث أكولوس الزيتون",
    "Peacock Spot": "بقعة عين الطاووس",
    "Citrus Greening (Huanglongbing)": "تخضير الحمضيات",
    "Bacterial Spot": "البقعة البكتيرية",
    "Early Blight": "اللفحة المبكرة",
    "Late Blight": "اللفحة المتأخرة",
    "Leaf Mold": "عفن الأوراق",
    "Mosaic Virus": "فيروس الفسيفساء",
    "Septoria Leaf Spot": "بقعة سبتوريا",
    "Spider Mites": "العنكبوت الأحمر",
    "Target Spot": "البقعة المستهدفة",
    "Yellow Leaf Curl Virus": "فيروس تجعد الأوراق الأصفر",
    "Aphid": "المن",
    "Black Rust": "الصدأ الأسود",
    "Brown Leaf Rust": "صدأ الأوراق البني",
    "Leaf Blight": "لفحة الأوراق",
    "Mite": "العث",
    "Powdery Mildew": "البياض الدقيقي",
    "Scab": "الجرب",
    "Stem Fly": "ذبابة الساق",
    "Yellow Rust": "الصدأ الأصفر",
    # Eggplant diseases
    "Insect Pest Disease": "مرض الآفات الحشرية",
    "Leaf Spot Disease": "مرض تبقع الأوراق",
    "Mosaic Virus Disease": "مرض فيروس الفسيفساء",
    "Small Leaf Disease": "مرض صغر الأوراق",
    "White Mold Disease": "مرض العفن الأبيض",
    "Wilt Disease": "مرض الذبول",
    # Orange extra diseases
    "Black Spot": "البقعة السوداء",
    "Canker": "التقرح",
}

def seed_data():
    create_count = 0
    update_count = 0
    
    for class_key, (crop, disease) in CLASSES.items():
        is_healthy = disease.lower() == 'healthy'
        health_status = 'healthy' if is_healthy else 'diseased'
        
        crop_ar = CROP_NAME_AR.get(crop, crop)
        disease_ar = DISEASE_NAME_AR.get(disease, disease)

        # Descriptions
        if is_healthy:
            desc_en = f"The {crop} crop appears perfectly healthy with no visible signs of disease."
            desc_ar = f"يبدو محصول {crop_ar} سليماً تماماً بدون أي علامات مرئية للمرض."
            causes_en = "Optimal growth conditions and good agricultural practices."
            causes_ar = "ظروف نمو مثالية وممارسات زراعية جيدة."
            treatment_en = "• Continue current care routine\n• Regular monitoring recommended"
            treatment_ar = "• استمر في روتين الرعاية الحالي\n• يُنصح بالمراقبة المنتظمة"
        else:
            desc_en = f"{disease} detected on {crop}. This condition may affect crop yield if left untreated."
            desc_ar = f"تم اكتشاف {disease_ar} على {crop_ar}. قد تؤثر هذه الحالة على المحصول إذا لم تُعالج."
            causes_en = f"Common causes include environmental stress, pests, or pathogens."
            causes_ar = f"تشمل الأسباب الشائعة الإجهاد البيئي أو الآفات أو مسببات الأمراض."
            treatment_en = "• Remove affected leaves and dispose properly\n• Apply appropriate fungicide or pesticide\n• Ensure proper spacing for air circulation\n• Monitor regularly for recurrence"
            treatment_ar = "• أزل الأوراق المصابة وتخلص منها بشكل صحيح\n• استخدم المبيد الفطري أو الحشري المناسب\n• تأكد من التباعد المناسب لتهوية الهواء\n• راقب بانتظام لمنع تكرار الإصابة"

        obj, created = DiseaseInfo.objects.update_or_create(
            class_key=class_key,
            defaults={
                'crop_name_en': crop,
                'crop_name_ar': crop_ar,
                'disease_name_en': disease,
                'disease_name_ar': disease_ar,
                'health_status': health_status,
                'description_en': desc_en,
                'description_ar': desc_ar,
                'causes_en': causes_en,
                'causes_ar': causes_ar,
                'treatment_en': treatment_en,
                'treatment_ar': treatment_ar,
            }
        )
        if created:
            create_count += 1
        else:
            update_count += 1
            
    # Delete any old class_keys that are no longer in the model
    valid_keys = set(CLASSES.keys())
    deleted_count = DiseaseInfo.objects.exclude(class_key__in=valid_keys).delete()[0]
    
    print(f"Seeded {create_count} new, updated {update_count}, deleted {deleted_count} old DiseaseInfo entries.")
    print(f"Total: {DiseaseInfo.objects.count()} classes in database.")

if __name__ == "__main__":
    seed_data()
