const CROP_NAME_AR: Record<string, string> = {
  Apple: 'تفاح',
  Cauliflower: 'قرنبيط',
  Eggplant: 'باذنجان',
  Grape: 'عنب',
  Maize: 'ذرة',
  Olive: 'زيتون',
  Orange: 'برتقال',
  Peach: 'خوخ',
  Potato: 'بطاطا',
  Tomato: 'طماطم',
  Wheat: 'قمح',
};

const DISEASE_NAME_EN: Record<string, string> = {
  Apple_scab: 'Apple Scab',
  Black_rot: 'Black Rot',
  Cedar_apple_rust: 'Cedar Apple Rust',
  healthy: 'Healthy',
  Bacterial_spot_rot: 'Bacterial Spot Rot',
  Black_Rot: 'Black Rot',
  Downy_Mildew: 'Downy Mildew',
  Insect_Pest_Disease: 'Insect Pest Disease',
  Leaf_Spot_Disease: 'Leaf Spot Disease',
  Mosaic_Virus_Disease: 'Mosaic Virus Disease',
  Small_Leaf_Disease: 'Small Leaf Disease',
  White_Mold_Disease: 'White Mold Disease',
  Wilt_Disease: 'Wilt Disease',
  Esca_Black_Measles: 'Esca (Black Measles)',
  Leaf_blight_Isariopsis_Leaf_Spot: 'Leaf Blight (Isariopsis)',
  Cercospora_leaf_spot_Gray_leaf_spot: 'Cercospora / Gray Leaf Spot',
  Common_rust: 'Common Rust',
  Northern_Leaf_Blight: 'Northern Leaf Blight',
  Aculus_olearius_mite: 'Aculus Olearius Mite',
  Peacock_spot: 'Peacock Spot',
  Black_spot: 'Black Spot',
  Canker: 'Canker',
  Citrus_greening: 'Citrus Greening (Huanglongbing)',
  Bacterial_spot: 'Bacterial Spot',
  Early_blight: 'Early Blight',
  Late_blight: 'Late Blight',
  Leaf_Mold: 'Leaf Mold',
  Mosaic_virus: 'Mosaic Virus',
  Septoria_leaf_spot: 'Septoria Leaf Spot',
  Spider_mites: 'Spider Mites',
  Target_Spot: 'Target Spot',
  Yellow_Leaf_Curl_Virus: 'Yellow Leaf Curl Virus',
  Aphid: 'Aphid',
  Black_rust: 'Black Rust',
  Brown_leaf_Rust: 'Brown Leaf Rust',
  Leaf_blight: 'Leaf Blight',
  Mite: 'Mite',
  Powdery_mildew: 'Powdery Mildew',
  Scab: 'Scab',
  Stem_fly: 'Stem Fly',
  Yellow_Rust: 'Yellow Rust',
};

const DISEASE_NAME_AR: Record<string, string> = {
  Apple_scab: 'جرب التفاح',
  Black_rot: 'العفن الأسود',
  Cedar_apple_rust: 'صدأ أرز التفاح',
  healthy: 'سليم',
  Bacterial_spot_rot: 'تعفن البقعة البكتيرية',
  Black_Rot: 'العفن الأسود',
  Downy_Mildew: 'البياض الزغبي',
  Insect_Pest_Disease: 'مرض الآفات الحشرية',
  Leaf_Spot_Disease: 'مرض تبقع الأوراق',
  Mosaic_Virus_Disease: 'مرض فيروس الفسيفساء',
  Small_Leaf_Disease: 'مرض صغر الأوراق',
  White_Mold_Disease: 'مرض العفن الأبيض',
  Wilt_Disease: 'مرض الذبول',
  Esca_Black_Measles: 'إسكا (الحصبة السوداء)',
  Leaf_blight_Isariopsis_Leaf_Spot: 'لفحة الأوراق',
  Cercospora_leaf_spot_Gray_leaf_spot: 'بقعة الأوراق الرمادية',
  Common_rust: 'الصدأ الشائع',
  Northern_Leaf_Blight: 'لفحة الأوراق الشمالية',
  Aculus_olearius_mite: 'عث أكولوس الزيتون',
  Peacock_spot: 'بقعة عين الطاووس',
  Black_spot: 'البقعة السوداء',
  Canker: 'التقرح',
  Citrus_greening: 'تخضير الحمضيات',
  Bacterial_spot: 'البقعة البكتيرية',
  Early_blight: 'اللفحة المبكرة',
  Late_blight: 'اللفحة المتأخرة',
  Leaf_Mold: 'عفن الأوراق',
  Mosaic_virus: 'فيروس الفسيفساء',
  Septoria_leaf_spot: 'بقعة سبتوريا',
  Spider_mites: 'العنكبوت الأحمر',
  Target_Spot: 'البقعة المستهدفة',
  Yellow_Leaf_Curl_Virus: 'فيروس تجعد الأوراق الأصفر',
  Aphid: 'المن',
  Black_rust: 'الصدأ الأسود',
  Brown_leaf_Rust: 'صدأ الأوراق البني',
  Leaf_blight: 'لفحة الأوراق',
  Mite: 'العث',
  Powdery_mildew: 'البياض الدقيقي',
  Scab: 'الجرب',
  Stem_fly: 'ذبابة الساق',
  Yellow_Rust: 'الصدأ الأصفر',
};

export interface LocalDiseaseInfo {
  class_key: string;
  crop_name_en: string;
  crop_name_ar: string;
  disease_name_en: string;
  disease_name_ar: string;
  health_status: 'healthy' | 'diseased';
  description_en: string;
  description_ar: string;
  causes_en: string;
  causes_ar: string;
  treatment_en: string;
  treatment_ar: string;
}

function fallbackLabel(value: string) {
  return value.replace(/_/g, ' ');
}

export function buildDiseaseInfo(classKey: string): LocalDiseaseInfo {
  const [cropName = 'Unknown', diseaseKey = 'Unknown'] = classKey.split('___');
  const isHealthy = diseaseKey.toLowerCase() === 'healthy';
  const cropNameAr = CROP_NAME_AR[cropName] || cropName;
  const diseaseNameEn = DISEASE_NAME_EN[diseaseKey] || fallbackLabel(diseaseKey);
  const diseaseNameAr = DISEASE_NAME_AR[diseaseKey] || diseaseNameEn;

  return {
    class_key: classKey,
    crop_name_en: cropName,
    crop_name_ar: cropNameAr,
    disease_name_en: diseaseNameEn,
    disease_name_ar: diseaseNameAr,
    health_status: isHealthy ? 'healthy' : 'diseased',
    description_en: isHealthy
      ? `The ${cropName} crop appears healthy with no visible signs of disease.`
      : `${diseaseNameEn} detected on ${cropName}.`,
    description_ar: isHealthy
      ? `يبدو محصول ${cropNameAr} سليماً بدون علامات مرئية للمرض.`
      : `تم اكتشاف ${diseaseNameAr} على ${cropNameAr}.`,
    causes_en: '',
    causes_ar: '',
    treatment_en: '',
    treatment_ar: '',
  };
}
