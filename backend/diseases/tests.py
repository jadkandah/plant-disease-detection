from django.test import TestCase
from .models import DiseaseInfo


class DiseaseInfoModelTest(TestCase):
    """Tests for the DiseaseInfo model."""

    def setUp(self):
        self.disease = DiseaseInfo.objects.create(
            class_key='Tomato___Bacterial_spot',
            crop_name_en='Tomato',
            crop_name_ar='طماطم',
            disease_name_en='Bacterial Spot',
            disease_name_ar='البقعة البكتيرية',
            health_status='diseased',
        )

    def test_disease_info_creation(self):
        """DiseaseInfo is created with correct fields."""
        self.assertEqual(self.disease.class_key, 'Tomato___Bacterial_spot')
        self.assertEqual(self.disease.crop_name_en, 'Tomato')
        self.assertEqual(self.disease.health_status, 'diseased')

    def test_disease_str(self):
        """String representation includes crop and disease."""
        self.assertIn('Tomato', str(self.disease))
        self.assertIn('Bacterial Spot', str(self.disease))

    def test_unique_class_key(self):
        """Duplicate class_key should raise IntegrityError."""
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            DiseaseInfo.objects.create(
                class_key='Tomato___Bacterial_spot',
                crop_name_en='Tomato',
                crop_name_ar='طماطم',
                disease_name_en='Duplicate',
                disease_name_ar='مكرر',
            )

    def test_healthy_disease(self):
        """Healthy entries have correct health_status."""
        healthy = DiseaseInfo.objects.create(
            class_key='Tomato___healthy',
            crop_name_en='Tomato',
            crop_name_ar='طماطم',
            disease_name_en='Healthy',
            disease_name_ar='سليم',
            health_status='healthy',
        )
        self.assertEqual(healthy.health_status, 'healthy')
