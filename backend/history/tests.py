from django.test import TestCase
from django.contrib.auth import get_user_model
from .models import PredictionRecord

User = get_user_model()


class PredictionRecordModelTest(TestCase):
    """Tests for the PredictionRecord model."""

    def setUp(self):
        self.user = User.objects.create_user(
            email='test@example.com',
            password='testpass123',
            full_name='Test User',
        )
        self.record = PredictionRecord.objects.create(
            user=self.user,
            crop_name='Tomato',
            disease_name_en='Bacterial Spot',
            disease_name_ar='البقعة البكتيرية',
            confidence=0.95,
            is_healthy=False,
            source_type='camera',
            sync_status='synced',
        )

    def test_record_creation(self):
        """PredictionRecord is created with correct fields."""
        self.assertEqual(self.record.crop_name, 'Tomato')
        self.assertEqual(self.record.confidence, 0.95)
        self.assertFalse(self.record.is_healthy)

    def test_record_str(self):
        """String representation includes user email and crop name."""
        self.assertIn('test@example.com', str(self.record))
        self.assertIn('Tomato', str(self.record))

    def test_record_default_sync_status(self):
        """Default sync_status is 'synced'."""
        record = PredictionRecord.objects.create(
            user=self.user,
            crop_name='Apple',
            disease_name_en='Healthy',
            disease_name_ar='سليم',
            confidence=0.99,
            is_healthy=True,
        )
        self.assertEqual(record.sync_status, 'synced')

    def test_user_predictions_queryset(self):
        """User's predictions are accessible via related manager."""
        predictions = self.user.predictions.all()
        self.assertEqual(predictions.count(), 1)
        self.assertEqual(predictions.first().crop_name, 'Tomato')

    def test_weather_context_nullable(self):
        """Weather fields default to null."""
        self.assertIsNone(self.record.weather_temperature)
        self.assertIsNone(self.record.weather_humidity)
        self.assertIsNone(self.record.weather_risk_level)

    def test_weather_context_saved(self):
        """Weather context is saved correctly."""
        record = PredictionRecord.objects.create(
            user=self.user,
            crop_name='Wheat',
            disease_name_en='Yellow Rust',
            disease_name_ar='الصدأ الأصفر',
            confidence=0.87,
            is_healthy=False,
            weather_temperature=28.0,
            weather_humidity=75.0,
            weather_wind_speed=3.5,
            weather_risk_level='medium',
            weather_description='Partly cloudy',
            weather_city_name='Amman',
        )
        self.assertEqual(record.weather_temperature, 28.0)
        self.assertEqual(record.weather_city_name, 'Amman')
        self.assertEqual(record.weather_risk_level, 'medium')
