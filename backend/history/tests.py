from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework import status
from rest_framework.test import APIClient
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

    def test_record_default_model_mode(self):
        """Default model_mode is 'online' for server-side predictions."""
        record = PredictionRecord.objects.create(
            user=self.user,
            crop_name='Apple',
            disease_name_en='Healthy',
            disease_name_ar='سليم',
            confidence=0.99,
            is_healthy=True,
        )
        self.assertEqual(record.model_mode, 'online')

    def test_user_predictions_queryset(self):
        """User's predictions are accessible via related manager."""
        predictions = self.user.predictions.all()
        self.assertEqual(predictions.count(), 1)
        self.assertEqual(predictions.first().crop_name, 'Tomato')


class HistorySyncEndpointTest(TestCase):
    """Tests for syncing locally generated predictions into server history."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='sync@example.com',
            password='testpass123',
            full_name='Sync User',
        )
        self.client.force_authenticate(user=self.user)

    def test_sync_offline_prediction_marks_model_mode_offline(self):
        response = self.client.post('/api/history/sync/', {
            'records': [{
                'crop_name': 'Tomato',
                'disease_name_en': 'Healthy',
                'disease_name_ar': 'سليم',
                'confidence': 0.98,
                'is_healthy': True,
                'source_type': 'camera',
                'predicted_at': '2026-05-09T10:15:00Z',
            }]
        }, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        record = PredictionRecord.objects.get(user=self.user)
        self.assertEqual(record.model_mode, 'offline')
        self.assertEqual(record.sync_status, 'synced')
        self.assertEqual(record.source_type, 'camera')
        self.assertEqual(record.predicted_at.isoformat().replace('+00:00', 'Z'), '2026-05-09T10:15:00Z')
