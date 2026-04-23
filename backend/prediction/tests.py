from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status

User = get_user_model()


class PredictionEndpointTest(TestCase):
    """Tests for the /api/predict/ endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='predict@test.com',
            password='testpass123',
            full_name='Predict User',
        )
        self.client.force_authenticate(user=self.user)

    def test_predict_requires_authentication(self):
        """Unauthenticated requests are rejected."""
        client = APIClient()
        response = client.post('/api/predict/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_predict_requires_image(self):
        """Request without image returns 400."""
        response = self.client.post('/api/predict/', {}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
