from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from .models import WeatherLog

User = get_user_model()


class HealthCheckTest(TestCase):
    """Tests for the /api/weather/health/ endpoint."""

    def test_health_check_returns_ok(self):
        """Health check endpoint is publicly accessible and returns ok."""
        client = APIClient()
        response = client.get('/api/weather/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'ok')


class WeatherLogModelTest(TestCase):
    """Tests for the WeatherLog model."""

    def setUp(self):
        self.user = User.objects.create_user(
            email='weather@test.com',
            password='testpass123',
            full_name='Weather User',
        )

    def test_weather_log_creation(self):
        """WeatherLog is created with correct fields."""
        log = WeatherLog.objects.create(
            user=self.user,
            latitude=31.95,
            longitude=35.93,
            city_name='Amman',
            temperature=28,
            humidity=65,
            wind_speed=3.2,
            risk_level='medium',
            risk_message='Moderate conditions.',
        )
        self.assertEqual(log.city_name, 'Amman')
        self.assertEqual(log.risk_level, 'medium')

    def test_weather_log_str(self):
        """String representation includes city and conditions."""
        log = WeatherLog.objects.create(
            user=self.user,
            latitude=31.95,
            longitude=35.93,
            city_name='Amman',
            temperature=28,
            humidity=65,
            risk_level='medium',
        )
        self.assertIn('Amman', str(log))
        self.assertIn('28', str(log))


class WeatherFetchEndpointTest(TestCase):
    """Tests for the /api/weather/fetch/ endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='weatherfetch@test.com',
            password='testpass123',
            full_name='Fetch User',
        )
        self.client.force_authenticate(user=self.user)

    def test_fetch_requires_authentication(self):
        """Unauthenticated requests to fetch weather are rejected."""
        client = APIClient()
        response = client.post('/api/weather/fetch/', {'latitude': 31.95, 'longitude': 35.93})
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_fetch_requires_coordinates(self):
        """Request without coordinates returns 400."""
        response = self.client.post('/api/weather/fetch/', {})
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
