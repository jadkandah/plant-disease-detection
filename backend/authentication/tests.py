from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status

User = get_user_model()


class AuthenticationTest(TestCase):
    """Tests for the authentication endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user_data = {
            'email': 'testuser@example.com',
            'password': 'SecurePass123!',
            'full_name': 'Test User',
            'phone_number': '+962791234567',
        }

    def test_register_user(self):
        """Registration creates a new user and returns tokens."""
        response = self.client.post('/api/auth/register/', self.user_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('tokens', response.data)
        self.assertIn('access', response.data['tokens'])
        self.assertIn('refresh', response.data['tokens'])
        self.assertEqual(response.data['user']['email'], self.user_data['email'])

    def test_register_duplicate_email(self):
        """Registering with an existing email returns 400."""
        self.client.post('/api/auth/register/', self.user_data)
        response = self.client.post('/api/auth/register/', self.user_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_valid_credentials(self):
        """Login with valid credentials returns tokens."""
        # Register first
        self.client.post('/api/auth/register/', self.user_data)
        # Login
        response = self.client.post('/api/auth/login/', {
            'email': self.user_data['email'],
            'password': self.user_data['password'],
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('tokens', response.data)

    def test_login_invalid_credentials(self):
        """Login with wrong password returns 400."""
        self.client.post('/api/auth/register/', self.user_data)
        response = self.client.post('/api/auth/login/', {
            'email': self.user_data['email'],
            'password': 'WrongPassword',
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_profile_authenticated(self):
        """Authenticated user can retrieve their profile."""
        # Register and get token
        reg_response = self.client.post('/api/auth/register/', self.user_data)
        token = reg_response.data['tokens']['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = self.client.get('/api/auth/profile/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], self.user_data['email'])

    def test_profile_unauthenticated(self):
        """Unauthenticated access to profile returns 401."""
        response = self.client.get('/api/auth/profile/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_change_password(self):
        """User can change their password."""
        reg_response = self.client.post('/api/auth/register/', self.user_data)
        token = reg_response.data['tokens']['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = self.client.put('/api/auth/change-password/', {
            'old_password': self.user_data['password'],
            'new_password': 'NewSecurePass456!',
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
