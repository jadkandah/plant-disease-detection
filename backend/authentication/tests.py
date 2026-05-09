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
            'phone_number': '0791234567',
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

    def test_register_normalizes_email_and_phone(self):
        """Registration stores lowercase email and a digit-only local phone number."""
        data = {
            **self.user_data,
            'email': 'MIXEDCASE@EXAMPLE.COM',
            'phone_number': '079 123-4567',
        }
        response = self.client.post('/api/auth/register/', data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        user = User.objects.get(email='mixedcase@example.com')
        self.assertEqual(user.phone_number, '0791234567')
        self.assertEqual(response.data['user']['email'], 'mixedcase@example.com')

    def test_register_rejects_invalid_email(self):
        """Email must look like a real address."""
        data = {**self.user_data, 'email': 'invalid-email'}
        response = self.client.post('/api/auth/register/', data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('email', response.data)

    def test_register_rejects_invalid_phone_number(self):
        """Phone number must be 10 digits and start with 07."""
        invalid_numbers = ['+962791234567', '061234567', '079123456', '07912345678']

        for phone_number in invalid_numbers:
            with self.subTest(phone_number=phone_number):
                data = {
                    **self.user_data,
                    'email': f'{phone_number.replace("+", "plus")}@example.com',
                    'phone_number': phone_number,
                }
                response = self.client.post('/api/auth/register/', data)
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                self.assertIn('phone_number', response.data)

    def test_register_rejects_weak_password(self):
        """Password must satisfy strength requirements."""
        weak_passwords = ['short1!', 'lowercase123!', 'UPPERCASE123!', 'NoNumber!', 'NoSpecial123']

        for index, password in enumerate(weak_passwords):
            with self.subTest(password=password):
                data = {
                    **self.user_data,
                    'email': f'weak{index}@example.com',
                    'password': password,
                }
                response = self.client.post('/api/auth/register/', data)
                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                self.assertIn('password', response.data)

    def test_register_password_cannot_contain_name_or_email(self):
        """Password cannot reuse obvious account identity fields."""
        data = {
            **self.user_data,
            'email': 'jane@example.com',
            'full_name': 'Jane Farmer',
            'password': 'JaneFarmer123!',
        }
        response = self.client.post('/api/auth/register/', data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)

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

    def test_change_password_rejects_weak_new_password(self):
        """Password strength rules also apply when changing a password."""
        reg_response = self.client.post('/api/auth/register/', self.user_data)
        token = reg_response.data['tokens']['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = self.client.put('/api/auth/change-password/', {
            'old_password': self.user_data['password'],
            'new_password': 'weak',
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('new_password', response.data)
