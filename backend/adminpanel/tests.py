from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from history.models import PredictionRecord

User = get_user_model()


class AdminEndpointTest(TestCase):
    """Tests for the admin panel endpoints."""

    def setUp(self):
        self.client = APIClient()
        # Regular user
        self.user = User.objects.create_user(
            email='regular@test.com',
            password='testpass123',
            full_name='Regular User',
        )
        # Admin user
        self.admin = User.objects.create_user(
            email='admin@test.com',
            password='adminpass123',
            full_name='Admin User',
        )
        self.admin.is_staff = True
        self.admin.is_admin = True
        self.admin.save()

    def test_admin_stats_requires_admin(self):
        """Regular users cannot access admin stats."""
        self.client.force_authenticate(user=self.user)
        response = self.client.get('/api/admin/predictions/stats/')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_admin_stats_accessible_by_admin(self):
        """Admin users can access stats endpoint."""
        self.client.force_authenticate(user=self.admin)
        response = self.client.get('/api/admin/predictions/stats/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('total_users', response.data)
        self.assertIn('total_predictions', response.data)

    def test_admin_users_list(self):
        """Admin can list all users."""
        self.client.force_authenticate(user=self.admin)
        response = self.client.get('/api/admin/users/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)  # regular + admin
