from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from rest_framework import status
from io import BytesIO
from unittest.mock import patch
import numpy as np
from PIL import Image

from .preprocessing.pipeline import preprocess_image
from .preprocessing.leaf_check import is_leaf_color

User = get_user_model()


def make_test_image_file(name='leaf.jpg', color=(64, 160, 64)):
    buffer = BytesIO()
    Image.new('RGB', (16, 16), color=color).save(buffer, format='JPEG')
    return SimpleUploadedFile(name, buffer.getvalue(), content_type='image/jpeg')


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

    @patch('prediction.views.predict_from_array')
    @patch('prediction.views.preprocess_image')
    def test_online_mode_is_passed_to_pipeline_and_inference(self, mock_preprocess, mock_predict):
        mock_preprocess.return_value = (np.zeros((16, 16, 3), dtype=np.uint8), 'OK')
        mock_predict.return_value = ('Tomato___healthy', 0.9)

        response = self.client.post(
            '/api/predict/',
            {'image': make_test_image_file(), 'source_type': 'camera', 'mode': 'online'},
            format='multipart',
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['mode'], 'online')
        self.assertEqual(mock_preprocess.call_args.kwargs['mode'], 'online')
        self.assertEqual(mock_predict.call_args.kwargs['mode'], 'online')

    @patch('prediction.views.predict_from_array')
    @patch('prediction.views.preprocess_image')
    def test_offline_mode_is_passed_to_pipeline_and_inference(self, mock_preprocess, mock_predict):
        mock_preprocess.return_value = (np.zeros((16, 16, 3), dtype=np.uint8), 'OK')
        mock_predict.return_value = ('Tomato___healthy', 0.9)

        response = self.client.post(
            '/api/predict/',
            {'image': make_test_image_file(), 'source_type': 'gallery', 'mode': 'offline'},
            format='multipart',
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['mode'], 'offline')
        self.assertEqual(mock_preprocess.call_args.kwargs['mode'], 'offline')
        self.assertEqual(mock_predict.call_args.kwargs['mode'], 'offline')


class PreprocessingPipelineTest(TestCase):
    """Tests for mode-specific preprocessing behavior."""

    @patch('prediction.preprocessing.pipeline.extract_leaf')
    @patch('prediction.preprocessing.pipeline.check_quality')
    def test_online_mode_runs_quality_then_sam(self, mock_quality, mock_extract):
        mock_quality.return_value = (True, 'good')
        sam_image = np.ones((16, 16, 3), dtype=np.uint8)
        mock_extract.return_value = sam_image

        image, status_message = preprocess_image(make_test_image_file(), mode='online')

        self.assertEqual(status_message, 'OK')
        self.assertIs(image, sam_image)
        mock_quality.assert_called_once()
        mock_extract.assert_called_once()

    @patch('prediction.preprocessing.pipeline.extract_leaf')
    @patch('prediction.preprocessing.pipeline.check_quality')
    def test_offline_mode_runs_quality_without_sam(self, mock_quality, mock_extract):
        mock_quality.return_value = (True, 'good')

        image, status_message = preprocess_image(make_test_image_file(), mode='offline')

        self.assertEqual(status_message, 'OK')
        self.assertIsNotNone(image)
        mock_quality.assert_called_once()
        mock_extract.assert_not_called()

    @patch('prediction.preprocessing.pipeline.extract_leaf')
    @patch('prediction.preprocessing.pipeline.check_quality')
    def test_non_leaf_image_is_rejected_after_quality_check(self, mock_quality, mock_extract):
        mock_quality.return_value = (True, 'good')

        image, status_message = preprocess_image(
            make_test_image_file(name='not_leaf.jpg', color=(128, 128, 128)),
            mode='offline',
        )

        self.assertIsNone(image)
        self.assertTrue(status_message.startswith('Rejected: Not a leaf'))
        mock_quality.assert_called_once()
        mock_extract.assert_not_called()

    @patch('prediction.preprocessing.pipeline.check_quality')
    def test_leaf_image_passes_preprocessing(self, mock_quality):
        mock_quality.return_value = (True, 'good')

        image, status_message = preprocess_image(make_test_image_file(), mode='offline')

        self.assertIsNotNone(image)
        self.assertEqual(status_message, 'OK')
        mock_quality.assert_called_once()


class LeafColorCheckTest(TestCase):
    """Tests for HSV color-based leaf detection."""

    def test_green_leaf_color_is_detected(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :] = (64, 160, 64)
        is_leaf, leaf_ratio = is_leaf_color(image)

        self.assertTrue(is_leaf)
        self.assertGreater(leaf_ratio, 0.1)

    def test_gray_non_leaf_color_is_rejected(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :] = (128, 128, 128)
        is_leaf, leaf_ratio = is_leaf_color(image)

        self.assertFalse(is_leaf)
        self.assertEqual(leaf_ratio, 0)

    def test_warm_brown_non_leaf_color_is_rejected_without_green(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :] = (35, 105, 150)
        is_leaf, leaf_ratio = is_leaf_color(image)

        self.assertFalse(is_leaf)
        self.assertGreater(leaf_ratio, 0.1)
