from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser
from authentication.models import User
from authentication.serializers import UserSerializer
from history.models import PredictionRecord
from history.serializers import PredictionRecordSerializer
from diseases.models import DiseaseInfo
from diseases.serializers import DiseaseInfoSerializer


class AdminUserViewSet(viewsets.ReadOnlyModelViewSet):
    """View all user accounts (admin only)."""
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]


class AdminPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """View ALL prediction records across all users (admin only)."""
    queryset = PredictionRecord.objects.all().order_by('-predicted_at')
    serializer_class = PredictionRecordSerializer
    permission_classes = [IsAdminUser]

    @action(detail=False, methods=['get'], url_path='stats')
    def stats(self, request):
        """Return aggregate statistics for the admin dashboard."""
        total_predictions = PredictionRecord.objects.count()
        total_users = User.objects.count()
        diseased = PredictionRecord.objects.filter(is_healthy=False).count()
        healthy = PredictionRecord.objects.filter(is_healthy=True).count()

        return Response({
            'total_users': total_users,
            'total_predictions': total_predictions,
            'diseased_detections': diseased,
            'healthy_detections': healthy,
        })


class AdminDiseaseViewSet(viewsets.ModelViewSet):
    """Full CRUD on disease information (admin only)."""
    queryset = DiseaseInfo.objects.all()
    serializer_class = DiseaseInfoSerializer
    permission_classes = [IsAdminUser]
