from rest_framework import viewsets
from rest_framework.permissions import AllowAny
from .models import DiseaseInfo
from .serializers import DiseaseInfoSerializer

class DiseaseInfoViewSet(viewsets.ReadOnlyModelViewSet):

    queryset = DiseaseInfo.objects.all()
    serializer_class = DiseaseInfoSerializer
    permission_classes = [AllowAny]
