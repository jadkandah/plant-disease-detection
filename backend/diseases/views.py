from rest_framework import viewsets
from rest_framework.permissions import AllowAny
from .models import DiseaseInfo
from .serializers import DiseaseInfoSerializer

class DiseaseInfoViewSet(viewsets.ReadOnlyModelViewSet):
    """
    A viewset that provides only `list()` and `retrieve()` actions.
    Useful for populating the Supported Crops & Diseases screen.
    """
    queryset = DiseaseInfo.objects.all()
    serializer_class = DiseaseInfoSerializer
    permission_classes = [AllowAny]
