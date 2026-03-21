from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DiseaseInfoViewSet

router = DefaultRouter()
router.register(r'diseases', DiseaseInfoViewSet, basename='diseaseinfo')

urlpatterns = [
    path('', include(router.urls)),
]
