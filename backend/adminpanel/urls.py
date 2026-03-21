from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AdminUserViewSet, AdminPredictionViewSet, AdminDiseaseViewSet

router = DefaultRouter()
router.register(r'users', AdminUserViewSet, basename='admin-users')
router.register(r'predictions', AdminPredictionViewSet, basename='admin-predictions')
router.register(r'diseases', AdminDiseaseViewSet, basename='admin-diseases')

urlpatterns = [
    path('', include(router.urls)),
]
