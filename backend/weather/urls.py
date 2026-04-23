from django.urls import path
from .views import WeatherFetchView, WeatherHistoryView, HealthCheckView

urlpatterns = [
    path('fetch/', WeatherFetchView.as_view(), name='weather-fetch'),
    path('history/', WeatherHistoryView.as_view(), name='weather-history'),
    path('health/', HealthCheckView.as_view(), name='health-check'),
]
