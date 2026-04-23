from rest_framework import serializers
from .models import WeatherLog


class WeatherLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeatherLog
        fields = '__all__'
        read_only_fields = ['user', 'fetched_at']


class WeatherRequestSerializer(serializers.Serializer):
    """Validates incoming weather fetch requests from the mobile app."""
    latitude = serializers.FloatField(required=True)
    longitude = serializers.FloatField(required=True)
