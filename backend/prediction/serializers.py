from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    source_type = serializers.ChoiceField(choices=['camera', 'gallery'], required=False, default='camera')

    # Optional weather context sent from the mobile app
    temperature = serializers.FloatField(required=False, allow_null=True, default=None)
    humidity = serializers.FloatField(required=False, allow_null=True, default=None)
    wind_speed = serializers.FloatField(required=False, allow_null=True, default=None)
    weather_description = serializers.CharField(required=False, allow_blank=True, default='')
    weather_risk_level = serializers.CharField(required=False, allow_blank=True, default='')
    city_name = serializers.CharField(required=False, allow_blank=True, default='')
