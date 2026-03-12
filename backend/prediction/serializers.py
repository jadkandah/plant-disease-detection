from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    source_type = serializers.ChoiceField(choices=['camera', 'gallery'], required=False, default='camera')
