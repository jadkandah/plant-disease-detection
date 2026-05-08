from rest_framework import serializers


class PredictionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    source_type = serializers.ChoiceField(choices=['camera', 'gallery'], required=False, default='camera')

    # Model mode: online uses SAM + backend ResNet50; offline uses the lightweight mobile model.
    mode = serializers.ChoiceField(choices=['online', 'offline'], required=False, default='offline')
