from rest_framework import serializers
from .models import Prediction
from diseases.serializers import DiseaseSerializer


class PredictionSerializer(serializers.ModelSerializer):
    disease = DiseaseSerializer(read_only=True)

    class Meta:
        model = Prediction
        fields = [
            'id',
            'user',
            'image',
            'disease',
            'confidence',
            'top_predictions',
            'created_at',
        ]