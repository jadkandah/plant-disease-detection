from rest_framework import serializers
from .models import PredictionRecord

class PredictionRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionRecord
        fields = '__all__'
        read_only_fields = ['user', 'predicted_at']

class SyncPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionRecord
        fields = '__all__'
        read_only_fields = ['user']
