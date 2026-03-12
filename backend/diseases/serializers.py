from rest_framework import serializers
from .models import DiseaseInfo

class DiseaseInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiseaseInfo
        fields = '__all__'
