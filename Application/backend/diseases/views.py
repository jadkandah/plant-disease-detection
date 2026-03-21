from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from .models import Disease
from .serializers import DiseaseSerializer


@api_view(['GET'])
@permission_classes([AllowAny])
def disease_list(request):
    diseases = Disease.objects.all()
    serializer = DiseaseSerializer(diseases, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([AllowAny])
def disease_detail(request, pk):
    try:
        disease = Disease.objects.get(pk=pk)
    except Disease.DoesNotExist:
        return Response({'error': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

    serializer = DiseaseSerializer(disease)
    return Response(serializer.data)
