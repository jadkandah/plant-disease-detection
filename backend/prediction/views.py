from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import PredictionRequestSerializer
from .inference import predict_image
from diseases.models import DiseaseInfo
from diseases.serializers import DiseaseInfoSerializer
from history.models import PredictionRecord

class PredictView(generics.CreateAPIView):
    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = PredictionRequestSerializer

    def post(self, request, *args, **kwargs):
        # 1. Validate the incoming request (needs an image)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        image_file = serializer.validated_data['image']
        source_type = serializer.validated_data.get('source_type', 'camera')
        
        # 2. Run REAL AI inference using the trained ResNet-50 model
        try:
            predicted_class_key, confidence = predict_image(image_file)
        except Exception as e:
            return Response(
                {"error": f"Model inference failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # 3. Look up the disease info from the database
        try:
            predicted_disease = DiseaseInfo.objects.get(class_key=predicted_class_key)
        except DiseaseInfo.DoesNotExist:
            return Response(
                {"error": f"No disease info found for class: {predicted_class_key}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 4. Save to prediction history
        # Re-open the image file for saving (seek back to start)
        image_file.seek(0)
        PredictionRecord.objects.create(
            user=request.user,
            image=image_file,
            crop_name=predicted_disease.crop_name_en,
            disease_name_en=predicted_disease.disease_name_en,
            disease_name_ar=predicted_disease.disease_name_ar,
            confidence=confidence,
            is_healthy=predicted_disease.health_status == 'healthy',
            source_type=source_type,
            sync_status='synced',
        )

        # 5. Return the result
        return Response({
            "success": True,
            "prediction_key": predicted_disease.class_key,
            "confidence": confidence,
            "is_healthy": predicted_disease.health_status == 'healthy',
            "disease_info": DiseaseInfoSerializer(predicted_disease).data
        }, status=status.HTTP_200_OK)

