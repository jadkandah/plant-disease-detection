from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import PredictionRequestSerializer
from .inference import predict_from_array
from .preprocessing.pipeline import preprocess_image
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
        mode = serializer.validated_data.get('mode', 'offline')

        # ──────────────────────────────────────────
        # 2. Preprocessing pipeline (quality check + optional SAM)
        #
        #   🟢 Offline: image → quality check → MobileNetV3-Small
        #   🔵 Online:  image → quality check → SAM → ResNet50 image-only 512
        # ──────────────────────────────────────────
        print(f"[predict] Mode: {mode}")
        preprocessed_image, preprocess_status = preprocess_image(image_file, mode=mode)

        if preprocessed_image is None:
            return Response(
                {"error": preprocess_status},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3. Run AI inference on the preprocessed image.
        try:
            predicted_class_key, confidence = predict_from_array(
                preprocessed_image,
                mode=mode,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Model inference failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 4. Look up the disease info from the database
        try:
            predicted_disease = DiseaseInfo.objects.get(class_key=predicted_class_key)
        except DiseaseInfo.DoesNotExist:
            # If the class isn't in the DB yet, return a basic response
            print(f"[predict] WARNING: No DiseaseInfo for class '{predicted_class_key}' — returning raw result")
            return Response({
                "success": True,
                "mode": mode,
                "prediction_key": predicted_class_key,
                "confidence": confidence,
                "is_healthy": "healthy" in predicted_class_key.lower(),
                "disease_info": None,
            }, status=status.HTTP_200_OK)

        # 5. Save to prediction history
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

        # 6. Return the result
        return Response({
            "success": True,
            "mode": mode,
            "prediction_key": predicted_disease.class_key,
            "confidence": confidence,
            "is_healthy": predicted_disease.health_status == 'healthy',
            "disease_info": DiseaseInfoSerializer(predicted_disease).data,
        }, status=status.HTTP_200_OK)
