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

        # Extract optional weather context
        weather_temperature = serializer.validated_data.get('temperature')
        weather_humidity = serializer.validated_data.get('humidity')
        weather_wind_speed = serializer.validated_data.get('wind_speed')
        weather_description = serializer.validated_data.get('weather_description', '')
        weather_risk_level = serializer.validated_data.get('weather_risk_level', '')
        weather_city_name = serializer.validated_data.get('city_name', '')

        if weather_temperature is not None:
            print(f"[predict] Weather context: {weather_temperature}°C, {weather_humidity}% humidity, "
                  f"wind {weather_wind_speed} m/s, {weather_description}, risk={weather_risk_level}, city={weather_city_name}")

        # ──────────────────────────────────────────
        # 2. Preprocessing pipeline (quality check + optional SAM)
        #
        #   🟢 Offline: image → quality check → inference
        #   🔵 Online:  image → quality check → SAM leaf extraction → inference
        # ──────────────────────────────────────────
        print(f"[predict] Mode: {mode}")
        preprocessed_image, preprocess_status = preprocess_image(image_file, mode=mode)

        if preprocessed_image is None:
            return Response(
                {"error": preprocess_status},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 3. Run AI inference on the preprocessed image
        try:
            predicted_class_key, confidence = predict_from_array(preprocessed_image)
        except Exception as e:
            return Response(
                {"error": f"Model inference failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 4. Look up the disease info from the database
        try:
            predicted_disease = DiseaseInfo.objects.get(class_key=predicted_class_key)
        except DiseaseInfo.DoesNotExist:
            return Response(
                {"error": f"No disease info found for class: {predicted_class_key}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 5. Save to prediction history (including weather context)
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
            weather_risk_level=weather_risk_level or None,
            weather_temperature=weather_temperature,
            weather_humidity=weather_humidity,
            weather_wind_speed=weather_wind_speed,
            weather_description=weather_description or None,
            weather_city_name=weather_city_name or None,
        )

        # 6. Build weather context for response
        weather_context = None
        if weather_temperature is not None:
            weather_context = {
                "temperature": weather_temperature,
                "humidity": weather_humidity,
                "wind_speed": weather_wind_speed,
                "description": weather_description,
                "risk_level": weather_risk_level,
                "city_name": weather_city_name,
            }

        # 7. Return the result
        return Response({
            "success": True,
            "mode": mode,
            "prediction_key": predicted_disease.class_key,
            "confidence": confidence,
            "is_healthy": predicted_disease.health_status == 'healthy',
            "disease_info": DiseaseInfoSerializer(predicted_disease).data,
            "weather_context": weather_context,
        }, status=status.HTTP_200_OK)
