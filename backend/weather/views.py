import requests
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from .models import WeatherLog
from .serializers import WeatherRequestSerializer, WeatherLogSerializer


def _estimate_risk(temp: float, humidity: float) -> dict:
    """Estimate plant disease risk based on temperature and humidity."""
    if humidity > 80 and 20 < temp < 35:
        return {
            'level': 'high',
            'message': 'High humidity & warm temps increase fungal disease risk. Inspect crops closely.',
        }
    if humidity > 60 and temp > 15:
        return {
            'level': 'medium',
            'message': 'Moderate conditions. Monitor your crops for early signs of disease.',
        }
    return {
        'level': 'low',
        'message': 'Current weather conditions are favorable for healthy crops.',
    }


def _get_weather_description(code: int) -> str:
    """Map WMO weather code to human-readable description."""
    if code == 0:
        return 'Clear sky'
    if code <= 3:
        return 'Partly cloudy'
    if code <= 49:
        return 'Foggy'
    if code <= 59:
        return 'Drizzle'
    if code <= 69:
        return 'Rain'
    if code <= 79:
        return 'Snow'
    if code <= 99:
        return 'Thunderstorm'
    return 'Clear'


class WeatherFetchView(APIView):
    """
    POST /api/weather/fetch/
    
    Fetches real-time weather data from Open-Meteo API based on the user's
    GPS coordinates. Computes a plant disease risk level, stores the data
    in WeatherLog, and returns the result.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = WeatherRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        lat = serializer.validated_data['latitude']
        lon = serializer.validated_data['longitude']

        # Fetch from Open-Meteo (free, no API key needed)
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
                f"surface_pressure,wind_speed_10m,weather_code"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return Response(
                {"error": f"Failed to fetch weather data: {str(e)}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        current = data.get('current', {})
        temp = current.get('temperature_2m', 0)
        humidity = current.get('relative_humidity_2m', 0)
        wind = current.get('wind_speed_10m', 0)
        feels = current.get('apparent_temperature', temp)
        pressure = current.get('surface_pressure', 0)
        weather_code = current.get('weather_code', 0)

        risk = _estimate_risk(temp, humidity)
        description = _get_weather_description(weather_code)

        # Save to database
        log = WeatherLog.objects.create(
            user=request.user,
            latitude=lat,
            longitude=lon,
            temperature=round(temp, 1),
            humidity=round(humidity, 1),
            wind_speed=round(wind, 1),
            feels_like=round(feels, 1),
            pressure=round(pressure, 1),
            weather_code=weather_code,
            description=description,
            risk_level=risk['level'],
            risk_message=risk['message'],
        )

        return Response({
            "temperature": log.temperature,
            "humidity": log.humidity,
            "wind_speed": log.wind_speed,
            "feels_like": log.feels_like,
            "pressure": log.pressure,
            "description": log.description,
            "weather_code": log.weather_code,
            "risk_level": log.risk_level,
            "risk_message": log.risk_message,
            "fetched_at": log.fetched_at.isoformat(),
        }, status=status.HTTP_200_OK)


class WeatherHistoryView(generics.ListAPIView):
    """
    GET /api/weather/history/
    
    Returns the weather log history for the authenticated user.
    """
    serializer_class = WeatherLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return WeatherLog.objects.filter(user=self.request.user).order_by('-fetched_at')[:50]


class HealthCheckView(APIView):
    """
    GET /api/weather/health/
    
    Simple connectivity check endpoint. Returns 200 OK if the server is reachable.
    Used by the mobile app to test backend connectivity.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            "status": "ok",
            "service": "plant-disease-detection-api",
        }, status=status.HTTP_200_OK)
