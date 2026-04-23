from django.db import models
from django.conf import settings

class PredictionRecord(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='predictions')
    image = models.ImageField(upload_to='predictions/', null=True, blank=True)
    crop_name = models.CharField(max_length=100)
    disease_name_en = models.CharField(max_length=100)
    disease_name_ar = models.CharField(max_length=100)
    confidence = models.FloatField()
    is_healthy = models.BooleanField(default=False)
    predicted_at = models.DateTimeField(auto_now_add=True)
    
    # Offline sync info
    sync_status = models.CharField(
        max_length=20,
        choices=[('synced', 'Synced'), ('pending', 'Pending')],
        default='synced'
    )
    source_type = models.CharField(
        max_length=20,
        choices=[('camera', 'Camera'), ('gallery', 'Gallery')],
        default='camera'
    )
    
    # Optional weather context (sent from the mobile app during online predictions)
    weather_risk_level = models.CharField(max_length=20, blank=True, null=True)
    weather_temperature = models.FloatField(blank=True, null=True)
    weather_humidity = models.FloatField(blank=True, null=True)
    weather_wind_speed = models.FloatField(blank=True, null=True)
    weather_description = models.CharField(max_length=100, blank=True, null=True)
    weather_city_name = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.user.email} - {self.crop_name} - {self.disease_name_en}"
