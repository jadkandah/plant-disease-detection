from django.db import models
from django.conf import settings


class WeatherLog(models.Model):
    """
    Stores a snapshot of weather data fetched for a user's location.
    Useful for analytics, auditing, and correlating weather with disease outbreaks.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='weather_logs',
        null=True,
        blank=True,
    )
    latitude = models.FloatField()
    longitude = models.FloatField()
    city_name = models.CharField(max_length=100, blank=True, default='')
    country = models.CharField(max_length=10, blank=True, default='')

    temperature = models.FloatField(help_text='Temperature in °C')
    humidity = models.FloatField(help_text='Relative humidity %')
    wind_speed = models.FloatField(help_text='Wind speed in m/s', default=0)
    feels_like = models.FloatField(help_text='Apparent temperature in °C', default=0)
    pressure = models.FloatField(help_text='Surface pressure in hPa', default=0)
    weather_code = models.IntegerField(help_text='WMO weather code', default=0)
    description = models.CharField(max_length=100, blank=True, default='')

    risk_level = models.CharField(
        max_length=10,
        choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')],
        default='low',
    )
    risk_message = models.TextField(blank=True, default='')

    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-fetched_at']

    def __str__(self):
        return f"{self.city_name} — {self.temperature}°C, {self.humidity}% humidity ({self.risk_level} risk)"
