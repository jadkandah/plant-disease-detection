# Generated manually after removing weather features from prediction history.

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('history', '0002_predictionrecord_weather_city_name_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='predictionrecord',
            name='weather_city_name',
        ),
        migrations.RemoveField(
            model_name='predictionrecord',
            name='weather_description',
        ),
        migrations.RemoveField(
            model_name='predictionrecord',
            name='weather_humidity',
        ),
        migrations.RemoveField(
            model_name='predictionrecord',
            name='weather_risk_level',
        ),
        migrations.RemoveField(
            model_name='predictionrecord',
            name='weather_temperature',
        ),
        migrations.RemoveField(
            model_name='predictionrecord',
            name='weather_wind_speed',
        ),
    ]
