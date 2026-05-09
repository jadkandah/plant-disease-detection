from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('history', '0004_drop_legacy_weather_table'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictionrecord',
            name='model_mode',
            field=models.CharField(
                choices=[('online', 'Online'), ('offline', 'Offline')],
                default='online',
                max_length=20,
            ),
        ),
    ]
