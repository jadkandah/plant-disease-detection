

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(blank=True, null=True, upload_to='predictions/')),
                ('crop_name', models.CharField(max_length=100)),
                ('disease_name_en', models.CharField(max_length=100)),
                ('disease_name_ar', models.CharField(max_length=100)),
                ('confidence', models.FloatField()),
                ('is_healthy', models.BooleanField(default=False)),
                ('predicted_at', models.DateTimeField(auto_now_add=True)),
                ('sync_status', models.CharField(choices=[('synced', 'Synced'), ('pending', 'Pending')], default='synced', max_length=20)),
                ('source_type', models.CharField(choices=[('camera', 'Camera'), ('gallery', 'Gallery')], default='camera', max_length=20)),
                ('weather_risk_level', models.CharField(blank=True, max_length=20, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='predictions', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
