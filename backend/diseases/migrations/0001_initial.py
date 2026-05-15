

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DiseaseInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('class_key', models.CharField(help_text='The exact string returned by the ML model.', max_length=255, unique=True)),
                ('crop_name_en', models.CharField(max_length=100)),
                ('crop_name_ar', models.CharField(max_length=100)),
                ('disease_name_en', models.CharField(max_length=100)),
                ('disease_name_ar', models.CharField(max_length=100)),
                ('health_status', models.CharField(choices=[('healthy', 'Healthy'), ('diseased', 'Diseased')], default='diseased', max_length=20)),
                ('description_en', models.TextField(blank=True, null=True)),
                ('description_ar', models.TextField(blank=True, null=True)),
                ('causes_en', models.TextField(blank=True, null=True)),
                ('causes_ar', models.TextField(blank=True, null=True)),
                ('treatment_en', models.TextField(blank=True, null=True)),
                ('treatment_ar', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
