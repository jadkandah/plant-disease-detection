from django.db import models

class DiseaseInfo(models.Model):
    class_key = models.CharField(max_length=255, unique=True, help_text="The exact string returned by the ML model.")
    crop_name_en = models.CharField(max_length=100)
    crop_name_ar = models.CharField(max_length=100)
    disease_name_en = models.CharField(max_length=100)
    disease_name_ar = models.CharField(max_length=100)

    health_status = models.CharField(
        max_length=20,
        choices=[('healthy', 'Healthy'), ('diseased', 'Diseased')],
        default='diseased'
    )

    def __str__(self):
        return f"{self.crop_name_en} - {self.disease_name_en} ({self.class_key})"
