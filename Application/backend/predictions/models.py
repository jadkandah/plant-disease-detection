from django.db import models
from django.contrib.auth.models import User
from diseases.models import Disease


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    image = models.ImageField(upload_to='predictions/')
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE, related_name='predictions')
    confidence = models.FloatField()
    top_predictions = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.disease} - {self.confidence:.2f}"