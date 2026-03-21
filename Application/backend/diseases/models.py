from django.db import models


class Disease(models.Model):
    plant = models.CharField(max_length=100)
    disease = models.CharField(max_length=150, blank=True)
    is_healthy = models.BooleanField(default=False)

    def __str__(self):
        if self.is_healthy:
            return f"{self.plant} - Healthy"
        return f"{self.plant} - {self.disease}"
