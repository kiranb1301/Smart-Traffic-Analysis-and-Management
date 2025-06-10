from django.db import models

class TrafficRecord(models.Model):
    date_time = models.DateTimeField()
    holiday = models.CharField(max_length=100, null=True, blank=True)
    temp = models.FloatField()
    rain_1h = models.FloatField()
    snow_1h = models.FloatField()
    clouds_all = models.IntegerField()
    weather_main = models.CharField(max_length=100)
    weather_description = models.CharField(max_length=255)
    traffic_volume = models.IntegerField()

    def __str__(self):
        return f"{self.date_time} - {self.traffic_volume}"

class UploadedCSV(models.Model):
    file = models.FileField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
