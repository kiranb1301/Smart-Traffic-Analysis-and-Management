from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import TrafficRecord, UploadedCSV

admin.site.register(TrafficRecord)
admin.site.register(UploadedCSV)
