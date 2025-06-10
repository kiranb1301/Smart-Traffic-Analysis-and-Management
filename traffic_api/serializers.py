from rest_framework import serializers
from .models import TrafficRecord, UploadedCSV

class TrafficRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrafficRecord
        fields = '__all__'

class UploadedCSVSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedCSV
        fields = '__all__'
