from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import pandas as pd
from .models import TrafficRecord, UploadedCSV
from .serializers import TrafficRecordSerializer, UploadedCSVSerializer
from django.utils.dateparse import parse_datetime
from django.shortcuts import render
from django.utils.dateparse import parse_datetime
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import TrafficRecord
from .serializers import UploadedCSVSerializer

from rest_framework.views import APIView
from rest_framework.response import Response
from ml_models.data_loader import load_static_data 

from rest_framework.views import APIView
from rest_framework.response import Response
from ml_models.traffic_optimizer import optimize_traffic
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ml_models.data_loader import load_static_data
from ml_models.traffic_predictor import train_rf_model_with_tuning, train_lstm_model_with_early_stopping
from ml_models.cluster_analysis import detect_congestion_clusters
from ml_models.realtime_utils import simulate_realtime_from_chunks
from rest_framework.decorators import api_view
from rest_framework.response import Response

def index(request):
    return render(request, 'index.html')

class CSVUploadView(generics.CreateAPIView):
    serializer_class = UploadedCSVSerializer
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['file']
        df = pd.read_csv(file_obj)

        # Process and insert into DB
        for _, row in df.iterrows():
            dt = parse_datetime(row['date_time'])
            if dt is None:
                # Skip or handle invalid datetime
                continue
            if timezone.is_naive(dt):
                dt = timezone.make_aware(dt)

            TrafficRecord.objects.create(
                date_time=dt,
                holiday=row['holiday'],
                temp=row['temp'],
                rain_1h=row['rain_1h'],
                snow_1h=row['snow_1h'],
                clouds_all=row['clouds_all'],
                weather_main=row['weather_main'],
                weather_description=row['weather_description'],
                traffic_volume=row['traffic_volume']
            )

        return Response({"message": "File uploaded and data saved."}, status=status.HTTP_201_CREATED)


class TrafficRecordListCreate(generics.ListCreateAPIView):
    queryset = TrafficRecord.objects.all()
    serializer_class = TrafficRecordSerializer


class PredictTrafficAPIView(APIView):
    def post(self, request):
        model_type = request.data.get("model", "rf")
        df = load_static_data()

        if model_type == 'rf':
            model, _ = train_rf_model_with_tuning(df)
            message = "Random Forest prediction completed."
        else:
            model, _ = train_lstm_model_with_early_stopping(df)
            message = "LSTM prediction completed."

        return Response({"message": message}, status=status.HTTP_200_OK)  
    

class OptimizeTrafficAPIView(APIView):
    def post(self, request):
        model_type = request.data.get("model", "rf")
        intersections = request.data.get("intersections", ["A", "B", "C"])

        optimized_durations, message = optimize_traffic(model_type, intersections)

        if not optimized_durations:
            return Response({"error": message}, status=400)

        print("Optimization:", message)
        return Response({"optimized_durations": optimized_durations})


class CongestionClusterAPIView(APIView):
    def post(self, request):
        df = load_static_data()
        result_df = detect_congestion_clusters(df)
        result = result_df.tail(10).to_dict(orient="records")
        return Response(result, status=status.HTTP_200_OK)


class RealTimeProcessorAPIView(APIView):
    def post(self, request):
        df = load_static_data()
        chunk_size = 500
        averages = []

        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start+chunk_size]
            avg_volume = chunk['traffic_volume'].mean()
            averages.append(avg_volume)

        overall_avg = sum(averages) / len(averages)
        return Response({"overall_avg_traffic_volume": round(overall_avg, 2)}, status=status.HTTP_200_OK)

@api_view(["GET"])
def run_simulation(request):
    results = simulate_realtime_from_chunks()
    return Response({
        "message": "✅ Simulation completed.",
        "chunks_processed": len(results),
        "results": results[-5:],  # send last 5 chunks for preview
    })
