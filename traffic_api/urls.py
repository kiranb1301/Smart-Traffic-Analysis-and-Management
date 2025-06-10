from django.urls import path
from .views import (
    index,
    TrafficRecordListCreate,
    CSVUploadView,
    PredictTrafficAPIView,
    OptimizeTrafficAPIView,
    CongestionClusterAPIView,
    RealTimeProcessorAPIView,
    run_simulation
)

urlpatterns = [
    path('', index, name='index'),
    path('api/traffic/', TrafficRecordListCreate.as_view(), name='traffic-data'),
    path('api/upload/', CSVUploadView.as_view(), name='upload-csv'),
    path('api/predict/', PredictTrafficAPIView.as_view(), name='predict-traffic'),
    path('api/optimize/', OptimizeTrafficAPIView.as_view(), name='optimize-traffic'),
    path('api/congestion/', CongestionClusterAPIView.as_view(), name='congestion-detection'),
    path('api/realtime/process/', RealTimeProcessorAPIView.as_view(), name='realtime-processing'),
    path("api/run-simulation/", run_simulation, name="run_simulation"),
]
