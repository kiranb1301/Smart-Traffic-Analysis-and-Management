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
    path('/smarttraffic/api/traffic/', TrafficRecordListCreate.as_view(), name='traffic-data'),
    path('/smarttraffic/api/upload/', CSVUploadView.as_view(), name='upload-csv'),
    path('/smarttraffic/api/predict/', PredictTrafficAPIView.as_view(), name='predict-traffic'),
    path('/smarttraffic/api/optimize/', OptimizeTrafficAPIView.as_view(), name='optimize-traffic'),
    path('/smarttraffic/api/congestion/', CongestionClusterAPIView.as_view(), name='congestion-detection'),
    path('/smarttraffic/api/realtime/process/', RealTimeProcessorAPIView.as_view(), name='realtime-processing'),
    path("/smarttraffic/api/run-simulation/", run_simulation, name="run_simulation"),
]
