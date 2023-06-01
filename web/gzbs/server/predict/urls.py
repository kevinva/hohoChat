from django.urls import path, include
from predict import views

urlpatterns = [
    path("fatigue", views.CUBatchFatiguePredictionView.as_view(), name = "fatigue"),
]
