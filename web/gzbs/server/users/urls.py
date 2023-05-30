from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from users import views


urlpatterns = [
    path("token/obtain", TokenObtainPairView.as_view(), name = "token_obtain"),
    path("detail", views.DetailsView.as_view(), name = "detail"),
    path("login", views.LoginView.as_view(), name = "login"),
]