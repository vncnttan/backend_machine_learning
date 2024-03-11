from django.urls import path
from RestAPI.views import Predict

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("predict/", Predict.as_view(), name="predict")
]