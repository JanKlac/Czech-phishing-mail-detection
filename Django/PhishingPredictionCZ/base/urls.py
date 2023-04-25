from django.urls import path
from . import views
  
urlpatterns = [
    path("", views.home, name="home"),
    path("modellabel/", views.modellabel, name="modellabel"),
    path("modellabel/result/", views.result, name="getPrediction"),
    path('modeltext/', views.image_to_text, name='modeltext'),
]