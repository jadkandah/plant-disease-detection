from django.urls import path
from . import views

urlpatterns = [
    path('', views.disease_list, name='disease-list'),
    path('<int:pk>/', views.disease_detail, name='disease-detail'),
]