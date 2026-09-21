from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('eda/', views.eda_view, name='eda'),
    path('train/', views.train_view, name='train'),
    path('train/status/', views.train_status, name='train_status'),
    path('predict/', views.predict_view, name='predict'),
    path('compare/', views.compare_view, name='compare'),
]
