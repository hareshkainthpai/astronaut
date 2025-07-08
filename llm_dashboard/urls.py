from django.urls import path
from . import views

app_name = 'llm_dashboard'

urlpatterns = [
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('load-model/', views.load_model, name='load_model'),
    path('unload-model/<int:model_id>/', views.unload_model, name='unload_model'),
    path('generate/', views.generate_text, name='generate_text'),
    path('test-model/', views.test_model, name='test_model'),
    path('stats/', views.model_stats, name='model_stats'),
    path('browse-directories/', views.browse_directories, name='browse_directories'),
    path('stop-loading/<int:model_id>/', views.stop_loading, name='stop_loading'),
    path('get-logs/<int:model_id>/', views.get_loading_logs, name='get_loading_logs'),
    path('debug/', views.debug_model_status, name='debug_model_status'),
    path('api/model/<int:model_id>/logs/', views.get_model_logs, name='get_model_logs'),
    path('api/model/<int:model_id>/clear-logs/', views.clear_model_logs, name='clear_model_logs'),
    path('force-cleanup-gpus/', views.force_cleanup_all_gpus, name='force_cleanup_gpus'),
]