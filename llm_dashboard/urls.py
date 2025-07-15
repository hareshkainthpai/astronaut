
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

    # GPU Stats API
    path('api/gpu-stats/', views.api_gpu_stats, name='api_gpu_stats'),

    # Request Details API
    path('api/request/<int:request_id>/details/', views.get_request_details, name='get_request_details'),
    path('api/request/<int:request_id>/export/', views.export_request_data, name='export_request_data'),

    # RAG endpoints (with model_id)
    path('api/models/<int:model_id>/generate-with-document-rag/', views.generate_with_document_rag, name='generate_with_document_rag'),
    path('api/models/<int:model_id>/documents/', views.list_documents, name='list_documents'),
    path('api/models/<int:model_id>/documents/add/', views.add_document_api, name='add_document_api'),
    path('api/models/<int:model_id>/documents/add-from-file/', views.add_document_from_file_path, name='add_document_from_file_path'),
    path('api/models/<int:model_id>/documents/<str:document_id>/chunks/', views.get_document_chunks, name='get_document_chunks'),
    path('api/models/<int:model_id>/documents/<str:document_id>/search/', views.search_document_chunks, name='search_document_chunks'),

    # Active model endpoints (no model_id required)
    path('api/active-model/generate/', views.generate_with_active_model, name='generate_with_active_model'),
    path('api/active-model/generate-with-document-rag/', views.generate_with_active_model_rag, name='generate_with_active_model_rag'),
    path('api/active-model/documents/', views.get_active_model_documents, name='get_active_model_documents'),
    path('api/active-model/documents/add-from-file/', views.add_document_from_file_path_active_model, name='add_document_from_file_path_active_model'),
    path('api/active-model/documents/search/', views.search_active_model_document_chunks, name='search_active_model_document_chunks'),

    path('api/refresh-model-status/', views.refresh_model_status, name='refresh_model_status'),

    # Streaming endpoints (no model_id required)
    path('api/generate-streaming/', views.generate_streaming_text, name='generate_streaming_text'),
    path('api/active-model/', views.get_active_model_status, name='get_active_model_status'),

# Global document endpoints (no model_id required)
    path('api/documents/global/add-from-file/', views.add_global_document_from_file_path,
         name='add_global_document_from_file_path'),
    path('api/documents/global/', views.list_global_documents, name='list_global_documents'),

]