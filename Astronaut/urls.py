from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('llm-dashboard/', include('llm_dashboard.urls')),
]