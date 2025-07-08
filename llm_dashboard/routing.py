from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # FIXED: More specific regex pattern to avoid matching non-numeric values
    re_path(r'^ws/model-loading/(?P<model_id>[1-9]\d*)/$', consumers.ModelLoadingConsumer.as_asgi()),
    re_path(r'^ws/llm-stream/$', consumers.LLMStreamConsumer.as_asgi()),
]