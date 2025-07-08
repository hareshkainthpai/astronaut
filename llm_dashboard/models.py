from django.db import models
import json
import os


class LLMModel(models.Model):
    loading_pid = models.IntegerField(null=True, blank=True)  # Track loading process PID
    loading_logs = models.TextField(blank=True)  # Store loading logs
    name = models.CharField(max_length=100)
    model_path = models.CharField(max_length=500, null=True)  # Path to local model files
    loaded_at = models.DateTimeField(auto_now_add=True)
    max_tokens = models.IntegerField(default=2048)
    memory_usage = models.FloatField(default=0.0)  # in GB
    device = models.CharField(max_length=50, default='GPU')  # CPU/GPU
    status = models.CharField(max_length=20, default='UNLOADED')  # LOADED, LOADING, ERROR, UNLOADED
    vllm_port = models.IntegerField(default=8000)
    tensor_parallel_size = models.IntegerField(default=1)  # Number of GPUs to use
    gpu_memory_utilization = models.FloatField(default=0.9)  # GPU memory utilization ratio
    model_type = models.CharField(max_length=50, default='llama')  # Model architecture type
    dtype = models.CharField(max_length=20, default='auto', choices=[
        ('auto', 'Auto'),
        ('float16', 'Float16'),
        ('bfloat16', 'BFloat16'),
        ('float32', 'Float32'),
    ])  # Data type for model weights

    def __str__(self):
        return f"{self.name} ({self.status})"

    def get_model_size_gb(self):
        """Calculate model size in GB"""
        if not os.path.exists(self.model_path):
            return 0.0

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)

        return total_size / (1024 ** 3)  # Convert to GB


class LLMRequest(models.Model):
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    prompt = models.TextField()
    response = models.TextField(blank=True)
    duration = models.FloatField(null=True)  # in seconds
    tokens_generated = models.IntegerField(default=0)
    tokens_prompt = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default='PENDING')  # PENDING, COMPLETED, ERROR
    error_message = models.TextField(blank=True)
    temperature = models.FloatField(default=0.7)
    top_p = models.FloatField(default=0.9)
    max_tokens = models.IntegerField(default=512)

    def set_response(self, response):
        self.response = response
        self.save()

    def add_stream_chunk(self, chunk):
        current = json.loads(self.response) if self.response else []
        current.append(chunk)
        self.response = json.dumps(current)
        self.save()