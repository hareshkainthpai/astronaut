from django.db import models
import json
import os
import uuid
import tiktoken


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

    # Vector store configuration
    supports_embeddings = models.BooleanField(default=False)
    embedding_model_path = models.CharField(max_length=500, null=True, blank=True)
    vector_store_path = models.CharField(max_length=500, null=True, blank=True)
    vector_store_type = models.CharField(max_length=50, default='faiss', choices=[
        ('faiss', 'FAISS'),
        ('chromadb', 'ChromaDB'),
        ('weaviate', 'Weaviate'),
    ])

    # Token management fields
    max_context_tokens = models.IntegerField(default=4096)  # Model's context window
    reserve_tokens_for_response = models.IntegerField(default=512)  # Reserve for response
    tokenizer_name = models.CharField(max_length=100, default='gpt-3.5-turbo')  # For token counting

    # RAG strategy configuration
    rag_strategy = models.CharField(max_length=20, default='sliding_window', choices=[
        ('sliding_window', 'Sliding Window'),
        ('map_reduce', 'Map-Reduce'),
        ('hybrid', 'Hybrid'),
    ])

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

    def get_available_context_tokens(self):
        """Get tokens available for context after reserving for response"""
        return self.max_context_tokens - self.reserve_tokens_for_response

    def get_tokenizer(self):
        """Get tokenizer for this model"""
        try:
            return tiktoken.encoding_for_model(self.tokenizer_name)
        except:
            return tiktoken.get_encoding("cl100k_base")  # Default fallback


class Document(models.Model):
    """Represents a document that can be chunked and stored in vector store"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=200)
    content = models.TextField()
    metadata = models.JSONField(default=dict)  # Store additional metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_indexed = models.BooleanField(default=False)  # Track if document is indexed in vector store

    # Token-related fields
    estimated_tokens = models.IntegerField(default=0)  # Estimated total tokens in document
    chunk_strategy = models.CharField(max_length=20, default='semantic', choices=[
        ('fixed_size', 'Fixed Size'),
        ('semantic', 'Semantic'),
        ('sliding', 'Sliding Window'),
    ])

    def __str__(self):
        return f"{self.title} ({self.id})"


class DocumentChunk(models.Model):
    """Represents a chunk of a document stored in vector store"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.IntegerField()  # Order of chunk in document
    content = models.TextField()
    vector_index = models.IntegerField()  # Index in the vector store
    embedding_hash = models.CharField(max_length=64, null=True, blank=True)  # Hash of embedding for verification
    created_at = models.DateTimeField(auto_now_add=True)

    # Token management
    token_count = models.IntegerField(default=0)
    token_start_position = models.IntegerField(default=0)  # Start position in original document tokens
    token_end_position = models.IntegerField(default=0)  # End position in original document tokens

    class Meta:
        unique_together = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
            models.Index(fields=['vector_index']),
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


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

    # RAG-specific fields
    document_id = models.UUIDField(null=True, blank=True)  # Document ID used for RAG
    rag_enabled = models.BooleanField(default=False)
    context_chunks_used = models.JSONField(default=list)  # Store which chunks were used

    # Token and strategy tracking
    total_context_tokens = models.IntegerField(default=0)
    strategy_used = models.CharField(max_length=20, default='direct')
    chunks_processed = models.IntegerField(default=0)
    map_reduce_steps = models.JSONField(default=list)  # Track map-reduce steps

    # New field for detailed logs
    logs = models.TextField(blank=True)  # Store detailed processing logs

    def set_response(self, response):
        self.response = response
        self.save()

    def add_stream_chunk(self, chunk):
        current = json.loads(self.response) if self.response else []
        current.append(chunk)
        self.response = json.dumps(current)
        self.save()

    def add_log(self, message):
        """Add a log entry with timestamp"""
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        self.logs = (self.logs or '') + log_entry
        self.save()

    def get_model_name(self):
        return self.model.name if self.model else 'Unknown'

    class Meta:
        ordering = ['-created_at']