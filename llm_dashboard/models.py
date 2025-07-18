from django.db import models
import json
import os
import uuid
import tiktoken
from django.utils import timezone


class LLMModel(models.Model):
    """
    Represents a machine learning model with various configurations and attributes for
    loading, execution, embedding support, and token management.

    The class serves as a representation of an LLM (Large Language Model) instance and
    includes essential information such as model path, loading status, supported device,
    and configuration settings for tokens, embeddings, and retrieval-augmented generation
    (RAG) strategy.

    :ivar loading_pid: Process ID of the loading operation.
    :type loading_pid: int, optional
    :ivar loading_logs: Logs generated during the model loading process.
    :type loading_logs: str
    :ivar name: Name of the model instance.
    :type name: str
    :ivar model_path: Absolute path where the model files are stored locally.
    :type model_path: str, optional
    :ivar loaded_at: Timestamp when the model was last loaded.
    :type loaded_at: datetime.datetime
    :ivar max_tokens: Maximum number of tokens supported by the model.
    :type max_tokens: int
    :ivar memory_usage: Memory usage of the model in gigabytes (GB).
    :type memory_usage: float
    :ivar device: Device type on which the model is executed (e.g., CPU, GPU).
    :type device: str
    :ivar status: Current loading or operational status of the model.
    :type status: str
    :ivar vllm_port: Port used for serving the model.
    :type vllm_port: int
    :ivar tensor_parallel_size: Number of GPUs used for tensor parallelization.
    :type tensor_parallel_size: int
    :ivar gpu_memory_utilization: Utilization ratio of GPU memory.
    :type gpu_memory_utilization: float
    :ivar model_type: Type or architecture of the model (e.g., 'llama').
    :type model_type: str
    :ivar dtype: Data type for model weights.
    :type dtype: str
    :ivar supports_embeddings: Indicates whether the model supports embeddings.
    :type supports_embeddings: bool
    :ivar embedding_model_path: Path to the embedding model files.
    :type embedding_model_path: str, optional
    :ivar vector_store_path: Path to the vector store.
    :type vector_store_path: str, optional
    :ivar vector_store_type: Type of the vector store used (e.g., FAISS, ChromaDB).
    :type vector_store_type: str
    :ivar max_context_tokens: Maximum number of tokens allowed in the model's context window.
    :type max_context_tokens: int
    :ivar reserve_tokens_for_response: Number of tokens reserved for generating responses.
    :type reserve_tokens_for_response: int
    :ivar tokenizer_name: Name of the tokenizer to be used for token management.
    :type tokenizer_name: str
    :ivar rag_strategy: Strategy used for retrieval-augmented generation (RAG),
        such as sliding window, map-reduce, or hybrid.
    :type rag_strategy: str
    """
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
    """
    Represents a document that can be associated with a language model. This class stores the
    document's content, metadata, and related attributes while also tracking key information like
    indexing status and token-related information.

    The purpose of this class is to facilitate the integration of documents into machine learning
    workflows, especially in scenarios involving language models. It provides fields covering
    document metadata, chunking strategies, and optional file information.

    :ivar id: Unique identifier for the document.
    :type id: UUID
    :ivar model: Foreign key relation to the `LLMModel`, defining which model the document belongs to.
    :type model: ForeignKey
    :ivar title: Title of the document.
    :type title: str
    :ivar content: Full content of the document.
    :type content: str
    :ivar metadata: JSON field storing additional metadata for the document.
    :type metadata: dict
    :ivar created_at: Timestamp of when the document was created.
    :type created_at: datetime
    :ivar updated_at: Timestamp of the last modification of the document.
    :type updated_at: datetime
    :ivar is_indexed: Indicates whether the document is indexed in a vector store.
    :type is_indexed: bool
    :ivar estimated_tokens: An estimation of the total tokens in the document.
    :type estimated_tokens: int
    :ivar chunk_strategy: Defines the strategy for chunking the document.
    :type chunk_strategy: str
    :ivar filename: Name of the file associated with the document (if applicable).
    :type filename: str, optional
    :ivar file_size: Size of the file associated with the document in bytes (if applicable).
    :type file_size: int, optional
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE, related_name='documents', null=True,
                              blank=True)  # Made optional
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

    # Add fields for when model is null
    filename = models.CharField(max_length=255, null=True, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.title} ({self.id})"

    @property
    def is_global(self):
        """Check if this is a global document (not tied to specific model)"""
        return self.model is None



class DocumentChunk(models.Model):
    """
    Represents a chunk of text content within a document.

    A DocumentChunk instance models a section of a document and contains
    information about the chunk's position, its associated content, token
    management details, and metadata for tracking and indexing. This class
    is particularly useful for splitting documents into manageable segments
    and associating additional data like embeddings for analysis.

    :ivar document: The document that this chunk is associated with.
    :type document: Document
    :ivar chunk_index: The order of the chunk within the document.
    :type chunk_index: int
    :ivar content: The text content of the chunk.
    :type content: str
    :ivar vector_index: The index of the chunk in the vector store.
    :type vector_index: int
    :ivar embedding_hash: A hash of the embedding associated with the chunk
        for verification purposes.
    :type embedding_hash: str, optional
    :ivar created_at: The timestamp indicating when the chunk was created.
    :type created_at: datetime
    :ivar token_count: The number of tokens in this chunk.
    :type token_count: int
    :ivar token_start_position: The starting position of the chunk in the
        original document tokens.
    :type token_start_position: int
    :ivar token_end_position: The ending position of the chunk in the
        original document tokens.
    :type token_end_position: int
    """
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
    """
    Represents a request made to an LLM (Large Language Model) system for processing
    a given prompt, capturing the response generated, and tracking detailed metrics
    related to the request.

    This class is intended to store information about requests sent to an LLM,
    including contextual details, processing metadata, and specific settings
    or configurations used during the request. It also tracks strategies used for
    retrieval-augmented generation (RAG) and can log any relevant processing details.

    :ivar model: The LLMModel associated with the request. A ForeignKey relation to
                 the LLMModel.
    :ivar created_at: The date and time the request was created.
    :ivar prompt: The input text prompt sent to the LLM.
    :ivar response: The output text response generated by the LLM.
    :ivar duration: The time taken, in seconds, for the request to be processed.
    :ivar tokens_generated: The number of tokens generated as the response.
    :ivar tokens_prompt: The number of tokens in the input prompt.
    :ivar status: The status of the request, including 'PENDING', 'COMPLETED',
                  or 'ERROR'.
    :ivar error_message: A detailed error message in case the status is 'ERROR'.
    :ivar temperature: Controls the randomness of the LLM's output.
    :ivar top_p: Specifies nucleus sampling for controlling diversity of output.
    :ivar max_tokens: The maximum number of tokens allowed in the response.
    :ivar document_id: A UUID representing a document ID, relevant to RAG scenarios.
    :ivar rag_enabled: A flag indicating whether retrieval-augmented generation
                       (RAG) was enabled.
    :ivar context_chunks_used: JSON storage for tracking which context chunks
                               were employed in the request.
    :ivar total_context_tokens: The total number of tokens in the context.
    :ivar strategy_used: Indicates the strategy employed for processing requests,
                         such as 'direct' or others.
    :ivar chunks_processed: The total number of context chunks processed in the
                            request.
    :ivar map_reduce_steps: JSON storage for tracking map-reduce steps in processing.
    :ivar logs: Detailed logs of the request's processing, including timestamps,
                messages, or other debug data.
    """
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