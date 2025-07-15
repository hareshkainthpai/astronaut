import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
import hashlib
from django.conf import settings
from django.db.models import Q

from llm_dashboard.models import LLMModel, Document, DocumentChunk


class DocumentVectorStoreService:
    def __init__(self, model: Optional[LLMModel] = None):
        self.model = model
        self.embedding_model = None
        self.vector_store = None
        self.chunk_size = 500  # Default chunk size
        self.chunk_overlap = 50  # Default overlap between chunks

    def initialize_embedding_model(self):
        """Initialize the embedding model"""
        if self.model and self.model.embedding_model_path:
            self.embedding_model = SentenceTransformer(self.model.embedding_model_path)
        else:
            # Default to a good embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk_document(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split document into chunks with overlap"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def add_document_to_vector_store(self, document: Document, chunk_size: int = None, overlap: int = None) -> bool:
        """Add a document to the vector store by chunking it"""
        try:
            if not self.embedding_model:
                self.initialize_embedding_model()

            # Delete existing chunks for this document
            DocumentChunk.objects.filter(document=document).delete()

            # Chunk the document
            chunks = self.chunk_document(document.content, chunk_size, overlap)

            if not chunks:
                return False

            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(chunks)

            # Load or create vector store
            if not self.vector_store:
                self.load_or_create_vector_store()

            # Add embeddings to vector store
            start_index = self.vector_store.ntotal
            self.vector_store.add(embeddings.astype('float32'))

            # Create DocumentChunk objects
            chunk_objects = []
            for i, chunk_content in enumerate(chunks):
                # Create embedding hash for verification
                embedding_hash = hashlib.sha256(embeddings[i].tobytes()).hexdigest()

                chunk_obj = DocumentChunk(
                    document=document,
                    chunk_index=i,
                    content=chunk_content,
                    vector_index=start_index + i,
                    embedding_hash=embedding_hash
                )
                chunk_objects.append(chunk_obj)

            # Bulk create chunks
            DocumentChunk.objects.bulk_create(chunk_objects)

            # Mark document as indexed
            document.is_indexed = True
            document.save()

            # Save vector store
            self._save_vector_store()

            return True

        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            return False

    def search_document_chunks(self, document_id: str, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks within a specific document"""
        try:
            if not self.vector_store or not self.embedding_model:
                self.load_or_create_vector_store()
                if not self.vector_store:
                    return []

            # Get document chunks - handle both model-specific and global documents
            if self.model:
                # For model-specific service, search both model documents and global documents
                document_chunks = DocumentChunk.objects.filter(
                    document__id=document_id
                ).filter(
                    models.Q(document__model=self.model) | models.Q(document__model__isnull=True)
                ).order_by('chunk_index')
            else:
                # For global service, only search global documents
                document_chunks = DocumentChunk.objects.filter(
                    document__id=document_id,
                    document__model__isnull=True
                ).order_by('chunk_index')

            if not document_chunks.exists():
                return []

            # Get vector indices for this document's chunks
            vector_indices = [chunk.vector_index for chunk in document_chunks]

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Search in vector store
            distances, indices = self.vector_store.search(query_embedding.astype('float32'), k * 2)

            # Filter results to only include chunks from the specified document
            filtered_results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx in vector_indices:
                    # Find the corresponding chunk
                    chunk = document_chunks.get(vector_index=idx)
                    filtered_results.append({
                        'chunk_id': chunk.id,
                        'chunk_index': chunk.chunk_index,
                        'content': chunk.content,
                        'document_id': str(chunk.document.id),
                        'document_title': chunk.document.title,
                        'similarity_score': float(1 / (1 + distance)),
                        'vector_index': idx,
                        'is_global': chunk.document.is_global
                    })

                if len(filtered_results) >= k:
                    break

            return filtered_results

        except Exception as e:
            print(f"Error searching document chunks: {e}")
            return []

    def get_document_chunks_by_id(self, document_id: str) -> List[Dict]:
        """Get all chunks for a specific document ordered by chunk_index"""
        try:
            # Handle both model-specific and global documents
            if self.model:
                chunks = DocumentChunk.objects.filter(
                    document__id=document_id
                ).filter(
                    Q(document__model=self.model) | Q(document__model__isnull=True)
                ).select_related('document').order_by('chunk_index')
            else:
                chunks = DocumentChunk.objects.filter(
                    document__id=document_id,
                    document__model__isnull=True
                ).select_related('document').order_by('chunk_index')

            return [
                {
                    'chunk_id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'document_id': str(chunk.document.id),
                    'document_title': chunk.document.title,
                    'vector_index': chunk.vector_index,
                    'is_global': chunk.document.is_global
                }
                for chunk in chunks
            ]

        except Exception as e:
            print(f"Error getting document chunks: {e}")
            return []

    def load_or_create_vector_store(self):
        """Load existing vector store or create new one"""
        if not self.embedding_model:
            self.initialize_embedding_model()

        vector_store_path = self._get_vector_store_path()

        if os.path.exists(f"{vector_store_path}.index"):
            # Load existing vector store
            self.vector_store = faiss.read_index(f"{vector_store_path}.index")
        else:
            # Create new vector store
            # Get embedding dimension from model
            sample_embedding = self.embedding_model.encode(["test"])
            dimension = sample_embedding.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self._save_vector_store()

    def _get_vector_store_path(self) -> str:
        """Get the path for vector store files"""
        if self.model:
            # Model-specific path
            if self.model.vector_store_path:
                return self.model.vector_store_path
            else:
                vector_dir = os.path.join(settings.BASE_DIR, 'vector_stores')
                os.makedirs(vector_dir, exist_ok=True)
                return os.path.join(vector_dir, f"model_{self.model.id}")
        else:
            # Global vector store path
            vector_dir = os.path.join(settings.BASE_DIR, 'vector_stores')
            os.makedirs(vector_dir, exist_ok=True)
            return os.path.join(vector_dir, "global_vector_store")

    def _save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store:
            vector_store_path = self._get_vector_store_path()
            faiss.write_index(self.vector_store, f"{vector_store_path}.index")