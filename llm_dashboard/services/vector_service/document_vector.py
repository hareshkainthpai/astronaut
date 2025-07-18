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
        """
        Initializes an instance of the class with the given model and default attribute values.

        :param model: Optional instance of an LLMModel to be used. If not provided, defaults to None.
        :type model: Optional[LLMModel]
        """
        self.model = model
        self.embedding_model = None
        self.vector_store = None
        self.chunk_size = 500  # Default chunk size
        self.chunk_overlap = 50  # Default overlap between chunks

    def initialize_embedding_model(self):
        """
        Initializes the embedding model for generating sentence embeddings.

        This method sets up the `embedding_model` by either using the path specified in
        `self.model.embedding_model_path` or defaults to a commonly used pre-trained
        model ('all-MiniLM-L6-v2') if no path is provided.

        :raises RuntimeError: If the `SentenceTransformer` fails to load the model.
        :return: None
        """
        if self.model and self.model.embedding_model_path:
            self.embedding_model = SentenceTransformer(self.model.embedding_model_path)
        else:
            # Default to a good embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk_document(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Chunks a given text into smaller segments of specified size. If the text length
        is smaller than or equal to the defined chunk size, the function returns the
        text as a single chunk. For longer texts, it attempts to chunk the document by
        utilizing sentence boundaries for improved segmentation. Overlap between
        consecutive chunks can also be configured.

        :param text: The input text to be chunked.
        :type text: str
        :param chunk_size: Maximum size of each chunk. If not provided, the default
            chunk size is used.
        :type chunk_size: int, optional
        :param overlap: The amount of overlap between consecutive chunks. If not
            provided, the default overlap is used.
        :type overlap: int, optional
        :return: A list containing the segmented text chunks.
        :rtype: List[str]
        """
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
        """
        Adds a given document to the vector store after processing it into chunks and generating
        embeddings for each chunk. This method ensures the document is indexed and properly stored
        for vector-based searches or operations.

        :param document: The document to be added to the vector store.
        :type document: Document
        :param chunk_size: Optional parameter to specify the size of each chunk, in number of characters.
                           If None, a default size will be used.
        :type chunk_size: int, optional
        :param overlap: Optional parameter specifying the overlap between chunks, in number of characters.
                        If None, no overlap is added.
        :type overlap: int, optional
        :return: True if the document was successfully added to the vector store, False otherwise.
        :rtype: bool
        """
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
        """
        Search document chunks based on a query using an embedding model and a vector store.

        This function retrieves chunks of a document specified by `document_id`, generates an
        embedding for the given `query`, and performs a similarity search in the vector store to
        find the most relevant chunks. Global and model-specific filters are applied when fetching
        document chunks. The results are filtered and ranked based on similarity scores.

        :param document_id: The unique identifier of the document whose chunks are to be searched.
        :type document_id: str
        :param query: The query text to be used for similarity search.
        :type query: str
        :param k: Number of top relevant chunks to return.
        :type k: int
        :return: A list of dictionaries containing details of the relevant document chunks, including
            chunk ID, content, document details, similarity score, and other metadata.
        :rtype: List[Dict]

        """
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
                    Q(document__model=self.model) | Q(document__model__isnull=True)
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
        """
        Retrieve all chunks of a document by its ID.

        This function fetches the relevant chunks of a document based on the provided
        document ID. It handles cases where the document is either model-specific or
        global. The chunks are retrieved, and their metadata along with content are
        organized into a list of dictionaries for easier consumption.

        :param document_id: The unique identifier of the document whose chunks should
            be fetched.
        :type document_id: str
        :return: A list of dictionaries where each dictionary represents a chunk of
            the document, including metadata such as the chunk ID, chunk index,
            content, document ID, document title, vector index, and whether the
            document is global.
        :rtype: List[Dict]
        """
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
        """
        Loads an existing vector store or creates a new one if it does not exist.

        The function first ensures that the embedding model is initialized. It then checks
        if the vector store file exists at the designated path. If it exists, the vector store
        is loaded. Otherwise, a new vector store is created based on the embedding model's
        configuration, and it is saved for future use.

        :param self: The instance of the class containing this function. The instance should
                     provide the necessary methods and attributes to initialize the embedding
                     model and handle persistence of the vector store.

        :raises FileNotFoundError: If the vector store path cannot be found or created.

        """
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
        """
        Generates and returns the path to the vector store directory. The path
        is dynamically determined based on whether a specific model is associated
        or a global vector store is required.

        If a model is provided and it has a predefined vector store path, that path
        is returned. Otherwise, the function creates a directory to store the vector
        for the specific model, ensuring that the directory exists. If no model is
        provided, a global vector store directory is used or created as required.

        :returns: The file path to the vector store.
        :rtype: str
        """
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
        """
        Saves the current vector store to the disk.

        This method persists the in-memory vector store if available by writing it
        to a file on disk using FAISS. The file name will include a ".index"
        extension.

        :raises RuntimeError: If the vector store cannot be saved due to an issue
            with the FAISS library or file path generation.
        """
        if self.vector_store:
            vector_store_path = self._get_vector_store_path()
            faiss.write_index(self.vector_store, f"{vector_store_path}.index")