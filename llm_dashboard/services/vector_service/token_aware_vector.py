import tiktoken
from typing import List, Dict
import json

from llm_dashboard.models import LLMModel, Document, DocumentChunk
from llm_dashboard.services.vector_service.document_vector import DocumentVectorStoreService


class TokenAwareVectorService(DocumentVectorStoreService):
    def __init__(self, model: LLMModel):
        super().__init__(model)
        self.tokenizer = model.get_tokenizer()
        self.max_context_tokens = model.get_available_context_tokens()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model's tokenizer"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback: rough estimation
            return len(text.split()) * 1.3

    def chunk_document_with_tokens(self, document: Document,
                                   target_chunk_tokens: int = 300,
                                   overlap_tokens: int = 50) -> List[Dict]:
        """Chunk document with precise token control"""
        content = document.content
        tokens = self.tokenizer.encode(content)

        chunks = []
        start_pos = 0
        chunk_index = 0

        while start_pos < len(tokens):
            # Determine chunk end position
            end_pos = min(start_pos + target_chunk_tokens, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_pos:end_pos]
            chunk_content = self.tokenizer.decode(chunk_tokens)

            # Create chunk info
            chunk_info = {
                'index': chunk_index,
                'content': chunk_content,
                'token_count': len(chunk_tokens),
                'token_start': start_pos,
                'token_end': end_pos,
                'overlap_start': max(0, start_pos - overlap_tokens),
                'overlap_end': min(len(tokens), end_pos + overlap_tokens)
            }

            chunks.append(chunk_info)

            # Move to next chunk with overlap
            start_pos = end_pos - overlap_tokens
            chunk_index += 1

            if start_pos >= len(tokens):
                break

        return chunks

    def get_chunks_within_token_limit(self, document_id: str, query: str,
                                      max_tokens: int, strategy: str = 'sliding_window') -> Dict:
        """Get chunks within token limit using specified strategy"""

        if strategy == 'sliding_window':
            return self._sliding_window_strategy(document_id, query, max_tokens)
        elif strategy == 'map_reduce':
            return self._map_reduce_strategy(document_id, query, max_tokens)
        elif strategy == 'hybrid':
            return self._hybrid_strategy(document_id, query, max_tokens)
        else:
            return self._direct_strategy(document_id, query, max_tokens)

    def _sliding_window_strategy(self, document_id: str, query: str, max_tokens: int) -> Dict:
        """Sliding window approach - dynamically adjust window size"""

        # Get all chunks for the document
        all_chunks = self.get_document_chunks_by_id(document_id)

        if not all_chunks:
            return {'chunks': [], 'strategy': 'sliding_window', 'total_tokens': 0}

        # Search for most relevant chunks
        relevant_chunks = self.search_document_chunks(document_id, query, k=len(all_chunks))

        # Build sliding window
        selected_chunks = []
        current_tokens = 0

        # Reserve tokens for query and formatting
        query_tokens = self.count_tokens(query)
        formatting_tokens = 100  # Estimate for formatting
        available_tokens = max_tokens - query_tokens - formatting_tokens

        # Start with most relevant chunks
        for chunk in relevant_chunks:
            chunk_tokens = self.count_tokens(chunk['content'])

            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit partial chunk if possible
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 50:  # Minimum meaningful chunk size
                    truncated_content = self._truncate_to_tokens(
                        chunk['content'], remaining_tokens
                    )
                    chunk_copy = chunk.copy()
                    chunk_copy['content'] = truncated_content
                    chunk_copy['truncated'] = True
                    selected_chunks.append(chunk_copy)
                    current_tokens += remaining_tokens
                break

        return {
            'chunks': selected_chunks,
            'strategy': 'sliding_window',
            'total_tokens': current_tokens,
            'chunks_used': len(selected_chunks),
            'chunks_truncated': sum(1 for c in selected_chunks if c.get('truncated', False))
        }

    def _map_reduce_strategy(self, document_id: str, query: str, max_tokens: int) -> Dict:
        """Map-reduce approach - process chunks in batches and summarize"""

        # Get all chunks for the document
        all_chunks = self.get_document_chunks_by_id(document_id)

        if not all_chunks:
            return {'chunks': [], 'strategy': 'map_reduce', 'total_tokens': 0}

        # Search for relevant chunks
        relevant_chunks = self.search_document_chunks(document_id, query, k=len(all_chunks))

        # Group chunks into batches that fit within token limits
        batches = self._create_token_batches(relevant_chunks, max_tokens // 3)  # Use 1/3 for each batch

        # Process each batch (this would need LLM calls)
        map_results = []
        for i, batch in enumerate(batches):
            batch_content = "\n\n".join([chunk['content'] for chunk in batch])
            batch_tokens = self.count_tokens(batch_content)

            map_results.append({
                'batch_index': i,
                'content': batch_content,
                'tokens': batch_tokens,
                'chunks_count': len(batch),
                'summary': None  # Would be filled by LLM processing
            })

        return {
            'chunks': map_results,
            'strategy': 'map_reduce',
            'total_tokens': sum(r['tokens'] for r in map_results),
            'batches_count': len(batches),
            'requires_llm_processing': True
        }

    def _hybrid_strategy(self, document_id: str, query: str, max_tokens: int) -> Dict:
        """Hybrid approach - combine sliding window with map-reduce"""

        # First try sliding window
        sliding_result = self._sliding_window_strategy(document_id, query, max_tokens // 2)

        # If we have remaining tokens, try map-reduce for additional context
        remaining_tokens = max_tokens - sliding_result['total_tokens']

        if remaining_tokens > 200:  # Minimum for meaningful map-reduce
            # Get chunks not already included
            used_chunk_ids = {chunk.get('chunk_id') for chunk in sliding_result['chunks']}
            all_chunks = self.get_document_chunks_by_id(document_id)
            remaining_chunks = [c for c in all_chunks if c.get('chunk_id') not in used_chunk_ids]

            if remaining_chunks:
                # Create a small map-reduce summary
                summary_batches = self._create_token_batches(remaining_chunks, remaining_tokens)

                return {
                    'primary_chunks': sliding_result['chunks'],
                    'summary_batches': summary_batches[:1],  # Take only first batch
                    'strategy': 'hybrid',
                    'total_tokens': sliding_result['total_tokens'] +
                                    (summary_batches[0]['tokens'] if summary_batches else 0),
                    'primary_strategy': 'sliding_window',
                    'secondary_strategy': 'map_reduce'
                }

        return sliding_result

    def _direct_strategy(self, document_id: str, query: str, max_tokens: int) -> Dict:
        """Direct approach - just get top chunks within limit"""
        return self._sliding_window_strategy(document_id, query, max_tokens)

    def _create_token_batches(self, chunks: List[Dict], max_tokens_per_batch: int) -> List[Dict]:
        """Group chunks into batches within token limits"""
        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk['content'])

            if current_tokens + chunk_tokens <= max_tokens_per_batch:
                current_batch.append(chunk)
                current_tokens += chunk_tokens
            else:
                if current_batch:
                    batches.append({
                        'chunks': current_batch,
                        'tokens': current_tokens,
                        'content': "\n\n".join([c['content'] for c in current_batch])
                    })

                current_batch = [chunk]
                current_tokens = chunk_tokens

        if current_batch:
            batches.append({
                'chunks': current_batch,
                'tokens': current_tokens,
                'content': "\n\n".join([c['content'] for c in current_batch])
            })

        return batches

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def add_document_to_vector_store(self, document: Document,
                                     target_chunk_tokens: int = 300,
                                     overlap_tokens: int = 50) -> bool:
        """Add document with token-aware chunking"""
        try:
            if not self.embedding_model:
                self.initialize_embedding_model()

            # Delete existing chunks
            DocumentChunk.objects.filter(document=document).delete()

            # Create token-aware chunks
            chunks_info = self.chunk_document_with_tokens(
                document, target_chunk_tokens, overlap_tokens
            )

            if not chunks_info:
                return False

            # Generate embeddings
            chunk_contents = [chunk['content'] for chunk in chunks_info]
            embeddings = self.embedding_model.encode(chunk_contents)

            # Load or create vector store
            if not self.vector_store:
                self.load_or_create_vector_store()

            # Add to vector store
            start_index = self.vector_store.ntotal
            self.vector_store.add(embeddings.astype('float32'))

            # Create DocumentChunk objects
            chunk_objects = []
            for i, chunk_info in enumerate(chunks_info):
                chunk_obj = DocumentChunk(
                    document=document,
                    chunk_index=chunk_info['index'],
                    content=chunk_info['content'],
                    vector_index=start_index + i,
                    token_count=chunk_info['token_count'],
                    token_start_position=chunk_info['token_start'],
                    token_end_position=chunk_info['token_end']
                )
                chunk_objects.append(chunk_obj)

            # Bulk create
            DocumentChunk.objects.bulk_create(chunk_objects)

            # Update document token count
            document.estimated_tokens = sum(chunk['token_count'] for chunk in chunks_info)
            document.is_indexed = True
            document.save()

            self._save_vector_store()
            return True

        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            return False