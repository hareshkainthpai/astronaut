from typing import List, Dict

from llm_dashboard.models import LLMModel, Document, DocumentChunk
from llm_dashboard.services.vector_service.document_vector import DocumentVectorStoreService


class TokenAwareVectorService(DocumentVectorStoreService):
    """
    TokenAwareVectorService provides functionality for chunking and token-aware processing
    of documents using a language model. It supports multiple strategies for managing
    tokens within a limit, including sliding window, map-reduce, and hybrid approaches.
    The service also integrates token counting and chunk splitting efficiently.

    :ivar tokenizer: The tokenizer derived from the provided language model, used for encoding
        and decoding text into tokens.
    :type tokenizer: Any
    :ivar max_context_tokens: The maximum number of context tokens available in the language model.
    :type max_context_tokens: int
    """
    def __init__(self, model: LLMModel):
        """
        Initializes an instance of the class.

        :param model: The language model object that provides the tokenizer and the
                      maximum number of available context tokens.
        :type model: LLMModel
        """
        super().__init__(model)
        self.tokenizer = model.get_tokenizer()
        self.max_context_tokens = model.get_available_context_tokens()

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens present in the given text. Tokenization is
        performed using the `self.tokenizer.encode` method. If tokenization
        fails for any reason, a fallback method is used, which provides a
        rough estimation based on splitting the text and applying a multiplier.

        :param text: The input text string to be tokenized.
        :type text: str

        :return: The estimated or actual number of tokens in the input text.
        :rtype: int
        """
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback: rough estimation
            return len(text.split()) * 1.3

    def chunk_document_with_tokens(self, document: Document,
                                   target_chunk_tokens: int = 300,
                                   overlap_tokens: int = 50) -> List[Dict]:
        """
        Divides the content of a document into smaller chunks based on the number of tokens,
        ensuring overlapping regions between chunks to maintain continuity.

        This method tokenizes the document content, splits the tokenized data into chunks of
        the specified size, and adds overlap to adjacent chunks. Each chunk includes metadata
        such as its position within the document, token count, and token range.

        :param document: A document instance whose content is to be chunked.
        :type document: Document
        :param target_chunk_tokens: The maximum number of tokens per chunk. Defaults to 300.
        :type target_chunk_tokens: int
        :param overlap_tokens: The number of tokens that should overlap between adjacent chunks.
            Defaults to 50.
        :type overlap_tokens: int
        :return: A list of dictionaries, where each dictionary contains chunk metadata and content.
        :rtype: List[Dict]
        """
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
        """
        Retrieve chunks of data within specified token limits, using the selected strategy.

        This function allows you to process a document by dividing it into chunks
        that fit within a specified token limit. Depending on the strategy provided,
        the division of the chunks is handled differently. Supported strategies
        include 'sliding_window', 'map_reduce', 'hybrid', and 'direct'.

        :param document_id: The unique identifier of the document to be processed.
        :type document_id: str
        :param query: The query string to apply while processing the document.
        :type query: str
        :param max_tokens: The maximum number of tokens allowed per chunk.
        :type max_tokens: int
        :param strategy: The strategy for chunking the document. Can be one of
            'sliding_window', 'map_reduce', 'hybrid', or 'direct'. Default is 'sliding_window'.
        :type strategy: str
        :return: A dictionary containing the chunks created based on the selected strategy
            within the specified token limits.
        :rtype: Dict
        """

        if strategy == 'sliding_window':
            return self._sliding_window_strategy(document_id, query, max_tokens)
        elif strategy == 'map_reduce':
            return self.map_reduce_strategy(document_id, query, max_tokens)
        elif strategy == 'hybrid':
            return self._hybrid_strategy(document_id, query, max_tokens)
        else:
            return self._direct_strategy(document_id, query, max_tokens)

    def _sliding_window_strategy(self, document_id: str, query: str, max_tokens: int) -> Dict:
        """
        Apply the sliding window strategy to retrieve and organize relevant document chunks
        based on a given query. This function ensures that the token usage does not exceed
        a specified maximum token limit, accounting for the query and formatting tokens.
        Chunks are prioritized by relevance, and partially truncated if necessary to fit
        within the remaining token budget.

        :param document_id: The unique identifier of the document from which chunks
            are retrieved.
        :type document_id: str

        :param query: The query string used to determine the relevance of document
            chunks.
        :type query: str

        :param max_tokens: The maximum number of tokens allowed in the returned set
            of chunks, including tokens reserved for the query and formatting.
        :type max_tokens: int

        :return: A dictionary containing the selected document chunks, the applied
            strategy, total tokens used, number of chunks used, and a count of
            truncated chunks.
        :rtype: Dict
        """

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

    def map_reduce_strategy(self, document_id: str, query: str, max_tokens: int) -> Dict:
        """
        Summarizes and processes document chunks using a map-reduce strategy.

        This method retrieves the chunks of a document, identifies relevant ones based
        on a query, groups them into batches that adhere to a token limit, and prepares
        them for further processing. Each batch is represented with its content,
        computed token count, the number of chunks it contains, and a placeholder for
        its summary.

        :param document_id: The unique identifier of the document to be processed.
        :type document_id: str
        :param query: The query used to search for relevant chunks in the document.
        :type query: str
        :param max_tokens: The maximum token limit that affects how batches are created.
        :type max_tokens: int
        :return: A dictionary containing the processed chunk details, map-reduce strategy
            metadata, total token usage, the number of created batches, and a flag
            indicating if further LLM processing is required.
        :rtype: Dict
        """

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
        """
        Combines the sliding window strategy with a map-reduce approach for generating
        summaries or extracting meaningful information from document chunks. This hybrid
        method attempts to maximize the utilization of available tokens by first applying
        a sliding window strategy and, if tokens remain, adds summaries using map-reduce
        from unused document chunks.

        This approach ensures that primary context is preserved while adding secondary
        information in a practical and efficient manner. The method returns a detailed
        results dictionary containing primary processed chunks, optional summary batches,
        and metadata about the strategies applied.

        :param document_id: The unique identifier of the document being processed.
        :param query: A textual query related to the extraction or summarization operation.
        :param max_tokens: An integer representing the maximum token capacity for processing.
        :return: A dictionary containing processed document chunks, optional summary
                 batches, primary and secondary strategies used, and the total number of tokens utilized.
        """

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
        """
        Executes the direct strategy for processing a document and obtaining results.

        The function processes a document using a direct sliding window strategy. It takes
        a document identifier, a query to process, and a maximum token limit as input. The
        results of the operation are returned as a dictionary.

        :param document_id: The unique identifier of the document to be processed.
        :type document_id: str
        :param query: The query string to process with the document.
        :type query: str
        :param max_tokens: The maximum number of tokens allowed during processing.
        :type max_tokens: int
        :return: A dictionary containing the results of the direct strategy processing.
        :rtype: Dict
        """
        return self._sliding_window_strategy(document_id, query, max_tokens)

    def _create_token_batches(self, chunks: List[Dict], max_tokens_per_batch: int) -> List[Dict]:
        """
        Create batches of content chunks based on the maximum allowed token count per batch.

        This method takes a list of content chunks and groups them into batches such that the
        total token count in each batch does not exceed the specified maximum tokens. Each
        batch contains the corresponding chunks, the total token count, and the combined content
        as a single string.

        :param chunks: A list of dictionaries, each representing a content chunk. Each dictionary
            must have a 'content' key containing the text content.
        :type chunks: List[Dict]
        :param max_tokens_per_batch: The maximum number of tokens allowed per batch.
        :type max_tokens_per_batch: int
        :return: A list of dictionaries where each dictionary represents a batch. Each dictionary
            contains the keys 'chunks' (list of chunks in the batch), 'tokens' (total token count
            for the batch), and 'content' (the combined content as a single string).
        :rtype: List[Dict]
        """
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
        """
        Truncates a given text to a maximum number of tokens based on the tokenizer.

        This method uses the tokenizer to encode the input text into tokens. If the number of tokens
        exceeds the specified `max_tokens`, the tokens are truncated to the allowed token limit.
        The truncated tokens are then decoded back into text using the same tokenizer.

        :param text: The input text to truncate.
        :type text: str
        :param max_tokens: The maximum number of tokens allowed in the truncated output.
        :type max_tokens: int
        :return: The truncated text that does not exceed the specified token limit.
        :rtype: str
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def add_document_to_vector_store(self, document: Document,
                                     target_chunk_tokens: int = 300,
                                     overlap_tokens: int = 50) -> bool:
        """
        Adds a given document to the vector store. This involves several steps, including tokenizing
        the document into chunks, generating embeddings for each chunk, and storing the chunks along
        with their respective vectors in the vector store. It ensures the document is properly indexed
        and updates its metadata accordingly.

        :param document: The Document instance that needs to be added to the vector store.
        :param target_chunk_tokens: The target number of tokens per chunk. Default is 300.
        :param overlap_tokens: The number of overlapping tokens between consecutive chunks. Default is 50.
        :return: A boolean indicating whether the document was successfully added to the vector store.
        """
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