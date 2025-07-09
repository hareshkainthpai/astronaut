from typing import List, Dict

from llm_dashboard.services.vector_service.token_aware_vector import TokenAwareVectorService


class MapReduceService:
    def __init__(self, model: 'LLMModel', vllm_service):
        self.model = model
        self.vllm_service = vllm_service
        self.vector_service = TokenAwareVectorService(model)

    def process_document_with_map_reduce(self, document_id: str, query: str,
                                         max_tokens: int = None) -> Dict:
        """Process document using map-reduce approach"""

        if max_tokens is None:
            max_tokens = self.model.get_available_context_tokens()

        # Get chunks organized in batches
        batches_info = self.vector_service.map_reduce_strategy(
            document_id, query, max_tokens
        )

        if not batches_info['chunks']:
            return {'error': 'No chunks found for document'}

        # Map phase: Process each batch
        map_results = []
        for batch in batches_info['chunks']:
            map_prompt = self._create_map_prompt(query, batch['content'])

            # Process with LLM
            result = self.vllm_service.generate_text(
                model=self.model,
                prompt=map_prompt,
                max_tokens=min(500, max_tokens // 4),  # Limit summary length
                temperature=0.3  # Lower temperature for factual summaries
            )

            map_results.append({
                'batch_index': batch['batch_index'],
                'original_content': batch['content'],
                'summary': result.get('response', ''),
                'tokens_used': result.get('tokens_generated', 0),
                'chunks_count': batch['chunks_count']
            })

        # Reduce phase: Combine summaries
        reduce_result = self._reduce_phase(query, map_results, max_tokens)

        return {
            'strategy': 'map_reduce',
            'map_results': map_results,
            'reduce_result': reduce_result,
            'total_batches': len(map_results),
            'total_tokens_used': sum(r['tokens_used'] for r in map_results) +
                                 reduce_result.get('tokens_used', 0)
        }

    def _create_map_prompt(self, query: str, content: str) -> str:
        """Create prompt for map phase"""
        return f"""Please analyze the following content and provide a concise summary that is relevant to this question: "{query}"

Content to analyze:
{content}

Instructions:
1. Focus on information that directly relates to the question
2. Preserve important facts, names, dates, and numbers
3. Keep the summary concise but informative
4. If the content doesn't relate to the question, state that clearly

Summary:"""

    def _reduce_phase(self, query: str, map_results: List[Dict], max_tokens: int) -> Dict:
        """Combine map results into final answer"""

        # Combine all summaries
        combined_summaries = []
        for i, result in enumerate(map_results):
            if result['summary'].strip():
                combined_summaries.append(f"Summary {i+1}: {result['summary']}")

        if not combined_summaries:
            return {'error': 'No relevant summaries generated'}

        # Create reduce prompt
        reduce_prompt = f"""Based on the following summaries, please provide a comprehensive answer to the question: "{query}"

Summaries:
{chr(10).join(combined_summaries)}

Instructions:
1. Synthesize information from all relevant summaries
2. Provide a clear, coherent answer
3. Cite which summary sections support your answer
4. If there are contradictions, note them
5. If you cannot answer based on the summaries, state that clearly

Answer:"""

        # Generate final answer
        result = self.vllm_service.generate_text(
            model=self.model,
            prompt=reduce_prompt,
            max_tokens=min(1000, max_tokens // 2),
            temperature=0.5
        )

        return {
            'final_answer': result.get('response', ''),
            'tokens_used': result.get('tokens_generated', 0),
            'summaries_processed': len(combined_summaries),
            'reduce_prompt': reduce_prompt
        }