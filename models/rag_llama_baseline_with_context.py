# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper for RAGModel that captures contexts used during generation for RAGAS evaluation.
"""

from typing import Any, Dict, List
from models.rag_llama_baseline import RAGModel


class RAGModelWithContext(RAGModel):
    """
    Extended RAGModel that stores the contexts used during answer generation.
    This is useful for RAGAS evaluation which requires the actual contexts used.
    """
    
    def __init__(self):
        super().__init__()
        self.last_batch_contexts = None  # Store contexts from last batch_generate_answer call
    
    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers and stores the contexts used for RAGAS evaluation.
        
        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries.
        
        Returns:
            List[str]: A list of plain text responses for each query in the batch.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            from models.rag_llama_baseline import NUM_CONTEXT_SENTENCES
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            
            batch_retrieval_results.append(retrieval_results)
        
        # Store contexts for RAGAS evaluation
        # Convert numpy array to list of strings for each query
        self.last_batch_contexts = [
            "\n".join(results.tolist() if hasattr(results, 'tolist') else results)
            for results in batch_retrieval_results
        ]
        
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            self._get_sampling_params(),
            use_tqdm=False
        )

        # Aggregate answers into List[str]
        answers = []
        for response in responses:
            answers.append(response.outputs[0].text)
        
        return answers
    
    def get_last_batch_contexts(self) -> List[str]:
        """
        Returns the contexts used in the last batch_generate_answer call.
        
        Returns:
            List[str]: List of context strings, one per query in the last batch.
        """
        return self.last_batch_contexts if self.last_batch_contexts else []
    
    def _get_sampling_params(self):
        """Helper method to get sampling parameters (extracted for clarity)."""
        import vllm
        return vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0.1,  # Randomness of the sampling
            skip_special_tokens=True,  # Whether to skip special tokens in the output.
            max_tokens=50,  # Maximum number of tokens to generate per output sequence.
        )

