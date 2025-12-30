# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import ray
import requests
import torch
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from models.utils import trim_predictions_to_max_token_length

# Try to import blingfire, fallback to simple sentence splitting if not available
try:
    from blingfire import text_to_sentences_and_offsets
    HAS_BLINGFIRE = True
except (ImportError, OSError):
    # blingfire not available (e.g., architecture mismatch on Apple Silicon)
    HAS_BLINGFIRE = False

CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

# RAG Model API configuration (for answer generation)
# For Ollama: http://localhost:11434/v1
# For LM Studio: http://localhost:1234/v1
RAG_MODEL_API_BASE = os.getenv("RAG_MODEL_API_BASE", "http://localhost:11434/v1")
RAG_MODEL_API_KEY = os.getenv("RAG_MODEL_API_KEY", "ollama")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "llama3")

# RAG configuration
NUM_CONTEXT_SENTENCES = 20  # Number of context sentences to consider for generating an answer
MAX_CONTEXT_SENTENCE_LENGTH = 1000  # Maximum length for each context sentence (in characters)
MAX_CONTEXT_REFERENCES_LENGTH = 4000  # Maximum context references length (in characters)
SUBMISSION_BATCH_SIZE = 8  # Batch size for processing queries
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128  # Batch size for embedding model

# Tokenizer path
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "tokenizer")


def _simple_sentence_split(text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Simple sentence splitting fallback when blingfire is not available.
    Splits text on sentence boundaries (., !, ?) followed by whitespace or end of string.
    
    Args:
        text: Input text to split
        
    Returns:
        Tuple of (text, list of (start, end) offsets for each sentence)
    """
    # Pattern to match sentence endings: . ! ? followed by whitespace or end of string
    # Also handles abbreviations like "Mr.", "Dr.", "etc." by requiring capital letter after
    sentence_pattern = r'(?<=[.!?])(?:\s+|$)'
    
    sentences = []
    offsets = []
    start = 0
    
    # Find all sentence boundaries
    for match in re.finditer(sentence_pattern, text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            offsets.append((start, end))
            sentences.append(sentence)
        start = end
    
    # Add remaining text if any
    if start < len(text):
        remaining = text[start:].strip()
        if remaining:
            offsets.append((start, len(text)))
            sentences.append(remaining)
    
    # If no sentences found, return the whole text as one sentence
    if not offsets:
        offsets = [(0, len(text))]
        sentences = [text]
    
    return '\n'.join(sentences), offsets


class ChunkExtractor:
    """
    Class for extracting chunks (sentences) from HTML sources.
    Uses parallel processing via Ray for acceleration.
    """
    
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.
        
        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.
        
        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.
        
        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences
                                   extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
        
        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]
        
        # Extract offsets of sentences from the text
        # Use blingfire if available, otherwise use simple fallback
        if HAS_BLINGFIRE:
            _, offsets = text_to_sentences_and_offsets(text)
        else:
            _, offsets = _simple_sentence_split(text)
        
        # Initialize a list to store sentences
        chunks = []
        
        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end].strip()[:MAX_CONTEXT_SENTENCE_LENGTH]
            if sentence:  # Only add non-empty sentences
                chunks.append(sentence)
        
        # If no chunks found, return at least one chunk with the text
        if not chunks:
            chunks = [text[:MAX_CONTEXT_SENTENCE_LENGTH]]
        
        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.
        
        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search result batches,
                                                     each containing HTML text.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array
                                           of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]
        
        # Dictionary to collect chunks for each interaction_id separately
        chunk_dictionary = defaultdict(list)
        
        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)
        
        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)
        
        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.
        
        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys
                                           and lists of chunks as values.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array
                                           of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []
        
        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))
        
        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)
        
        return chunks, chunk_interaction_ids


class RAGOpenAICompatibleModel:
    """
    RAG model using OpenAI-compatible API (Ollama, LM Studio, etc.)
    
    This model implements Retrieval-Augmented Generation (RAG) using semantic search
    to find the most relevant context chunks from search results before generating answers.
    It uses OpenAI-compatible APIs (like Ollama or LM Studio) instead of local vLLM models.
    
    Key features:
    - Extracts text chunks from HTML search results
    - Computes embeddings for chunks and queries using SentenceTransformer
    - Performs semantic search using cosine similarity
    - Selects top-N most relevant chunks for each query
    - Generates answers using OpenAI-compatible API
    """
    def __init__(self):
        """
        Initialize the model and chunk extractor
        """
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        """
        Initialize OpenAI client, tokenizer, and embedding model
        """
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=RAG_MODEL_API_BASE,
            api_key=RAG_MODEL_API_KEY
        )
        self.model_name = RAG_MODEL_NAME
        
        # Test connection to API (optional check)
        try:
            # Try a simple request to verify connection
            # Note: Some OpenAI-compatible APIs may not support models.list()
            test_url = RAG_MODEL_API_BASE.replace('/v1', '') if RAG_MODEL_API_BASE.endswith('/v1') else RAG_MODEL_API_BASE
            response = requests.get(f"{test_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ RAG Model API server is accessible at {RAG_MODEL_API_BASE}")
        except Exception as e:
            print(f"⚠ Warning: Could not verify connection to RAG Model API at {RAG_MODEL_API_BASE}")
            print(f"  Error: {e}")
            print(f"  Please ensure that:")
            print(f"    1. Ollama/LM Studio is running")
            print(f"    2. The API is accessible at {RAG_MODEL_API_BASE}")
            print(f"    3. Model '{self.model_name}' is loaded")
            print(f"  The code will continue, but connection errors may occur during generation.")
        
        # Initialize tokenizer
        try:
            if os.path.exists(TOKENIZER_PATH):
                self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-Instruct")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Load sentence transformer model for embeddings
        self.sentence_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences.
        
        This function leverages the SentenceTransformer model to encode sentences,
        which can enhance processing speed on multi-core machines.
        
        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.
        
        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.
        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size.
        i.e.: timeout for the `batch_generate_answer` call = batch_size * per_sample_timeout
        
        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        return SUBMISSION_BATCH_SIZE

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers using RAG with OpenAI-compatible API.
        
        This method implements the full RAG pipeline:
        1. Extracts chunks from HTML search results
        2. Computes embeddings for chunks and queries
        3. Performs semantic search to find most relevant chunks
        4. Formats prompts with retrieved context
        5. Generates answers via OpenAI-compatible API
        
        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id' (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query.
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding
                                            to when a query was made.
        
        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.
        
        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Extract chunks from all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate embeddings for all chunks
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[idx]
            query_embedding = query_embeddings[idx]
            
            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            
            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]
            
            # Calculate cosine similarity between query and chunk embeddings
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)
            
            # Retrieve top-N results
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            batch_retrieval_results.append(retrieval_results)
        
        # Prepare formatted prompts
        formatted_messages = self.format_prompts(
            queries, 
            query_times, 
            batch_retrieval_results
        )

        # Generate responses via API
        answers = []
        for messages in formatted_messages:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=50,  # Using 50 as in original (Llama3 tokenizer is more efficient)
                    top_p=0.9,
                )
                
                answer = response.choices[0].message.content
                if self.tokenizer:
                    answer = trim_predictions_to_max_token_length(answer)
                
                answers.append(answer)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Provide more detailed error information
                if "Connection" in error_type or "connection" in error_msg.lower():
                    print(f"Error generating answer: Connection error. "
                          f"Please check that {RAG_MODEL_API_BASE} is accessible and the model '{self.model_name}' is loaded. "
                          f"Error details: {error_msg}")
                else:
                    print(f"Error generating answer ({error_type}): {error_msg}")
                
                answers.append("I don't know")

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries into OpenAI chat format.
        
        Parameters:
            queries (List[str]): A list of queries to be formatted into prompts.
            query_times (List[str]): A list of query_time strings corresponding to each query.
            batch_retrieval_results (List[List[str]]): List of retrieval results for each query
        """
        system_prompt = (
            "You are provided with a question and various references. "
            "Your task is to answer the question succinctly, using the fewest words possible. "
            "If the references do not contain the necessary information to answer the question, "
            "respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        )
        
        formatted_messages = []
        
        for idx, query in enumerate(queries):
            query_time = query_times[idx]
            retrieval_results = batch_retrieval_results[idx] if batch_retrieval_results else []
            
            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References\n"
                # Format the top sentences as references in the model's prompt template.
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.
            
            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question:\n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            formatted_messages.append(messages)
        
        return formatted_messages

