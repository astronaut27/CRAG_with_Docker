# RAG OpenAI-Compatible Model

The RAG OpenAI-Compatible Model is an implementation of Retrieval-Augmented Generation (RAG) that uses OpenAI-compatible APIs (such as Ollama or LM Studio) instead of local vLLM models. This allows you to run CRAG evaluations without requiring GPU resources for model inference, making it ideal for development and testing scenarios.

## Overview

This model implements a full RAG pipeline:

1. **Chunk Extraction**: Extracts text chunks (sentences) from HTML search results using parallel processing
2. **Embedding Generation**: Computes semantic embeddings for both chunks and queries using SentenceTransformer
3. **Semantic Search**: Performs cosine similarity search to find the most relevant chunks for each query
4. **Context Retrieval**: Selects top-N most relevant chunks (default: 20) based on semantic similarity
5. **Answer Generation**: Formats prompts with retrieved context and generates answers via OpenAI-compatible API

## Key Features

- **OpenAI-Compatible API Support**: Works with any OpenAI-compatible API endpoint (Ollama, LM Studio, etc.)
- **Semantic Search**: Uses embeddings to find the most relevant context chunks, not just keyword matching
- **Parallel Processing**: Leverages Ray for parallel chunk extraction from HTML sources
- **No Local Model Weights Required**: Only needs the API endpoint, no need to download large model files
- **Flexible Configuration**: Easy to configure via environment variables

## Architecture

```
Query → Search Results (HTML)
    ↓
ChunkExtractor → Text Chunks (sentences)
    ↓
SentenceTransformer → Embeddings (chunks + queries)
    ↓
Cosine Similarity → Top-N Relevant Chunks
    ↓
Prompt Formatting → OpenAI API → Answer
```

## Configuration

The model can be configured using environment variables:

### Required Environment Variables

- `RAG_MODEL_API_BASE`: Base URL for the OpenAI-compatible API (for answer generation)
  - For Ollama: `http://localhost:11434/v1`
  - For LM Studio: `http://localhost:1234/v1`
- `RAG_MODEL_API_KEY`: API key (Ollama doesn't require a real key, use "ollama")
- `RAG_MODEL_NAME`: Name of the model in your API server (e.g., "llama3", "llama2", etc.)

### Optional Configuration

- `CRAG_MOCK_API_URL`: URL of the CRAG mock API (default: `http://localhost:8000`)

### Model Parameters (in code)

- `NUM_CONTEXT_SENTENCES`: Number of context sentences to retrieve (default: 20)
- `MAX_CONTEXT_SENTENCE_LENGTH`: Maximum length of each context sentence in characters (default: 1000)
- `MAX_CONTEXT_REFERENCES_LENGTH`: Maximum total length of references in characters (default: 4000)
- `SUBMISSION_BATCH_SIZE`: Batch size for processing queries (default: 8)
- `SENTENTENCE_TRANSFORMER_BATCH_SIZE`: Batch size for embedding model (default: 128)

## Setup Instructions

### Using with Ollama

1. **Install and start Ollama**:
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama serve
   ```

2. **Pull a model**:
   ```bash
   ollama pull llama3
   # or
   ollama pull llama2
   ```

3. **Set environment variables**:
   ```bash
   export RAG_MODEL_API_BASE=http://localhost:11434/v1
   export RAG_MODEL_API_KEY=ollama
   export RAG_MODEL_NAME=llama3
   ```

4. **Update `models/user_config.py`**:
   ```python
   from models.rag_openai_compatible_model import RAGOpenAICompatibleModel
   UserModel = RAGOpenAICompatibleModel
   ```

### Using with LM Studio

1. **Install and start LM Studio** (see https://lmstudio.ai)

2. **Load a model** in LM Studio

3. **Enable Local Server** in LM Studio (usually runs on port 1234)

4. **Set environment variables**:
   ```bash
   export RAG_MODEL_API_BASE=http://localhost:1234/v1
   export RAG_MODEL_API_KEY=lm-studio  # or any value
   export RAG_MODEL_NAME=your-model-name
   ```

5. **Update `models/user_config.py`** (same as above)

## Usage

After configuration, you can run evaluations as usual:

```bash
python local_evaluation.py
```

The model will:
1. Receive batches of queries with search results
2. Extract and process chunks from HTML
3. Compute embeddings and perform semantic search
4. Generate answers using the configured OpenAI-compatible API

## Implementation Details

### ChunkExtractor

The `ChunkExtractor` class handles HTML parsing and chunk extraction:

- Uses BeautifulSoup to parse HTML and extract text
- Uses blingfire's `text_to_sentences_and_offsets` to split text into sentences (optional: fallback to simple regex-based splitting if blingfire is not available or incompatible with your architecture)
- Processes chunks in parallel using Ray for better performance
- De-duplicates chunks within each interaction

### Embedding Model

The model uses `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings:

- Lightweight and fast
- Generates 384-dimensional embeddings
- Normalized embeddings for cosine similarity computation
- Can run on CPU or GPU (automatically detects CUDA)

### Semantic Search

The retrieval process:

1. Computes embeddings for all chunks and queries
2. Filters chunks by `interaction_id` to ensure relevance
3. Calculates cosine similarity between query and chunk embeddings
4. Selects top-N chunks with highest similarity scores

### Answer Generation

Answers are generated using the OpenAI chat completions API:

- Uses system and user messages format
- Includes retrieved context as references
- Limits response to 75 tokens (using 50 max_tokens for Llama3 efficiency)
- Trims responses using tokenizer if available

## Comparison with Other Models

### vs. `rag_llama_baseline.py`

| Feature | RAG Llama Baseline | RAG OpenAI-Compatible |
|---------|-------------------|----------------------|
| Model Inference | Local vLLM | OpenAI-compatible API |
| GPU Required | Yes (for inference) | No (only for embeddings) |
| Model Weights | Required locally | Not required |
| Setup Complexity | High | Low |
| Performance | Fast (local) | Depends on API |
| Flexibility | Fixed model | Any compatible model |

### vs. `openai_compatible_model.py`

| Feature | OpenAI-Compatible | RAG OpenAI-Compatible |
|---------|-------------------|----------------------|
| RAG Pipeline | No | Yes |
| Semantic Search | No | Yes |
| Embeddings | No | Yes (SentenceTransformer) |
| Chunk Selection | Simple (first N) | Smart (top-N by similarity) |
| Quality | Lower | Higher |
| Speed | Faster | Slower (due to embeddings) |

## Troubleshooting

### Connection Errors

If you see connection errors:
- Ensure Ollama/LM Studio is running
- Check the `RAG_MODEL_API_BASE` URL is correct
- Verify the model name in `RAG_MODEL_NAME` matches your loaded model

### Slow Performance

If generation is slow:
- Reduce `SUBMISSION_BATCH_SIZE` to process fewer queries at once
- Reduce `NUM_CONTEXT_SENTENCES` to retrieve fewer chunks
- Use GPU for embeddings if available (automatically detected)

### Tokenizer Warnings

If you see tokenizer warnings:
- The model will still work, but token trimming may not be accurate
- Ensure the tokenizer directory exists or the model can be downloaded from HuggingFace

### Blingfire Architecture Issues (Apple Silicon)

If you encounter `OSError: incompatible architecture` with `blingfire`:
- This is expected on Apple Silicon (ARM64) when `blingfire` is installed for x86_64
- The model automatically falls back to a simple regex-based sentence splitting method
- The fallback works on all architectures and doesn't require `blingfire`
- For best performance, you can try installing `blingfire` from source or use Docker (which may have better architecture support)

## References

- Implementation: [`models/rag_openai_compatible_model.py`](../models/rag_openai_compatible_model.py)
- OpenAI Python SDK: https://github.com/openai/openai-python
- Ollama: https://ollama.ai
- LM Studio: https://lmstudio.ai
- Sentence Transformers: https://www.sbert.net/

