# CRAG: Comprehensive RAG Benchmark

The Comprehensive RAG Benchmark (CRAG) is a rich and comprehensive factual question answering benchmark designed to advance research in RAG. Besides question-answer pairs, CRAG provides mock APIs to simulate web and knowledge graph search. CRAG is designed to encapsulate a diverse array of questions across five domains and eight question categories, reflecting varied entity popularity from popular to long-tail, and temporal dynamisms ranging from years to seconds.

This repository is migrated from [meta-comprehensive-rag-benchmark-kdd-cup-2024](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024).

## 📊 Dataset and Mock APIs

Please find more details about the CRAG dataset (download, schema, etc.) in [docs/dataset.md](docs/dataset.md) and mock APIs in [mock_api](mock_api).


## 📏 Evaluation Metrics
RAG systems are evaluated using a scoring method that measures response quality to questions in the evaluation set. Responses are rated as perfect, acceptable, missing, or incorrect:

- Perfect: The response correctly answers the user question and contains no hallucinated content.

- Acceptable: The response provides a useful answer to the user question, but may contain minor errors that do not harm the usefulness of the answer.

- Missing: The answer does not provide the requested information. Such as “I don’t know”, “I’m sorry I can’t find …” or similar sentences without providing a concrete answer to the question.

- Incorrect: The response provides wrong or irrelevant information to answer the user question


Auto-evaluation: 
- Automatic evaluation employs rule-based matching and LLM assessment to check answer correctness. It will assign three scores: correct (1 point), missing (0 points), and incorrect (-1 point).


Please refer to [local_evaluation.py](local_evaluation.py) for more details on how the evaluation was implemented.

## ✍️ How to run end-to-end evaluation?
1. **Install** specific dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models.

3. After writing your own model(s), update [models/user_config.py](models/user_config.py)

   For example, in models/user_config.py, specify InstructModel to call llama3-8b-instruct model
   ```bash
   from models.vanilla_llama_baseline import InstructModel 
   UserModel = InstructModel

   ```

4. Test your model locally using `python local_evaluation.py`. This script will run answer generation and auto-evaluation.

### Using LM Studio or Ollama for Evaluation

By default, evaluation uses OpenAI's API. To use LM Studio or Ollama for evaluation instead (free, local):

1. **Start LM Studio or Ollama**:
   - **LM Studio**: Load a model and enable Local Server (usually runs on port 1234)
   - **Ollama**: Run `ollama serve` and pull a model (e.g., `ollama pull llama3`)

2. **Set environment variables**:
   ```bash
   # For LM Studio
   export EVALUATION_API_BASE=http://localhost:1234/v1
   export EVALUATION_API_KEY=lm-studio
   export EVALUATION_MODEL_NAME=llama3  # or your model name
   
   # For Ollama
   export EVALUATION_API_BASE=http://localhost:11434/v1
   export EVALUATION_API_KEY=ollama
   export EVALUATION_MODEL_NAME=llama3
   ```

3. **Run evaluation**:
   ```bash
   python local_evaluation.py
   ```

   If `EVALUATION_API_BASE` is not set, the script will use the default OpenAI API (requires API key).


## 🏁 Baselines
We include three baselines for demonstration purposes, and you can read more about them in [docs/baselines.md](docs/baselines.md).

## 🔌 OpenAI-Compatible Models
We also provide OpenAI-compatible model implementations that work with Ollama, LM Studio, and other OpenAI-compatible APIs. These model allow you to run CRAG evaluations without requiring local GPU resources for model inference:

- **[RAG OpenAI-Compatible Model](docs/rag_openai_compatible_model.md)**: Full RAG pipeline with semantic search using OpenAI-compatible APIs

See the [RAG OpenAI-Compatible Model documentation](docs/rag_openai_compatible_model.md) for setup instructions and usage examples.

**Note**: Both answer generation and evaluation can use LM Studio/Ollama. See the "Using LM Studio or Ollama for Evaluation" section above for evaluation setup.


## Citations

```
@article{yang2024crag,
      title={CRAG -- Comprehensive RAG Benchmark}, 
      author={Xiao Yang and Kai Sun and Hao Xin and Yushi Sun and Nikita Bhalla and Xiangsen Chen and Sajal Choudhary and Rongze Daniel Gui and Ziran Will Jiang and Ziyu Jiang and Lingkun Kong and Brian Moran and Jiaqi Wang and Yifan Ethan Xu and An Yan and Chenyu Yang and Eting Yuan and Hanwen Zha and Nan Tang and Lei Chen and Nicolas Scheffer and Yue Liu and Nirav Shah and Rakesh Wanga and Anuj Kumar and Wen-tau Yih and Xin Luna Dong},
      year={2024},
      journal={arXiv preprint arXiv:2406.04744},
      url={https://arxiv.org/abs/2406.04744}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
