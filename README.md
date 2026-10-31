# LLM & RAG Evaluation and Observability Pipeline

## Project Overview

This project provides a pipeline for evaluating and monitoring Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) applications.

The project focuses on evaluating response quality, retrieval performance, hallucinations, uncertainty, explainability, bias, and security-related behaviour. It also includes observability components for tracing retrieved chunks, prompts, and sentence-level citation coverage.

## Key Features

- RAG response evaluation
- Answer and retrieval quality metrics
- Hallucination analysis
- Uncertainty analysis
- Chain-of-Thought evaluation
- SHAP and LIME-based explainability
- Bias analysis
- Prompt-injection detection
- Sentence-level citation coverage
- Retrieved chunk tracing and scoring
- Prompt snapshot and truncation information
- FastAPI-based API integration

## Project Components

### RAG Evaluation

The evaluation components analyse the quality of RAG responses and retrieval performance using configurable metrics.

### Hallucination and Uncertainty Analysis

The pipeline includes modules for analysing potentially unsupported responses and estimating uncertainty in generated outputs.

### Explainability

SHAP and LIME modules are included to support analysis of model behaviour and important features contributing to results.

### Security and Bias

The project includes components for prompt-injection detection and bias analysis to support safer and more reliable AI applications.

### Observability

The observability modules provide visibility into important stages of the RAG pipeline, including:

- Sentence-level citation coverage
- Retrieved chunk traces and scores
- Prompt snapshots
- Truncation information

## Technologies Used

- Python
- FastAPI
- Hugging Face
- LangChain
- Ollama
- Sentence Transformers
- SHAP
- LIME
- Pandas and NumPy
- PyTorch

## Project Structure

```text
LLM-RAG-Evaluation-Pipeline/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── rag_evaluation.py
│   ├── evaluation_metrics.py
│   ├── hallucination.py
│   ├── uncertainity.py
│   ├── shap.py
│   ├── lime.py
│   ├── bias.py
│   ├── prompt_injection.py
│   └── observability/
│
└── api/
    └── v1/
        └── rag_routes.py
```

## Running the Project

1. Clone the repository.
2. Create a Python environment.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the required model and application settings.
5. Run the FastAPI application:

```bash
uvicorn app:app --reload
```

## Important Note

Before running the project, configure the required models and environment settings. Do not commit API keys, access tokens, private datasets, or confidential information to the repository.

## Author

**Harshita Sharma**

AI/ML Engineer

GitHub: github.com/harshitasharma891  
LinkedIn: linkedin.com/in/harshita-sharma-b843031b5
