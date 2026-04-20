# AI Research Assistant: Local RAG Pipeline

A Retrieval-Augmented Generation (RAG) system developed to facilitate technical analysis of academic research papers. This project utilizes LangChain for orchestration and Ollama for local Large Language Model (LLM) inference.

## Technical Architecture

The system implements a standard RAG architecture with specific optimizations for academic document structures:

* **Orchestration Framework:** LangChain (RetrievalQA)
* **Inference Engine:** Ollama (Llama 3)
* **Vector Database:** ChromaDB (Persistent)
* **Embedding Model:** Ollama Embeddings (Llama 3)
* **Document Processing:** PyPDFLoader

## Key Implementation Details

### Context Preservation
To maintain the integrity of complex technical arguments and data tables, a RecursiveCharacterTextSplitter was configured with a chunk size of 2000 characters and a 300-character overlap. This prevents the loss of context across page breaks and section headers.

### Retrieval Optimization
The system utilizes Maximum Marginal Relevance (MMR) for document retrieval. This ensures that the context provided to the LLM is both relevant and diverse, effectively mitigating the "Bibliography Trap" where citation lists can dominate search results based on keyword density.

### Grounding and Prompt Engineering
A custom prompt template is utilized to ground the model in the provided context. The instructions prioritize research findings and explicitly direct the model to disregard bibliographic noise and acknowledge information gaps to prevent hallucinations.

## Installation and Usage

### Prerequisites
1. Install [Ollama](https://ollama.com).
2. Pull the required model: `ollama pull llama3`.

### Setup
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Create a directory named data in the root of the project
4. Place your target PDF research papers inside the /data directory. The system is configured to automatically ingest and index all PDF files within this folder