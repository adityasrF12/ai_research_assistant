# AI Research Assistant: Containerized RAG API

An industry-standard Retrieval-Augmented Generation (RAG) system developed to facilitate technical analysis of academic research papers. This project has evolved from a basic script into a scalable, Dockerized FastAPI service optimized for local LLM inference.

## Technical Architecture

The system implements a production-ready RAG architecture with a focus on modularity and high-performance retrieval:

* **API Framework:** FastAPI (Asynchronous request handling)
* **Orchestration:** LangChain (RetrievalQA chain)
* **Inference Engine:** Ollama (Llama 3)
* **Vector Database:** ChromaDB (Persistent storage layer)
* **Infrastructure:** Docker & Docker Compose (Containerized deployment)
* **Document Processing:** PyPDFLoader with Recursive Character Splitting



## Key Features

* **Containerized Environment:** Fully Dockerized setup ensuring consistent deployment across different environments.
* **Context Preservation:** Utilizes `RecursiveCharacterTextSplitter` (chunk size: 2000, overlap: 300) to maintain technical context across complex PDF structures.
* **Database Management:** Includes a dedicated `/clear` endpoint to programmatically reset and re-initialize the vector database.
* **Source Transparency:** The API response includes specific metadata, citing the exact file name, page number, and text snippet used to generate the answer.

## Installation & Setup

### Prerequisites
1. Install [Ollama](https://ollama.com).
2. Pull the required model: `ollama pull llama3`.
3. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### Deployment
1. Clone the repository and navigate to the root directory.
2. Place your target PDF research papers inside the `/data` directory.
3. Launch the service using Docker Compose:
   ```bash
   docker-compose up --build