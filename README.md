# AI Research Assistant: Containerized RAG API

A Retrieval-Augmented Generation (RAG) system for technical analysis of academic research papers. The project started as a local script using ChromaDB and has evolved into a Dockerized FastAPI service backed by a managed vector database, with a cross-encoder reranking layer added on top of retrieval for better answer quality.

## Technical Architecture

* **API Framework:** FastAPI (asynchronous request handling)
* **Orchestration:** LangChain
* **Inference Engine:** Ollama (Llama 3)
* **Vector Database:** Pinecone (serverless, cloud-hosted)
* **Retrieval Quality:** Base similarity search (top-20) re-ranked with a BGE cross-encoder down to the top-5 most relevant chunks per query
* **Infrastructure:** Docker & Docker Compose (containerized deployment)
* **Document Processing:** PyPDFLoader with recursive character splitting

### Why Pinecone over ChromaDB

The project's first phase used ChromaDB for local, persistent vector storage. It was moved to Pinecone to decouple the vector store from any single machine and support a cloud-hosted deployment. This surfaced a real constraint worth documenting: Pinecone's serverless indexes only support delete-by-ID, not delete-by-metadata-filter, which shaped how ingestion tracks and stores each chunk's vector ID (see `ingest.py`) so that individual documents can be cleanly removed later.

## Key Features

* **Containerized Backend:** The FastAPI service, retrieval pipeline, and reranker run in Docker for consistent deployment.
* **Context Preservation:** Uses `RecursiveCharacterTextSplitter` (chunk size: 1500, overlap: 150) to keep technical context intact across complex PDF structures.
* **Reranked Retrieval:** A cross-encoder reranking pass sits between vector search and the LLM, prioritizing the chunks most relevant to the specific question rather than relying on raw similarity scores alone.
* **Grounded Answers:** The API refuses to answer when retrieval returns nothing relevant, rather than falling back on the LLM's own pretrained knowledge -- so responses are traceable to the ingested papers, not general knowledge.
* **Per-Document Management:** PDFs can be ingested and deleted individually from the UI, with deletions properly removing that document's vectors from Pinecone.
* **Source Transparency:** API responses cite the exact file name, page number, and text snippet used to generate the answer.

## Installation & Setup

### Prerequisites
1. Install [Ollama](https://ollama.com).
2. Pull the required models: `ollama pull llama3` and `ollama pull nomic-embed-text`.
3. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
4. Create a free [Pinecone](https://www.pinecone.io) account and index, and get an API key.

### Configuration
1. Clone the repository and navigate to the root directory.
2. Copy `.env.example` to `.env` and fill in your Pinecone API key and index name.

### Deployment
1. Launch the backend using Docker Compose:
   ```bash
   docker compose up --build
   ```
2. In a separate terminal, run the UI:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Upload PDF research papers through the sidebar and click **Process & Ingest**.
