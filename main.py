import os
import logging
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from ingest import run_ingestion


load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-API")

# FIRST: Initialize the FastAPI app object so endpoints can use it
app = FastAPI(title="Advanced AI Research Assistant")

# 1. Configuration & Environment Variables
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
MANIFEST_PATH = "./data/manifest.json"


def load_manifest():
    """Manifest is a dict: {filename: [vector_id, vector_id, ...]}"""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            data = json.load(f)
            # Backwards-compat with the old flat-list manifest format.
            if isinstance(data, list):
                return {name: [] for name in data}
            return data
    return {}


def write_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)


# 2. Initialize Embeddings and Pinecone
logger.info("Initializing Embeddings and connecting to Pinecone...")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_URL
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# 3. Setup Advanced Reranking (The Quality Layer)
logger.info("Loading Reranker Model (BGE-Reranker)...")
rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=rerank_model, top_n=5)

base_retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 4. LLM and Prompt Configuration
template = """You are a research assistant analyzing academic papers. Answer the question using ONLY the context blocks below -- do not use outside or pre-trained knowledge, even if you recognize the paper or topic.

The context is made up of excerpts from a PDF and may be incomplete or not perfectly worded for this question. That's expected: read all the excerpts together and synthesize the best answer you can from whatever relevant information they contain, rather than requiring one excerpt to directly and completely answer the question.

Only if, after reviewing all the excerpts, none of them relate to the question's topic at all, reply with exactly: "I don't know. The library is empty or no relevant context was found."

Context: {context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOllama(model="llama3", temperature=0, base_url=OLLAMA_URL)

NO_CONTEXT_MESSAGE = "I don't know. The library is empty or no relevant context was found."


class Query(BaseModel):
    text: str


# --- ENDPOINTS ---

@app.get("/")
def health_check():
    """Verify system health and active components."""
    return {
        "status": "Online",
        "engine": "Advanced RAG with Reranking",
        "vector_store": "Pinecone",
        "index": INDEX_NAME
    }


@app.get("/library")
def get_library():
    """Retrieve list of currently indexed files in the research assistant library."""
    manifest = load_manifest()
    return {"indexed_files": list(manifest.keys())}


@app.post("/ask")
async def ask_question(query: Query):
    """Process a user query using the reranked retrieval pipeline."""
    logger.info(f"Processing query with reranking: {query.text}")
    try:
        
        docs = compression_retriever.invoke(query.text)

        if not docs:
            return {"answer": NO_CONTEXT_MESSAGE, "sources": []}

        context_text = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = QA_CHAIN_PROMPT.format(
            context=context_text, question=query.text
        )
        result = llm.invoke(formatted_prompt)
        answer = getattr(result, "content", str(result))

        sources = [
            {
                "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": int(doc.metadata.get("page", 0)),
                "snippet": doc.page_content[:150] + "..."
            }
            for doc in docs
        ]

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"RAG Chain Error: {str(e)}")
        return {"error": "Processing failed", "details": str(e)}


@app.post("/ingest")
async def ingest_documents():
    """
    Scan the shared ./data directory for PDFs not yet in the manifest and
    ingest them into Pinecone. Runs the same logic as `python ingest.py`,
    but triggered over HTTP from inside this container -- so it always
    runs with the container's environment (correct OLLAMA_BASE_URL,
    Pinecone credentials, etc.) instead of depending on a local venv path.

    Ingestion is CPU/IO-bound and synchronous under the hood (PDF parsing,
    embeddings), so it's offloaded to a thread to avoid blocking the
    event loop for other requests while it runs.
    """
    try:
        await asyncio.to_thread(run_ingestion)
        manifest = load_manifest()
        return {"message": "Ingestion complete", "indexed_files": list(manifest.keys())}
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_database():
    """Wipe all cloud vectors from the Pinecone index and reset local manifest."""
    try:
        vectorstore.delete(delete_all=True)
        # Reset manifest file safely
        if os.path.exists(MANIFEST_PATH):
            os.remove(MANIFEST_PATH)
        return {"message": "Pinecone index and local manifest cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear Pinecone: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-document")
async def delete_document(filename: str):
    """Delete all vectors associated with a specific filename from Pinecone and manifest."""
    try:
        # Strip any path components so a crafted filename (e.g. "../../x")
        # can't escape the data directory.
        filename = os.path.basename(filename)
        manifest = load_manifest()

        if filename not in manifest:
            raise HTTPException(status_code=404, detail=f"{filename} not found in library")

        vector_ids = manifest[filename]

        if vector_ids:
            
            native_index = vectorstore.get_pinecone_index(INDEX_NAME)
            BATCH_SIZE = 1000  # defensive batching for large papers
            for i in range(0, len(vector_ids), BATCH_SIZE):
                native_index.delete(ids=vector_ids[i:i + BATCH_SIZE])
            logger.info(f"Deleted {len(vector_ids)} vectors for {filename}")
        else:
           
            logger.warning(
                f"No tracked vector IDs for {filename}; its vectors may "
                "still exist in Pinecone. Re-ingest after a /clear to fix."
            )

        del manifest[filename]
        write_manifest(manifest)

        local_file_path = f"./data/{filename}"
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

        return {"message": f"Successfully deleted {filename} from library and database."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
