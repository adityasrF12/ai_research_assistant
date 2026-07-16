import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv(override=True)

# 1. Setup Configuration
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
MANIFEST_PATH = "./data/manifest.json"


def load_manifest():
    """
    Manifest is now a dict: {filename: [vector_id, vector_id, ...]}

    We store the exact Pinecone vector IDs generated for each file at ingest
    time. This is required because Pinecone *serverless* indexes do not
    support delete-by-metadata-filter -- only delete-by-id. Without tracking
    IDs here, there is no reliable way to later remove just one paper's
    vectors from the index.
    """
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            data = json.load(f)
            
            if isinstance(data, list):
                return {name: [] for name in data}
            return data
    return {}


def save_to_manifest(filename, vector_ids):
    manifest = load_manifest()
    manifest[filename] = vector_ids
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)


def run_ingestion():
    manifest = load_manifest()

    # Check if there are any new PDFs to process
    all_pdfs = [f for f in os.listdir('./data') if f.endswith('.pdf')]
    new_pdfs = [f for f in all_pdfs if f not in manifest]

    if not new_pdfs:
        print("No new documents to ingest.")
        return

    print(f"--- Found {len(new_pdfs)} new PDF(s) to ingest ---")

    # 2. Split and process files one by one to isolate metadata state
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)

    
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

    for pdf in new_pdfs:
        print(f"Processing: {pdf}")
        loader = PyPDFLoader(os.path.join('./data', pdf))
        data = loader.load()
        chunks = text_splitter.split_documents(data)

        
        for chunk in chunks:
            chunk.metadata["source"] = pdf

        
        vector_ids = [f"{pdf}::chunk-{i}" for i in range(len(chunks))]

        print(f"Pushing {len(chunks)} chunks to Pinecone...")
        vectorstore.add_documents(documents=chunks, ids=vector_ids)

        # Save to manifest immediately on success, including the IDs
        save_to_manifest(pdf, vector_ids)
        print(f"Successfully indexed and recorded {pdf}")

    print("Ingestion Complete!")


if __name__ == "__main__":
    run_ingestion()
