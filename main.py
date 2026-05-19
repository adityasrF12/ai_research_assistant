import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import shutil


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG-API")

app = FastAPI(title="Research Assistant API")


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "research_db")


logger.info("Initializing Embeddings and Vector Store...")
embeddings = OllamaEmbeddings(
    model="llama3",
    base_url=OLLAMA_URL
)

vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

template = """You are a research assistant managing a library of technical papers. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOllama(model="llama3", temperature=0, base_url=OLLAMA_URL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

class Query(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "Online", "db_connected": os.path.exists(db_path)}

@app.post("/ask")
async def ask_question(query: Query):
    logger.info(f"Processing query: {query.text}")
    try:
        response = qa_chain.invoke({"query": query.text})
        
        
        sources = [
            {
                "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content[:150] + "..."
            } 
            for doc in response["source_documents"]
        ]
        
        logger.info(f"Successfully generated answer using {len(sources)} sources.")
        return {
            "answer": response["result"],
            "sources": sources
        }
    except Exception as e:
        logger.error(f"RAG Chain Error: {str(e)}")
        return {"error": "Processing failed", "details": str(e)}

@app.delete("/clear")
async def clear_database():
    global vectorstore, qa_chain
    logger.warning("Request received to clear the vector database.")
    
    try:
        if vectorstore is not None:
            # 1. Delete the specific collection
            vectorstore.delete_collection()
            logger.info("Collection deleted.")
            
            # 2. Immediately re-create it so the ID is valid for the next query
            vectorstore = Chroma(
                persist_directory=db_path, 
                embedding_function=embeddings
            )
            
            # 3. Refresh the QA chain with the new (empty) vectorstore
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            
            logger.info("Empty collection re-initialized.")
    
    except Exception as e:
        logger.error(f"Failed to clear database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)