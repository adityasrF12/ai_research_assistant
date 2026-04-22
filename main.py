import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
# from langchain.chains import RetrievalQA # Updated import
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="Research Assistant API")

# Initialize RAG

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "research_db")

embeddings = OllamaEmbeddings(
    model="llama3",
    base_url="http://localhost:11434" # Explicitly point to local Ollama
)

vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

template = """You are a research assistant managing a library of technical papers.
Context: {context}
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOllama(model="llama3", temperature=0)

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
def read_root():
    return {"status": "Online", "database": db_path}

@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = qa_chain.invoke({"query": query.text})
        
        sources = [
            {"page": doc.metadata.get("page"), "content": doc.page_content[:100]} 
            for doc in response["source_documents"]
        ]
        
        return {
            "answer": response["result"],
            "sources": sources
        }
    except Exception as e:
        return {"error": "RAG Chain failed", "details": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)