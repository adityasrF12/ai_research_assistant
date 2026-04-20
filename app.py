import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader


llm = ChatOllama(model="llama3", temperature=0)
embeddings = OllamaEmbeddings(model="llama3")


loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=300,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(data)


vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./research_db"
)


template = """You are a research assistant managing a library of technical papers.
Your goal is to synthesize an answer based on the provided snippets. 

1. If different snippets discuss different papers, summarize the key points from each.
2. If the snippets are technical examples (like math problems), explain what concept they are illustrating.
3. If the context is insufficient to answer fully, provide the best summary possible from what is available.

Context: {context}

Question: {question}

Final Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 8, 'fetch_k': 20}
),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
)


query = "What are the main findings across all the research papers provided?"

response = qa_chain.invoke({"query": query})

print("\n--- ANSWER ---")
print(response["result"])

print("\n--- SOURCES USED ---")
for doc in response["source_documents"]:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100].replace(os.linesep, ' ')}...")