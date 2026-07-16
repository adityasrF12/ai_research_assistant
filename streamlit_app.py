import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Research Assistant", 
    # page_icon="📚", 
    layout="wide"
)


# --- Sidebar: Document Management ---
st.sidebar.header("Document Management")

# Fetch persistent library from FastAPI backend
indexed_files = []
try:
    lib_response = requests.get("http://localhost:8000/library")
    if lib_response.status_code == 200:
        indexed_files = lib_response.json().get("indexed_files", [])
except Exception:
    pass  # Backend container might be spinning up

# Display Active Library
# Display Active Library with individual delete options
if indexed_files:
    st.sidebar.markdown("### Active Research Library")
    for file in indexed_files:
        # Create two columns: one for the filename, one for a delete button
        col1, col2 = st.sidebar.columns([4, 1])
        col1.caption(f"✓ {file}")
        
        
        if col2.button("🗑️", key=f"del_{file}", help=f"Delete {file}"):
            with st.sidebar.spinner(f"Removing {file}..."):
                res = requests.delete(f"http://localhost:8000/delete-document?filename={file}")
                if res.status_code == 200:
                    st.sidebar.success(f"Deleted {file}")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to delete.")
    st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files to add", 
    type="pdf", 
    accept_multiple_files=True
)

if st.sidebar.button("Process & Ingest"):
    if uploaded_files:
        with st.sidebar.status("Processing documents...", expanded=True) as status:
            if not os.path.exists("./data"):
                os.makedirs("./data")
            
            st.write("Saving files to disk...")
            for f in uploaded_files:
                file_path = os.path.join("./data", f.name)
                with open(file_path, "wb") as buffer:
                    buffer.write(f.getvalue())
            
            st.write("Requesting backend to embed new papers...")
            try:
                # ./data is a shared Docker volume, so the files saved
                # above are already visible inside the API container.
                # We just ask it to scan for and embed anything new --
                # this always runs with the container's environment,
                # unlike the old approach of shelling out to a local venv.
                response = requests.post("http://localhost:8000/ingest", timeout=600)

                if response.status_code == 200:
                    status.update(label="Ingestion Complete!", state="complete", expanded=False)
                    st.sidebar.success("Pinecone Index Updated!")
                    st.rerun() # Force layout rerun to display newly added files instantly
                else:
                    status.update(label="Ingestion Failed", state="error")
                    st.sidebar.error(f"Error: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Failed to reach ingestion endpoint: {e}")
    else:
        st.sidebar.warning("Please upload at least one PDF first.")

# --- Main Interface ---
st.title("AI Research Assistant")
st.info("Query your local research library. Results are searched via Pinecone and reranked for accuracy.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 View Evidence & Citations"):
                for src in message["sources"]:
                    st.write(f"📄 **File:** {src['file']} | **Page:** {src['page'] + 1}")
                    st.caption(f"Snippet: {src['snippet']}")

# Chat Input
if prompt := st.chat_input("Ask a research question..."):
    # Add user message to history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing papers and reranking results..."):
            try:
                # Talking to your FastAPI Docker container
                response = requests.post("http://localhost:8000/ask", json={"text": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources
                    if sources:
                        with st.expander("View Evidence & Citations"):
                            for src in sources:
                                st.write(f"📄 **File:** {src['file']} | **Page:** {src['page'] + 1}")
                                st.caption(f"Snippet: {src['snippet']}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")