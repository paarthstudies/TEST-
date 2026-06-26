import streamlit as st
import requests

# Constants
FASTAPI_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"

st.set_page_config(page_title="Deep Hybrid Cognition RAG", layout="wide")

# Custom Cyberpunk CSS Injections
st.markdown("""
<style>
    /* Dark Cyberpunk Theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Glowing accents */
    .neon-text-cyan {
        color: #00e5ff;
        text-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff;
    }
    .neon-text-purple {
        color: #7b2cbf;
        text-shadow: 0 0 5px #7b2cbf, 0 0 10px #7b2cbf;
    }
    .neon-border {
        border: 1px solid #00e5ff;
        box-shadow: 0 0 8px #00e5ff;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Header Styling */
    h1 {
        color: #00e5ff !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px #00e5ff;
        border-bottom: 2px solid #7b2cbf;
        padding-bottom: 10px;
    }

    /* Success Badge */
    .success-badge {
        background-color: rgba(0, 229, 255, 0.1);
        border: 1px solid #00e5ff;
        color: #00e5ff;
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        box-shadow: 0 0 10px #00e5ff;
        margin-top: 10px;
    }

    /* Chat Messages */
    .stChatMessage {
        border: 1px solid #7b2cbf;
        box-shadow: 0 0 8px #7b2cbf;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: rgba(123, 44, 191, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# Left Sidebar
# -----------------
with st.sidebar:
    st.markdown("<h2 class='neon-text-cyan'>System Status</h2>", unsafe_allow_html=True)
    
    def check_health(url):
        try:
            response = requests.get(url, timeout=2)
            return response.status_code == 200 or response.status_code == 404 # 404 means server is up but root is not found
        except:
            return False

    fastapi_up = check_health(FASTAPI_URL + "/docs")
    ollama_up = check_health(OLLAMA_URL)

    st.markdown(f"**FastAPI Backend:** {'✅ ONLINE' if fastapi_up else '❌ OFFLINE'}")
    st.markdown(f"**Ollama Engine:** {'✅ ONLINE' if ollama_up else '❌ OFFLINE'}")
    
    st.markdown("---")
    st.markdown("<h2 class='neon-text-purple'>Data Ingestion</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "csv", "xlsx"])
    if uploaded_file is not None:
        if st.button("Process Data"):
            with st.spinner("Ingesting into LanceDB..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{FASTAPI_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.markdown(f"<div class='success-badge'>UPLOAD SUCCESS // {data['chunks_processed']} CHUNKS PROCESSED</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# -----------------
# Main Interface Dashboard
# -----------------
st.markdown("<h1>CORE SYSTEM // DEEP HYBRID COGNITION RAG</h1>", unsafe_allow_html=True)

# Dynamic Query Customization Panel
st.markdown("<h3 class='neon-text-cyan'>Query Configuration</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    target_mode = st.radio("Target Mode", options=["Tabular Point-Lookup", "Standard Semantic Document"])
    source_type = "tabular" if target_mode == "Tabular Point-Lookup" else "standard"

with col2:
    st.markdown("#### Hybrid Search Ratio")
    if source_type == "tabular":
        st.progress(30, text="30% Dense / 70% BM25")
    else:
        st.progress(60, text="60% Dense / 40% BM25")

st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Source Inspector"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Document:** {src.get('source', 'Unknown')}")
                    st.markdown(f"**Chunk:** {src.get('text', '')}")
                    # You might also want to display similarity scores if they are returned by FastAPI
                    st.markdown("---")

# Interactive Conversational Chat Console
if prompt := st.chat_input("Enter your query here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Computing...")
        
        try:
            # Send all messages EXCEPT the current prompt
            history_payload = st.session_state.messages[:-1]
            payload = {
                "query": prompt,
                "source_type": source_type,
                "chat_history": history_payload
            }
            response = requests.post(f"{FASTAPI_URL}/chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                reply = data.get("response", "No response")
                sources = data.get("sources", [])
                
                message_placeholder.markdown(reply)
                
                if sources:
                    with st.expander("🔍 Source Inspector"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"**Document:** {src.get('source', 'Unknown')}")
                            st.markdown(f"**Chunk:** {src.get('text', '')}")
                            st.markdown("---")
                            
                st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")
