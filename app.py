import torch
torch.classes.__path__ = []
import os
import tempfile
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from rag_system import RAGSystem

load_dotenv(find_dotenv())

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_OK = bool(SUPABASE_URL and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")))

st.set_page_config(
    page_title="Multi-Model Chat System | Crafted by Mubashir",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------
# CSS Styling for Minimalist Full-Canvas Chatbot Layout
# -------------------------------------------------------------------
st.markdown("""
<style>
:root{
    --bg:#0b0f14;
    --card:#11161d;
    --text:#f8fafc;
    --muted:#94a3b8;
    --border:#273244;
    --accent:#ef4444;
}

html, body, [class*="css"]{
    background: var(--bg) !important;
    color: var(--text) !important;
}

.stApp{
    background: linear-gradient(180deg, #0b0f14 0%, #111827 45%, #0b0f14 100%);
}

section[data-testid="stSidebar"]{
    background: #0f172a;
    border-right: 1px solid var(--border);
}

.block-container{
    max-width: 1000px;
    padding-top: 30px !important;
    padding-bottom: 0rem !important; 
}

.stButton > button {
    background: #11161d !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    background: #1c2330 !important;
}

div.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: white !important;
    border-color: #991b1b !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #dc2626 !important;
    border-color: #991b1b !important;
}

div[data-testid="stFormSubmitButton"] > th, div[data-testid="stSelectbox"] > div {
    margin-bottom: 0px !important;
}

div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border-bottom: 1px solid rgba(39, 50, 68, 0.4);
    padding: 20px 10px !important;
    border-radius: 0px !important;
}

.fixed-chat-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, transparent 0%, #0b0f14 30%);
    padding: 30px 0 20px 0;
    z-index: 99;
}

.fixed-chat-inner {
    max-width: 1000px;
    margin: 0 auto;
    padding: 0 20px;
}

[data-testid="stChatInput"] {
    border-top: none !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False
if "mode" not in st.session_state:
    st.session_state.mode = "Q&A"
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None

mode_mapping = {
    "Document": "User Document Mode",
    "Research": "Company Research Mode",
    "Q&A": "General Q&A Mode",
    "Web Search": "Web Search Mode",
    "Stock Analysis": "Financial / Stock Research Mode"
}

def initialize_rag_system():
    try:
        embedding_model = st.session_state.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        llm_model = st.session_state.get("llm_model", "nvidia/nemotron-3-ultra-550b-a55b:free")
        temperature = st.session_state.get("temperature", 0.2)
        base_url = st.session_state.get("base_url", "https://openrouter.ai/api/v1")
        st.session_state.rag_system = RAGSystem(
            model_name=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            base_url=base_url
        )
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False

def process_pdf(uploaded_file):
    try:
        if st.session_state.rag_system is None:
            if not initialize_rag_system():
                return False
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        rag_system = st.session_state.rag_system
        with st.spinner("Extracting text from PDF..."):
            text = rag_system.extract_text_from_pdf(tmp_path)
        if not text.strip():
            st.error("No text could be extracted from the PDF.")
            os.unlink(tmp_path)
            return False

        with st.spinner("Chunking and embedding text..."):
            chunk_size = st.session_state.get("chunk_size", 900)
            chunk_overlap = st.session_state.get("chunk_overlap", 150)
            chunks_data = rag_system.chunk_and_embed_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        storage_path = f"user-documents/{uploaded_file.name}"
        with st.spinner("Uploading PDF to Supabase..."):
            uploaded_ok = rag_system.upload_pdf_to_storage("user-documents", tmp_path, storage_path)
        if not uploaded_ok:
            st.error("Failed to upload PDF.")
            os.unlink(tmp_path)
            return False

        document_id = rag_system.insert_document_record(scope="user", report_type="annual_report", file_name=uploaded_file.name, storage_path=storage_path)
        if not document_id:
            st.error("Failed to insert record.")
            os.unlink(tmp_path)
            return False

        with st.spinner("Saving chunks..."):
            inserted_cnt = rag_system.insert_chunks_record(document_id=document_id, scope="user", chunks_data=chunks_data)

        if inserted_cnt is None:
            st.error("Failed to insert chunks.")
            os.unlink(tmp_path)
            return False

        st.session_state.current_document_id = document_id
        st.session_state.vector_store_loaded = True
        os.unlink(tmp_path)
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def main():
    # 1. Horizontal Mode Buttons Row
    modes = ["Document", "Research", "Q&A", "Web Search", "Stock Analysis"]
    
    st.write(" ") 
    mode_cols = st.columns([1, 1, 0.8, 1.1, 1.3, 0.2, 2.2])
    
    for idx, m in enumerate(modes):
        with mode_cols[idx]:
            is_active = (st.session_state.mode == m)
            if st.button(m, key=f"mode_tab_{m}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.mode = m
                st.rerun()

    st.divider()

    # 2. Contextual View Engine (Dropdown/Upload Layer placed cleanly under tabs)
    backend_mode_str = mode_mapping[st.session_state.mode]

    if st.session_state.mode == "Document" and not st.session_state.vector_store_loaded:
        uploaded_file = st.file_uploader("Upload reference knowledge base context PDF:", type="pdf", label_visibility="collapsed")
        if uploaded_file:
            if st.button("Process & Embed Document", use_container_width=True):
                if process_pdf(uploaded_file):
                    st.success("Document optimized successfully!")
                    st.rerun()
        st.write("---")

    elif st.session_state.mode == "Research":
        # Pull companies with bulletproof fallback arrays if database table is missing/empty
        companies = []
        if st.session_state.rag_system is None:
            initialize_rag_system()
            
        if st.session_state.rag_system:
            try:
                resp = st.session_state.rag_system.supabase.table("companies").select("id, symbol, name").order("name").execute()
                companies = resp.data or []
            except Exception:
                pass

        # If database table is unreachable or empty, use fallback array so the dropdown ALWAYS appears
        if not companies:
            companies = [
                {"id": 1, "name": "Apple Inc.", "symbol": "AAPL"},
                {"id": 2, "name": "Microsoft Corp.", "symbol": "MSFT"},
                {"id": 3, "name": "NVIDIA Corp.", "symbol": "NVDA"},
                {"id": 4, "name": "Google LLC", "symbol": "GOOGL"}
            ]

        options = [f"{c['name']} ({c['symbol']})" for c in companies]
        
        current_idx = 0
        if st.session_state.current_company:
            curr_label = f"{st.session_state.current_company['name']} ({st.session_state.current_company['symbol']})"
            if curr_label in options:
                current_idx = options.index(curr_label)
        else:
            st.session_state.current_company = companies[0]

        # The Dropdown Selectbox is explicitly rendered here no matter what!
        selected_label = st.selectbox(
            "Select Target Company Dossier:", 
            options=options, 
            index=current_idx,
            key="header_research_selectbox"
        )
        
        new_selection = companies[options.index(selected_label)]
        if st.session_state.current_company != new_selection:
            st.session_state.current_company = new_selection
            st.rerun()
            
        st.write("---")

    # 3. Canvas Chat Render Flow Engine
    for message in st.session_state.chat_history:
        role = message["role"]
        avatar = "🫵🏽" if role == "user" else "🧟"
        with st.chat_message(role, avatar=avatar):
            if role == "user":
                st.markdown(f"**You**\n\n{message['content']}")
            else:
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}**")
                            st.write(source.page_content)

    # 4. Clean, Fixed Base Prompt Area Interface Engine
    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🫵🏽"):
            st.markdown(f"**You**\n\n{prompt}")

        with st.chat_message("assistant", avatar="🧟"):
            placeholder = st.empty()
            try:
                if st.session_state.rag_system is None:
                    initialize_rag_system()

                if backend_mode_str == "Company Research Mode":
                    if not st.session_state.current_company:
                        placeholder.warning("Please configure system company dossier targets above first.")
                        return
                    result = st.session_state.rag_system.query_company_documents(st.session_state.current_company["id"], prompt, top_k=st.session_state.get("k_results", 3))
                elif backend_mode_str == "User Document Mode":
                    if not st.session_state.vector_store_loaded:
                        placeholder.warning("Please upload reference document content vectors to system memory profiles first.")
                        return
                    result = st.session_state.rag_system.query_user_document(st.session_state.current_document_id, prompt, top_k=st.session_state.get("k_results", 3))
                elif backend_mode_str == "General Q&A Mode":
                    result = st.session_state.rag_system.query_general_question(prompt)
                elif backend_mode_str == "Web Search Mode":
                    result = st.session_state.rag_system.query_web_search(prompt)
                else:
                    result = st.session_state.rag_system.query_financial_analysis(prompt)

                answer = result["answer"]
                sources = result.get("source_documents", [])
                placeholder.markdown(answer)
                
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()
            except Exception as e:
                placeholder.error(f"Error generating engine response sequences: {str(e)}")

    # 5. Hidden Core Global Configurations Controls Side Drawer
    with st.sidebar:
        st.markdown("### Advanced Core Systems Controls")
        with st.expander("Model Configuration Framework Tuning"):
            st.session_state["embedding_model"] = st.selectbox("Embedding Model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"])
            st.session_state["llm_model"] = st.selectbox("LLM Model", ["nvidia/nemotron-3-ultra-550b-a55b:free", "openai/gpt-4o-mini"])
            st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
            st.session_state["chunk_size"] = st.slider("Chunk Size", 200, 2000, 900, 100)
            st.session_state["chunk_overlap"] = st.slider("Chunk Overlap", 0, 500, 150, 50)
            st.session_state["k_results"] = st.slider("Retrieved Chunks Count", 1, 10, 3)

        if st.button("Reset Global Chat State Containers", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.vector_store_loaded = False
            st.session_state.current_company = None
            st.session_state.current_document_id = None
            st.rerun()

if __name__ == "__main__":
    main()
