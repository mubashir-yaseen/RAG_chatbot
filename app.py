import os
import tempfile
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from rag_system import RAGSystem

load_dotenv(find_dotenv())

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_OK = bool(SUPABASE_URL and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")))

st.set_page_config(
    page_title="Mubashir & Hassan | RAG Chat System",
    page_icon="🧟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
:root{
    --bg:#f6f8fb;
    --card:#ffffff;
    --text:#0f172a;
    --muted:#64748b;
    --border:#e2e8f0;
    --accent:#14532d;
    --accent2:#1b5e20;
    --user:#e8f3ea;
    --assistant:#0f172a;
    --assistantText:#ffffff;
}
html, body, [class*="css"]{
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp{
    background: linear-gradient(180deg, #f7faf7 0%, #f6f8fb 48%, #f6f8fb 100%);
}
section[data-testid="stSidebar"]{
    background: #ffffff;
    border-right: 1px solid var(--border);
}
.block-container{
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.hero{
    background: linear-gradient(135deg, #0f172a 0%, #14532d 55%, #1b5e20 100%);
    color: white;
    border-radius: 24px;
    padding: 28px 28px 22px 28px;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
}
.hero h1{
    font-size: 2.05rem;
    margin: 0;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.hero p{
    margin: 10px 0 0 0;
    color: rgba(255,255,255,0.82);
    font-size: 0.98rem;
    line-height: 1.5;
}
.pill-row{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    margin-top:14px;
}
.pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    color: white;
    font-size: 0.85rem;
    border: 1px solid rgba(255,255,255,0.14);
}
.section-card{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(15,23,42,0.04);
    margin-bottom: 16px;
}
.section-title{
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 10px;
}
.helper{
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
}
.status{
    display:inline-block;
    padding: 0.36rem 0.75rem;
    border-radius: 999px;
    background: #ecfdf3;
    color: #166534;
    font-weight: 700;
    font-size: 0.82rem;
    border: 1px solid #bbf7d0;
}
.company-badge{
    display:inline-block;
    padding: 0.36rem 0.75rem;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-weight: 700;
    font-size: 0.82rem;
    border: 1px solid #bfdbfe;
}
small, .small-note{
    color: var(--muted) !important;
}
div[data-testid="stChatMessage"]{
    border-radius: 18px;
}
div[data-testid="stChatMessage"][aria-label="assistant"]{
    background: transparent;
}
div[data-testid="stChatMessage"][aria-label="user"]{
    background: transparent;
}
[data-testid="stChatInput"]{
    border-top: 1px solid var(--border);
    padding-top: 12px;
    margin-top: 0.5rem;
    background: rgba(255,255,255,0.96);
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] div,
div[data-testid="stChatMessage"] span{
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False
if "mode" not in st.session_state:
    st.session_state.mode = "User Document Mode"
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None
if "company_options" not in st.session_state:
    st.session_state.company_options = []

def initialize_rag_system():
    try:
        embedding_model = st.session_state.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        llm_model = st.session_state.get("llm_model", "openai/gpt-3.5-turbo")
        temperature = st.session_state.get("temperature", 0.3)
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
            chunk_size = st.session_state.get("chunk_size", 1000)
            chunk_overlap = st.session_state.get("chunk_overlap", 200)
            chunks_data = rag_system.chunk_and_embed_document(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        storage_path = f"user-documents/{uploaded_file.name}"
        with st.spinner("Uploading PDF to Supabase Storage..."):
            uploaded_ok = rag_system.upload_pdf_to_storage("user-documents", tmp_path, storage_path)

        if not uploaded_ok:
            st.error("Failed to upload PDF to Supabase storage.")
            os.unlink(tmp_path)
            return False

        document_id = rag_system.insert_document_record(
            scope="user",
            report_type="annual_report",
            file_name=uploaded_file.name,
            storage_path=storage_path
        )

        if not document_id:
            st.error("Failed to insert document record into Supabase.")
            os.unlink(tmp_path)
            return False

        with st.spinner("Saving chunks to database..."):
            inserted_cnt = rag_system.insert_chunks_record(
                document_id=document_id,
                scope="user",
                chunks_data=chunks_data
            )

        if inserted_cnt is None:
            st.error("Failed to insert chunks into Supabase.")
            os.unlink(tmp_path)
            return False

        st.session_state.current_document_id = document_id
        st.session_state.vector_store_loaded = True
        st.session_state.current_pdf = uploaded_file.name
        os.unlink(tmp_path)
        return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def load_companies():
    try:
        if st.session_state.rag_system is None:
            if not initialize_rag_system():
                return []
        resp = st.session_state.rag_system.supabase.table("companies").select("id, symbol, name, sector").order("name").execute()
        return resp.data or []
    except Exception:
        return []

def select_company_ui():
    companies = load_companies()
    st.session_state.company_options = companies

    if not companies:
        st.info("No companies found in the database yet.")
        return

    options = [f"{c['name']} ({c['symbol']})" for c in companies]
    current_label = None
    if st.session_state.current_company:
        current_label = f"{st.session_state.current_company['name']} ({st.session_state.current_company['symbol']})"
    default_index = options.index(current_label) if current_label in options else 0

    selected_label = st.selectbox("Choose a company", options, index=default_index, label_visibility="collapsed")
    selected_company = companies[options.index(selected_label)]

    if st.session_state.current_company != selected_company:
        st.session_state.current_company = selected_company
        st.session_state.current_document_id = None
        st.session_state.chat_history = []

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"<span class='company-badge'>{selected_company['name']} ({selected_company['symbol']})</span>", unsafe_allow_html=True)
    with c2:
        if selected_company.get("sector"):
            st.markdown(f"<span class='status'>Sector: {selected_company['sector']}</span>", unsafe_allow_html=True)

def render_chat_history():
    for message in st.session_state.chat_history:
        role = message["role"]
        avatar = "🫵🏽" if role == "user" else "🧟"
        with st.chat_message(role, avatar=avatar):
            if role == "user":
                st.markdown(f"**You**\n\n{message['content']}")
            else:
                st.markdown(
                    f"<div style='color:#0f172a; background:transparent;'>{message['content']}</div>",
                    unsafe_allow_html=True
                )
                if message.get("sources"):
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}**")
                            st.write(source.page_content)

def chat_interface():
    if st.session_state.mode == "Company Research Mode":
        if not st.session_state.current_company:
            st.info("Please select a company first.")
            return
    else:
        if not st.session_state.vector_store_loaded:
            st.info("Please upload and process a PDF first.")
            return

    render_chat_history()

    prompt = st.chat_input("Ask a question about your document or company report...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🫵🏽"):
            st.markdown(f"**You**\n\n{prompt}")

        with st.chat_message("assistant", avatar="🧟"):
            placeholder = st.empty()
            try:
                if st.session_state.rag_system is None:
                    if not initialize_rag_system():
                        return

                if st.session_state.mode == "Company Research Mode":
                    result = st.session_state.rag_system.query_company_documents(
                        st.session_state.current_company["id"], prompt,
                        top_k=st.session_state.get("k_results", 3)
                    )
                else:
                    result = st.session_state.rag_system.query_user_document(
                        st.session_state.current_document_id, prompt,
                        top_k=st.session_state.get("k_results", 3)
                    )

                answer = result["answer"]
                sources = result.get("source_documents", [])

                placeholder.markdown(
                    f"<div style='color:#0f172a; background:transparent;'>{answer}</div>",
                    unsafe_allow_html=True
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
            except Exception as e:
                placeholder.error(f"Error generating response: {str(e)}")

def main():
    st.markdown("""
    <div class="hero">
        <h1>Mubashir & Hassan RAG Chat System</h1>
	<p>A document intelligence workspace for uploading PDFs, researching company reports, and chatting with your knowledge base.</p>
        <p>🪦Wait a minute!! who are you?</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("rag_system") is not None:
        supabase_connected = getattr(st.session_state.rag_system, "supabase_ok", False)
    else:
        supabase_connected = SUPABASE_OK

    with st.sidebar:
        st.markdown("### Controls")
        st.session_state.mode = st.selectbox(
            "Mode",
            ["User Document Mode", "Company Research Mode"],
            index=0 if st.session_state.mode == "User Document Mode" else 1,
            label_visibility="collapsed"
        )

        if st.session_state.mode == "Company Research Mode" and st.session_state.current_company:
            st.markdown(f"<span class='company-badge'>{st.session_state.current_company['symbol']}</span>", unsafe_allow_html=True)

        with st.expander("Advanced Settings"):
            st.session_state["embedding_model"] = st.selectbox(
                "Embedding Model",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-MiniLM-L6-v2"
                ],
                index=0
            )

            st.session_state["llm_model"] = st.selectbox(
                "LLM Model",
                ["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-haiku", "meta-llama/llama-3-8b-instruct"],
                index=0
            )

            st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            st.session_state["chunk_size"] = st.slider("Chunk Size (characters)", 200, 2000, 1000, 100)
            st.session_state["chunk_overlap"] = st.slider("Chunk Overlap (characters)", 0, 500, 200, 50)
            st.session_state["k_results"] = st.slider("Number of Retrieved Chunks", 1, 10, 3)

        if st.button("Initialize RAG System"):
            if not initialize_rag_system():
                st.error("RAG system failed to initialize.")
            else:
                st.success("RAG system initialized.")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.vector_store_loaded = False
            st.session_state.current_company = None
            st.session_state.current_document_id = None
            st.rerun()

    if not supabase_connected:
        st.warning("Supabase is not connected. Check your .env file and restart the app.")

    col_left, col_right = st.columns([0.92, 1.08], gap="large")

    with col_left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Document Mode</div>", unsafe_allow_html=True)

        if st.session_state.mode == "Company Research Mode":
            if not st.session_state.rag_system:
                if not initialize_rag_system():
                    st.stop()

            st.markdown("#### Company Selection")
            select_company_ui()

            st.divider()
            if st.session_state.current_company:
                docs = st.session_state.rag_system.get_documents_for_company(st.session_state.current_company["id"])
                if docs:
                    st.markdown("**Available company reports**")
                    for d in docs:
                        st.write(f"• {d.get('file_name')}  —  {d.get('year')}")
                else:
                    st.info("No company reports found. Use the backend uploader to add reports.")
        else:
            uploaded_file = st.file_uploader("Upload a PDF", type="pdf", help="Upload a document to create a searchable knowledge base")
            if uploaded_file:
                if not st.session_state.rag_system:
                    if not os.environ.get("OPENAI_API_KEY"):
                        st.error("Please set your API key in the sidebar first.")
                    else:
                        if not initialize_rag_system():
                            st.stop()

                if st.button("Process PDF"):
                    process_pdf(uploaded_file)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Chat</div>", unsafe_allow_html=True)
        st.markdown("<div class='helper'>Ask questions in plain English. Answers are grounded in your uploaded documents or selected company reports.</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        chat_interface()
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
